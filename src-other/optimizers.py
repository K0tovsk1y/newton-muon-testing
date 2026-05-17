import torch
from torch.optim import Optimizer

@torch.compile
def newton_schulz(G: torch.Tensor, steps: int = 5):
    X = G.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    
    m, n = X.shape
    # Динамический выбор стороны умножения для избежания OOM на вытянутых матрицах
    if m > n:
        for _ in range(steps):
            A = X.T @ X             # [n, n]
            B = b * A + c * (A @ A) # [n, n]
            X = a * X + X @ B       # [m, n] @ [n, n] -> [m, n]
    else:
        for _ in range(steps):
            A = X @ X.T             #[m, m]
            B = b * A + c * (A @ A) # [m, m]
            X = a * X + B @ X       # [m, m] @ [m, n] -> [m, n]
            
    return X.to(G.dtype)


class MatrixOptimizer(Optimizer):
    def __init__(self, params, lr=0.002, momentum=0.95, nesterov=True, weight_decay=0.01, 
                 use_newton=False, nm_beta=0.95, nm_gamma=0.2, nm_refresh_interval=32):
        defaults = dict(
            lr=lr, 
            momentum=momentum, 
            nesterov=nesterov,
            weight_decay=weight_decay, 
            use_newton=use_newton,
            nm_beta=nm_beta,       # Сохранение истории (EMA)
            nm_gamma=nm_gamma,     # Ridge penalty для стабильности
            nm_refresh_interval=nm_refresh_interval
        )
        super().__init__(params, defaults)
        self.missing_z_warned = False
        self._step_count = 0

    @torch.no_grad()
    def step(self):
        metrics = {"cond_nums":[]}
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]
                
                # Инициализация состояния (безопасная проверка по ключам)
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                
                if group['use_newton'] and 'accum_K' not in state:
                    state['K'] = None
                    state['inv_K'] = None
                    state['accum_K'] = None
                    state['accum_count'] = 0
                    state['precond_warmed_up'] = False

                m, n = p.shape

                # =====================================================
                # БЛОК NEWTON-MUON
                # =====================================================
                if group['use_newton']:
                    if getattr(p, 'Z', None) is None:
                        if not self.missing_z_warned:
                            print("\n[WARNING] Newton-Muon: no 'Z' attribute! Falling back to Muon.")
                            self.missing_z_warned = True
                    else:
                        Z = p.Z.to(torch.float32)
                        
                        if Z.ndim > 2:
                            Z = Z.view(-1, Z.size(-1)) if Z.size(-1) == n else Z.view(n, -1)

                        if Z.size(0) == n:    
                            ZZT = torch.mm(Z, Z.T) / Z.size(1)
                        elif Z.size(1) == n:  
                            ZZT = torch.mm(Z.T, Z) / Z.size(0)
                        else:
                            raise ValueError(f"Shape mismatch: param is {p.shape}, Z is {p.Z.shape}")
                        
                        # 1. Накопление ZZT внутри интервала (БЕЗ применения EMA)
                        if state['accum_K'] is None:
                            state['accum_K'] = torch.zeros_like(ZZT)
                        
                        state['accum_K'].add_(ZZT)
                        state['accum_count'] += 1

                        # 2. Инициализация K и inv_K на самом первом шаге
                        if state['K'] is None:
                            state['K'] = 1e-3 * torch.eye(n, device=grad.device, dtype=torch.float32)
                            # Изначально предобуславливатель - это единичная матрица (действует как обычный Muon)
                            state['inv_K'] = torch.eye(n, device=grad.device, dtype=grad.dtype)

                        # 3. Обновление EMA и обратной матрицы ТОЛЬКО раз в refresh_interval
                        if (self._step_count + 1) % group['nm_refresh_interval'] == 0:
                            beta = group['nm_beta']
                            
                            # Усредненный ZZT за последние k шагов
                            avg_ZZT = state['accum_K'] / max(state['accum_count'], 1)
                            
                            # Применяем EMA один раз за интервал (согласно Алгоритму 1)
                            state['K'].lerp_(avg_ZZT, 1.0 - beta)
                            
                            # Сброс аккумуляторов
                            state['accum_K'].zero_()
                            state['accum_count'] = 0

                            K = state['K']
                            trace = torch.trace(K)
                            
                            gamma_val = group['nm_gamma'] * (trace / n) + 1e-8
                            K_damped = K + gamma_val * torch.eye(n, device=K.device)
                            
                            # 4. БЕЗОПАСНЫЙ Cholesky Inverse (со 100% совпадением с оф. кодом)
                            try:
                                L, info = torch.linalg.cholesky_ex(K_damped)
                                if info.item() == 0:
                                    inv_K = torch.cholesky_inverse(L).to(grad.dtype)
                                else:
                                    # FALLBACK: Возвращаемся к единичной матрице, если сингулярна!
                                    inv_K = torch.eye(n, device=grad.device, dtype=grad.dtype)
                            except Exception:
                                inv_K = torch.eye(n, device=grad.device, dtype=grad.dtype)
                                
                            state['inv_K'] = inv_K
                            
                            # Сбор метрик (опционально)
                            try:
                                L_eig = torch.linalg.eigvalsh(K_damped.to(torch.float64))
                                state['last_cond'] = (L_eig[-1] / torch.clamp(L_eig[0], min=1e-15)).item()
                            except:
                                state['last_cond'] = float('inf')

                        # Применение предобуславливателя (если он готов)
                        if state['inv_K'] is not None:
                            grad = (grad.to(torch.float32) @ state['inv_K'].to(torch.float32)).to(grad.dtype)
                            
                        if 'last_cond' in state:
                            metrics["cond_nums"].append(state['last_cond'])


                # =====================================================
                # СТАНДАРТНЫЙ MUON PIPELINE
                # =====================================================
                
                # 1. Decoupled Weight Decay
                if group['weight_decay'] != 0:
                    p.data.mul_(1.0 - group['lr'] * group['weight_decay'])

                # 2. Классический Heavy-Ball Momentum
                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(grad)
                
                if group['nesterov']:
                    grad = grad.add(buf, alpha=group['momentum'])
                else:
                    grad = buf.clone()
                
                # 3. Ортогонализация градиента
                update = newton_schulz(grad)
                
                # 4. Правильный Scaling Learning Rate
                scale = max(m, n) ** 0.5
                eff_lr = group['lr'] * scale
                
                # Применение шага
                p.data.add_(update, alpha=-eff_lr)
                
        self._step_count += 1
        return metrics
