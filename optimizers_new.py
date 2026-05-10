import torch
import math
from torch.optim import Optimizer

@torch.compile
def newton_schulz(G: torch.Tensor, steps: int = 5):
    X = G.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    
    m, n = X.shape
    if m > n:
        for _ in range(steps):
            A = X.T @ X             
            B = b * A + c * (A @ A) 
            X = a * X + X @ B       
    else:
        for _ in range(steps):
            A = X @ X.T             
            B = b * A + c * (A @ A) 
            X = a * X + B @ X       
            
    return X.to(G.dtype)


class MatrixOptimizer(Optimizer):
    def __init__(self, params, lr=0.002, momentum=0.95, nesterov=True, weight_decay=0.01, 
                 use_newton=False, nm_beta=0.95, nm_gamma=0.2, nm_update_threshold=0.01, nm_refresh_interval=6,new_gamma=False,new_interval=False):
        defaults = dict(
            lr=lr, 
            momentum=momentum, 
            nesterov=nesterov,
            weight_decay=weight_decay, 
            use_newton=use_newton,
            nm_beta=nm_beta,       
            nm_gamma=nm_gamma,     
            nm_update_threshold=nm_update_threshold,
            nm_refresh_interval=nm_refresh_interval,
            new_gamma=new_gamma,
            new_interval=new_interval
        )
        super().__init__(params, defaults)
        self.missing_z_warned = False

    @torch.no_grad()
    def step(self):
        metrics = {"cond_nums":[]}
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]
                
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                
                # Инициализация трекеров для статистики
# Инициализация трекеров для статистики
                if group['use_newton'] and 'step_count' not in state:
                    state['K'] = None
                    state['inv_K'] = None
                    state['accum_K'] = None
                    state['accum_count'] = 0
                    state['step_count'] = 0
                    state['accumulated_path'] = 0.0 
                    state['update_count'] = 0       # Считаем количество вызовов Холецкого
                    state['gamma_frob'] = 0.0       # Последний посчитанный наш Gamma
                    state['gamma_trace'] = 0.0      # Какой был бы Gamma по статье
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
                        
# 1. Накопление ZZT внутри интервала
                        if state['accum_K'] is None:
                            state['accum_K'] = torch.zeros_like(ZZT)
                        
                        state['accum_K'].add_(ZZT)
                        state['accum_count'] += 1

                        # 2. Инициализация K и inv_K на самом первом шаге (Identity Warmup)
                        if state['K'] is None:
                            state['K'] = 1e-3 * torch.eye(n, device=grad.device, dtype=torch.float32)
                            state['inv_K'] = torch.eye(n, device=grad.device, dtype=grad.dtype)

                        # 3. Правильная проверка интервала по шагам
                        # weight_norm = torch.linalg.matrix_norm(p.data, ord='fro').item() + 1e-8
                        relative_path = state['accumulated_path'] #/ weight_norm
                        
                        # Не обновляем на нулевом шаге (храним warmup). Обновляем, когда прошли нужный путь.
                        if(group["new_interval"]):
                            do_update = (state['step_count'] > 0) and (relative_path > group['nm_update_threshold'])
                        else:
                            do_update = ((state['step_count'] + 1) % group["nm_refresh_interval"] == 0)

                        if do_update:
                            # Сбрасываем таймер пути
                            state['accumulated_path'] = 0.0
                            beta = group['nm_beta']
                            
                            # Усредненный ZZT за последние k шагов
                            avg_ZZT = state['accum_K'] / max(state['accum_count'], 1)
                            
                            # Применяем EMA к усредненной матрице
                            state['K'].lerp_(avg_ZZT, 1.0 - beta) 
                            
                            # Сброс аккумуляторов
                            state['accum_K'].zero_()
                            state['accum_count'] = 0

                            K = state['K']
                            
                            # === РАСЧЕТ ДВУХ ГАММ ДЛЯ АНАЛИТИКИ ===
                            trace = torch.trace(K)
                            frob_norm = torch.linalg.matrix_norm(K, ord='fro')
                            
                            # Как в статье (для аналитики):
                            gamma_trace = group['nm_gamma'] * (trace / n) + 1e-8
                            # Как правильно (используем для дела):
                            gamma_frob = group['nm_gamma'] * (frob_norm / math.sqrt(n)) + 1e-8
                            
                            # Сохраняем в стейт
                            state['gamma_frob'] = gamma_frob.item()
                            state['gamma_trace'] = gamma_trace.item()
                            state['update_count'] += 1
                            if(group["new_gamma"]):
                                K_damped = K + gamma_frob * torch.eye(n, device=K.device)
                            else:
                                K_damped = K + gamma_trace * torch.eye(n, device=K.device)
                            try:
                                L, info = torch.linalg.cholesky_ex(K_damped)
                                if info.item() == 0:
                                    inv_K = torch.cholesky_inverse(L).to(grad.dtype)
                                else:
                                    inv_K = torch.eye(n, device=grad.device, dtype=grad.dtype)
                            except Exception:
                                inv_K = torch.eye(n, device=grad.device, dtype=grad.dtype)
                                
                            state['inv_K'] = inv_K
                            state['accumulated_path'] = 0.0
                            
                            try:
                                L_eig = torch.linalg.eigvalsh(K_damped.to(torch.float64))
                                state['last_cond'] = (L_eig[-1] / torch.clamp(L_eig[0], min=1e-15)).item()
                            except:
                                state['last_cond'] = float('inf')

                        if state['inv_K'] is not None:
                            grad = (grad.to(torch.float32) @ state['inv_K'].to(torch.float32)).to(grad.dtype)
                            
                        if 'last_cond' in state:
                            metrics["cond_nums"].append(state['last_cond'])

                # =====================================================
                # СТАНДАРТНЫЙ MUON PIPELINE
                # =====================================================
                if group['weight_decay'] != 0:
                    p.data.mul_(1.0 - group['lr'] * group['weight_decay'])

                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(grad)
                
                if group['nesterov']:
                    grad = grad.add(buf, alpha=group['momentum'])
                else:
                    grad = buf.clone()
                
                update = newton_schulz(grad)
                
                scale = max(m, n) ** 0.5
                eff_lr = group['lr'] * scale
                
                p.data.add_(update, alpha=-eff_lr)

                # =====================================================
                # НАКОПЛЕНИЕ ПУТИ И ИНКРЕМЕНТ СЧЕТЧИКА
                # =====================================================
                if group['use_newton']:
                    step_norm = group['lr']
                    state['accumulated_path'] += step_norm
                    state['step_count'] += 1
                
        return metrics