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
            X = a * X + X @ B       #[m, n] @ [n, n] -> [m, n]
    else:
        for _ in range(steps):
            A = X @ X.T             # [m, m]
            B = b * A + c * (A @ A) # [m, m]
            X = a * X + B @ X       # [m, m] @ [m, n] -> [m, n]
            
    return X.to(G.dtype)

class MatrixOptimizer(Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.01, use_newton=False,
                 nm_beta=0.95, nm_gamma=0.2, nm_refresh_interval=16): # 16 взято из Appendix C для малых датасетов
        defaults = dict(
            lr=lr, 
            momentum=momentum, 
            weight_decay=weight_decay, 
            use_newton=use_newton,
            nm_beta=nm_beta,
            nm_gamma=nm_gamma,
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
                
                # Инициализация состояния
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                    if group['use_newton']:
                        state['K'] = None
                        state['inv_K'] = None

                m, n = p.shape

                # =====================================================
                # БЛОК NEWTON-MUON
                # =====================================================
                if group['use_newton']:
                    if getattr(p, 'Z', None) is None:
                        if not self.missing_z_warned:
                            print("\n[WARNING] Newton-Muon is active, but a weight parameter has no 'Z' attribute! Falling back to Muon.")
                            self.missing_z_warned = True
                    else:
                        # 1. Приводим Z к float32 для стабильности (Appendix A)
                        Z = p.Z.to(torch.float32)
                        
                        # Flatten Z до 2D на случай если он 3D+
                        if Z.ndim > 2:
                            if Z.size(-1) == n:
                                Z = Z.view(-1, n)
                            elif Z.size(0) == n:
                                Z = Z.view(n, -1)
                            else:
                                Z = Z.reshape(-1, Z.size(-1))

                        # 2. Умное вычисление ковариации [n, n] вне зависимости от формы Z
                        if Z.size(0) == n:    # Форма [n, N_samples] (как в вашем хуке)
                            N_samples = Z.size(1)
                            ZZT = torch.mm(Z, Z.T) / N_samples
                        elif Z.size(1) == n:  # Форма [N_samples, n]
                            N_samples = Z.size(0)
                            ZZT = torch.mm(Z.T, Z) / N_samples
                        else:
                            raise ValueError(f"Shape mismatch: param is {p.shape}, but Z is {p.Z.shape}")
                        
                        beta = group['nm_beta']
                        
                        # 3. Инициализация (Algorithm 1)
                        if state['K'] is None:
                            state['K'] = 1e-3 * torch.eye(n, device=grad.device, dtype=torch.float32)
                            
                        # Экспоненциальное сглаживание
                        state['K'].mul_(beta).add_(ZZT, alpha=1 - beta)
                            
                        # 4. Обращение матрицы только раз в `k` шагов (Refresh Interval)
                        if self._step_count % group['nm_refresh_interval'] == 0 or state['inv_K'] is None:
                            K = state['K']
                            trace = torch.trace(K)
                            
                            gamma_val = group['nm_gamma'] * (trace / n)
                            K_damped = K + gamma_val * torch.eye(n, device=K.device)
                            
                            # Cholesky Inverse как рекомендуется в Appendix B.2
                            try:
                                L, info = torch.linalg.cholesky_ex(K_damped)
                                if info.item() == 0:
                                    inv_K = torch.cholesky_inverse(L).to(grad.dtype)
                                else:
                                    inv_K = torch.linalg.inv(K_damped).to(grad.dtype)
                            except Exception:
                                inv_K = torch.linalg.inv(K_damped).to(grad.dtype)
                                
                            state['inv_K'] = inv_K
                            
                            # Логируем число обусловленности
                            try:
                                L_eig = torch.linalg.eigvalsh(K_damped.to(torch.float64))
                                cond = (L_eig[-1] / torch.clamp(L_eig[0], min=1e-15)).item()
                                metrics["cond_nums"].append(cond)
                            except Exception:
                                metrics["cond_nums"].append(float('inf'))

                        # 5. Предобуславливание градиента строго во float32 (Appendix A)
                        if state['inv_K'] is not None:
                            grad_f32 = grad.to(torch.float32)
                            grad = (grad_f32 @ state['inv_K'].to(torch.float32)).to(grad.dtype)
                # =====================================================

                # Scale Learning Rate для прямоугольных матриц
                eff_lr = group['lr'] * max(1, m / n) ** 0.5
                
                # Decoupled Weight Decay
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])
                
                # Momentum (до применения Newton-Schulz)
                buf = state['momentum_buffer']
                buf.lerp_(grad, 1 - group['momentum'])
                grad = grad.lerp_(buf, group['momentum'])
                
                # Ортогонализация (Standard Muon pipeline)
                update = newton_schulz(grad)
                p.add_(update, alpha=-eff_lr)
                
        self._step_count += 1
        return metrics