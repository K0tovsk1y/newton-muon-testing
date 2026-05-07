import torch
from torch.optim import Optimizer

@torch.compile # Ускоряем выполнение математики через компиляцию
def newton_schulz(G: torch.Tensor, steps: int = 5):
    """Ортогонализация матрицы через итерации Ньютона-Шульца (из статьи)."""
    # Нормализуем градиент
    X = G.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.to(G.dtype)

class MatrixOptimizer(Optimizer):
    """
    Базовый класс, который умеет делать и Muon, и Newton-Muon.
    Если use_newton=True, умножает градиент на (ZZ^T)^-1 перед ортогонализацией.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.01, use_newton=False):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, use_newton=use_newton)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            use_newton = group['use_newton']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # Инициализация моментума
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(grad)
                
                # ------ Блок Newton-Muon ------
                if use_newton and hasattr(p, 'Z'):
                    Z = p.Z
                    # Вычисляем ковариацию входов: Z @ Z^T 
                    # Для float32/bf16 лучше использовать mm
                    ZZT = torch.mm(Z, Z.T) 
                    
                    # Дампинг (чтобы матрица не была вырожденной)
                    # Авторы статьи пишут: gamma = tr(ZZT)/n * 1e-4 ... 1e-6
                    gamma = 1e-4 * (torch.trace(ZZT) / ZZT.size(0)) + 1e-6
                    K = ZZT + gamma * torch.eye(ZZT.size(0), device=Z.device, dtype=ZZT.dtype)
                    
                    # Обращаем матрицу. float32 нужен для стабильности инверсии!
                    inv_K = torch.linalg.inv(K.float()).to(grad.dtype)
                    
                    # Правое предобуславливание: G <- G @ (ZZ^T)^-1
                    grad = grad @ inv_K
                # -------------------------------

                # Масштабирование LR как в статье (зависит от размера матрицы)
                eff_lr = lr * max(1, p.size(-2) / p.size(-1)) ** 0.5
                
                # Weight decay
                if group['weight_decay'] != 0:
                    p.mul_(1 - lr * group['weight_decay'])
                
                # Momentum
                buf = state['momentum_buffer']
                buf.lerp_(grad, 1 - momentum)
                grad = grad.lerp_(buf, momentum)
                
                # Ортогонализация (сердце Muon)
                update = newton_schulz(grad)
                
                # Шаг
                p.add_(update, alpha=-eff_lr)

        return loss