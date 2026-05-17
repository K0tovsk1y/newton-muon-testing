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
                 use_newton=False, nm_beta=0.95, nm_gamma=0.2, nm_update_threshold=0.01,
                 nm_refresh_interval=6, new_gamma=False, new_interval=False):
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
            new_interval=new_interval,
        )
        super().__init__(params, defaults)
        self.missing_z_warned = False
        # Глобальные счётчики (раньше дублировались в state каждого параметра,
        # хотя при общем lr эволюционировали идентично).
        self._step_count = 0
        self._accumulated_path = 0.0

    @torch.no_grad()
    def step(self):
        metrics = {"cond_nums": []}

        for group in self.param_groups:
            # Решение о пересчёте предобуславливателя — одно на всю группу.
            if group['use_newton']:
                if group['new_interval']:
                    do_update = (self._step_count > 0) and (self._accumulated_path > group['nm_update_threshold'])
                else:
                    do_update = ((self._step_count + 1) % group['nm_refresh_interval'] == 0)
            else:
                do_update = False

            for p in group['params']:
                if p.grad is None:
                    continue
                # Conv/любые >2D веса разворачиваем как (out, in*kH*kW*...) для NM/NS-математики.
                orig_shape = p.shape
                if p.ndim > 2:
                    grad = p.grad.view(orig_shape[0], -1)
                else:
                    grad = p.grad
                state = self.state[p]

                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(grad)

                if group['use_newton'] and 'accum_K' not in state:
                    state['K'] = None
                    state['inv_K'] = None
                    state['accum_K'] = None
                    state['accum_count'] = 0
                    state['update_count'] = 0     # Сколько раз пересчитали Холецкий
                    state['gamma_frob'] = 0.0     # Наш Gamma (Frobenius)
                    state['gamma_trace'] = 0.0    # Gamma из статьи (trace)

                m, n = grad.shape

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

                        if do_update:
                            beta = group['nm_beta']

                            avg_ZZT = state['accum_K'] / max(state['accum_count'], 1)
                            state['K'].lerp_(avg_ZZT, 1.0 - beta)

                            state['accum_K'].zero_()
                            state['accum_count'] = 0

                            K = state['K']

                            # === Гаммы остаются тензорами; .item() — отложенный
                            # (см. finalize_metrics ниже). Это убирает GPU→CPU sync в hot loop.
                            trace = torch.trace(K)
                            frob_norm = torch.linalg.matrix_norm(K, ord='fro')

                            gamma_trace = group['nm_gamma'] * (trace / n) + 1e-8
                            gamma_frob = group['nm_gamma'] * (frob_norm / math.sqrt(n)) + 1e-8

                            state['gamma_frob_t'] = gamma_frob.detach()
                            state['gamma_trace_t'] = gamma_trace.detach()
                            state['update_count'] += 1

                            if group['new_gamma']:
                                K_damped = K + gamma_frob * torch.eye(n, device=K.device)
                            else:
                                K_damped = K + gamma_trace * torch.eye(n, device=K.device)
                            # cholesky без _ex и без info.item() — на singular K
                            # ловим исключение и откатываемся к I.
                            try:
                                L = torch.linalg.cholesky(K_damped)
                                inv_K = torch.cholesky_inverse(L).to(grad.dtype)
                            except Exception:
                                inv_K = torch.eye(n, device=grad.device, dtype=grad.dtype)

                            state['inv_K'] = inv_K
                            # last_cond не считаем — eigvalsh съедал >50% времени блока.

                        if state['inv_K'] is not None:
                            grad = (grad.to(torch.float32) @ state['inv_K'].to(torch.float32)).to(grad.dtype)

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

                if p.ndim > 2:
                    p.data.add_(update.view(orig_shape), alpha=-eff_lr)
                else:
                    p.data.add_(update, alpha=-eff_lr)

            # Сброс глобального счётчика пути после обновления (по группе).
            if group['use_newton'] and group['new_interval'] and do_update:
                self._accumulated_path = 0.0

        # =====================================================
        # ГЛОБАЛЬНЫЙ ИНКРЕМЕНТ СЧЁТЧИКОВ
        # =====================================================
        if len(self.param_groups) > 0:
            self._accumulated_path += float(self.param_groups[0]['lr'])
        self._step_count += 1

        return metrics
