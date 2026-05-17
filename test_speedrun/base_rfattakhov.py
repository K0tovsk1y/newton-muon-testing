import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import glob
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
#from triton_kernels import XXT, ba_plus_cAA

# -----------------------------------------------------------------------------
# Custom operators: activation XtX accumulation (for preconditioner)

def _dummy_scalar_like(x: torch.Tensor) -> torch.Tensor:
    return x.new_empty(())

# compile once at module scope (do not define #@torch.compile inside the custom op call path)
##@torch.compile
@torch.compile
def _accum_xtx_impl(x_2d: Tensor, accum: Tensor, count: Tensor, tmp: Tensor) -> Tensor:
    # Вместо XXT используем стандартный matmul
    # x_2d имеет форму [N, D], нам нужно D x D (x_2d.T @ x_2d)
    torch.mm(x_2d.T, x_2d, out=tmp) 
    tmp.mul_(1.0 / x_2d.size(0))
    accum.add_(tmp)
    count.add_(1.0)
    return _dummy_scalar_like(accum)
@torch.compile
def _accum_xtx_blocks4_impl(x_2d: Tensor, accum: Tensor, count: Tensor, tmp: Tensor) -> Tensor:
    N, fourD = x_2d.shape
    D = fourD // 4
    # Разрезаем на 4 блока и считаем батчем [4, N, D]
    A = x_2d.view(N, 4, D).permute(1, 0, 2)
    torch.bmm(A.transpose(1, 2), A, out=tmp)
    tmp.mul_(1.0 / N)
    accum.add_(tmp)
    count.add_(1.0)
    return _dummy_scalar_like(accum)

@torch.library.custom_op("nanogpt::accum_xtx", mutates_args=("accum", "count", "tmp"))
@torch.no_grad()
def accum_xtx_op(x_2d: Tensor, accum: Tensor, count: Tensor, tmp: Tensor) -> Tensor:
    return _accum_xtx_impl(x_2d, accum, count, tmp)

@accum_xtx_op.register_fake
def accum_xtx_fake(x_2d: Tensor, accum: Tensor, count: Tensor, tmp: Tensor):
    return accum.new_empty(())

@torch.library.custom_op("nanogpt::accum_xtx_blocks4", mutates_args=("accum", "count", "tmp"))
@torch.no_grad()
def accum_xtx_blocks4_op(x_2d: Tensor, accum: Tensor, count: Tensor, tmp: Tensor) -> Tensor:
    return _accum_xtx_blocks4_impl(x_2d, accum, count, tmp)

@accum_xtx_blocks4_op.register_fake
def accum_xtx_blocks4_fake(x_2d: Tensor, accum: Tensor, count: Tensor, tmp: Tensor):
    return accum.new_empty(())

# -----------------------------------------------------------------------------
# Muon optimizer

#@torch.compile
@torch.compile
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    
    # Нормализация
    X = G.bfloat16() / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
        
    for _ in range(steps):
        A = torch.mm(X, X.T)
        # Формула: B = c*A^2 + b*A
        # В Тритоне это ba_plus_cAA, здесь - обычный mm
        B = torch.addmm(A.mul(b), A, A, alpha=c)
        # X = X + X @ B + a * X (упрощенно)
        X = torch.addmm(X.mul(a), B, X)
        
    return X.T.to(G.dtype) if G.size(0) > G.size(1) else X.to(G.dtype)

class Muon(torch.optim.Optimizer):
    """
    Muon: MomentUm Orthogonalized by Newton-schulz

    + Right-preconditioner (EWMA second moment of activations), refresh logic:
        do_refresh = (t%32==0)
        precond_ewma = 0.950
      On refresh steps: update EWMA and compute batched Cholesky inverse.
    + Applies the inverse as a right-preconditioner to gradients BEFORE momentum+NS.

    This version additionally:
      - batches gradient preconditioning across layers (single GPU) using fp32 buffers + torch.bmm.
    """
    def __init__(
        self, params, lr=3e-4, momentum=0.95, nesterov=True, backend_steps=5,
        precond_init_diag: float = 0.001, precond_ridge_mult: float = 0.2, precond_eps: float = 1e-8,
        lr_mult_max: float = 1.0, lr_mult_ramp_steps: int = 32,
    ):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, backend_steps=backend_steps)
        super().__init__(params, defaults)

        self.precond_init_diag = float(precond_init_diag)
        self.precond_ridge_mult = float(precond_ridge_mult)
        self.precond_eps = float(precond_eps)
        self.lr_mult_max = float(lr_mult_max)
        self.lr_mult_ramp_steps = int(lr_mult_ramp_steps)

        self.global_step = 0
        self._regime_step = 0
        self._precond_attached = False
        self._precond_ready = False
        self._precond_d = None
        self._refresh_map = []
        self._refresh_K = None

        self._apply_plan = None

    def _regime_schedule_(self, step: int) -> tuple[bool, float, float]:
        since = max(0, int(step) - int(self._regime_step))
        t = since + 1
        do_refresh = (t % 32 == 0)
        precond_ewma = 0.950

        ramp = float(self.lr_mult_ramp_steps)
        if ramp <= 1.0:
            lr_mult = self.lr_mult_max
        else:
            frac = min(float(since), ramp) / ramp
            lr_mult = 1.0 + (self.lr_mult_max - 1.0) * frac

        return bool(do_refresh), float(precond_ewma), float(lr_mult)

    def precond_flag_for_step(self, step: int) -> bool:
        do_refresh, _, _ = self._regime_schedule_(int(step))
        return self._precond_attached and do_refresh

    def attach_preconditioner(self):
        self._precond_attached = True
        self._finalize_precond_buffers_()

    def _iter_params_with_stats_(self):
        for group in self.param_groups:
            for p in group['params']:
                stref = getattr(p, "_stats_ref", None)
                if stref is not None:
                    yield p, stref

    def _init_precond_state_for_param_(self, p: Tensor, stref: dict) -> None:
        st = self.state[p]
        if "precond_kind" in st:
            return

        kind = stref["kind"]
        d = int(stref["d"])
        st["precond_kind"] = kind
        st["precond_d"] = d

        if self._precond_d is None:
            self._precond_d = d
        else:
            assert self._precond_d == d, f"Expected one d; got {self._precond_d} vs {d}"

        def _fp32_mat():
            t = torch.empty((d, d), device=p.device, dtype=torch.float32)
            t.zero_()
            t.diagonal().fill_(self.precond_init_diag)
            return t

        if kind in ("qkv", "o", "c_fc"):
            st["precond_cov"] = _fp32_mat()
        elif kind == "c_proj":
            cov = torch.empty((4, d, d), device=p.device, dtype=torch.float32)
            cov.zero_()
            cov.diagonal(dim1=-2, dim2=-1).fill_(self.precond_init_diag)
            st["precond_cov"] = cov

    @torch.no_grad()
    def _apply_precond_all_grads_batched_(self):
        if (not self._precond_attached) or (not self._precond_ready):
            return
        plan = self._apply_plan
        if plan is None:
            return
            
        # Этот параметр должен строго совпадать с eps из _refresh_precond_all_batched_
        eps = 1e-5 

        def apply_woodbury_batched(G, V, D):
            """
            Применяет тождество Вудбери для быстрого правого предобуславливания.
            Формула: G_new = (G - (G @ V) * D @ V.T) / eps
            Размерности: G: [B, m, d], V: [B, d, k], D: [B, k]
            """
            GV = torch.bmm(G, V)                # [B, m, k]
            GVD = GV * D.unsqueeze(1)           # [B, m, k] (broadcasting по размерности m)
            correction = torch.bmm(GVD, V.mT)   # [B, m, d]
            return (G - correction) / eps

        # --- 1. QKV Projections ---
        if plan["g_qkv"] is not None:
            G = plan["g_qkv"]
            for i, p in enumerate(plan["qkv_params"]):
                if p.grad is None:
                    G[i].zero_()
                else:
                    G[i].copy_(p.grad, non_blocking=True)
            
            # Применяем Вудбери: вычисление за O(m*d*k) вместо O(m*d*d)
            G_new = apply_woodbury_batched(G, plan["V_qkv"], plan["D_qkv"])
            
            for i, p in enumerate(plan["qkv_params"]):
                if p.grad is not None:
                    p.grad.copy_(G_new[i], non_blocking=True)

        # --- 2. Output Projections ---
        if plan["g_o"] is not None:
            G = plan["g_o"]
            for i, p in enumerate(plan["o_params"]):
                if p.grad is None:
                    G[i].zero_()
                else:
                    G[i].copy_(p.grad, non_blocking=True)
            
            G_new = apply_woodbury_batched(G, plan["V_o"], plan["D_o"])
            
            for i, p in enumerate(plan["o_params"]):
                if p.grad is not None:
                    p.grad.copy_(G_new[i], non_blocking=True)

        # --- 3. FC Expansions ---
        if plan["g_fc"] is not None:
            G = plan["g_fc"]
            for i, p in enumerate(plan["fc_params"]):
                if p.grad is None:
                    G[i].zero_()
                else:
                    G[i].copy_(p.grad, non_blocking=True)
            
            G_new = apply_woodbury_batched(G, plan["V_fc"], plan["D_fc"])
            
            for i, p in enumerate(plan["fc_params"]):
                if p.grad is not None:
                    p.grad.copy_(G_new[i], non_blocking=True)

        # --- 4. Proj Contractions (Блочная структура) ---
        if plan["g_proj"] is not None:
            Gp = plan["g_proj"]
            for i, p in enumerate(plan["proj_params"]):
                if p.grad is None:
                    Gp[i].zero_()
                else:
                    Gp[i].copy_(p.grad, non_blocking=True)

            n = Gp.size(0)
            d = plan["d"]
            k = plan["rank"]

            # Пересобираем Gp из [n, d, 4*d] в плоский батч независимых блоков [n*4, d, d].
            # permute(0, 2, 1, 3) делает хитрое транспонирование, чтобы блоки выстроились подряд.
            G_blocks = Gp.view(n, d, 4, d).permute(0, 2, 1, 3).reshape(n * 4, d, d)
            
            # Разворачиваем V и D в плоский батч, чтобы совпасть с G_blocks
            V_blocks = plan["V_proj4"].view(n * 4, d, k)
            D_blocks = plan["D_proj4"].view(n * 4, k)

            # Применяем Вудбери ко всем 4*n блокам одновременно (максимальная загрузка GPU)
            G_new_blocks = apply_woodbury_batched(G_blocks, V_blocks, D_blocks)

            # Собираем независимые блоки обратно в единый тензор градиента [n, d, 4*d]
            G_new = G_new_blocks.view(n, 4, d, d).permute(0, 2, 1, 3).reshape(n, d, 4 * d)

            for i, p in enumerate(plan["proj_params"]):
                if p.grad is not None:
                    p.grad.copy_(G_new[i], non_blocking=True)

    @torch.no_grad()
    def _finalize_precond_buffers_(self):
        if self._precond_ready:
            return

        # Ранг аппроксимации (лучше вынести в __init__, но пока захардкодим для наглядности)
        rank = 128 
        
        refresh_map = []
        qkv_params, o_params, fc_params, proj_params = [], [], [], []

        for p, stref in self._iter_params_with_stats_():
            kind = stref["kind"]
            self._init_precond_state_for_param_(p, stref)

            if kind in ("qkv", "o", "c_fc"):
                refresh_map.append((p, kind, -1))
            elif kind == "c_proj":
                # Для c_proj сохраняем логику 4 независимых блоков
                for j in range(4):
                    refresh_map.append((p, kind, j))

            if kind == "qkv":
                qkv_params.append(p)
            elif kind == "o":
                o_params.append(p)
            elif kind == "c_fc":
                fc_params.append(p)
            elif kind == "c_proj":
                proj_params.append(p)

        self._refresh_map = refresh_map
        d = int(self._precond_d) if self._precond_d is not None else 0
        
        # Буфер _refresh_K остается полным (d x d), так как он нужен 
        # исключительно как временный тензор для передачи ковариации в метод _refresh...
        self._refresh_K = None if not refresh_map else torch.empty(
            (len(refresh_map), d, d),
            device=refresh_map[0][0].device,
            dtype=torch.float32
        )

        dev = refresh_map[0][0].device if refresh_map else torch.device("cuda")

        def alloc_grad_buf(params, out_mult):
            n = len(params)
            if n == 0:
                return None
            return torch.empty((n, out_mult * d, d), device=dev, dtype=torch.float32)

        # Вспомогательные функции для выделения памяти под низкоранговые компоненты
        def alloc_V_buf(n, dim_in, k):
            if n == 0: return None
            return torch.empty((n, dim_in, k), device=dev, dtype=torch.float32)
            
        def alloc_D_buf(n, k):
            if n == 0: return None
            return torch.empty((n, k), device=dev, dtype=torch.float32)

        plan = {
            "d": d,
            "rank": rank,
            "qkv_params": qkv_params,
            "o_params": o_params,
            "fc_params": fc_params,
            "proj_params": proj_params,

            # Буферы под сырые градиенты остаются без изменений
            "g_qkv": alloc_grad_buf(qkv_params, 3),
            "g_o":   alloc_grad_buf(o_params,   1),
            "g_fc":  alloc_grad_buf(fc_params,  4),
            "g_proj": torch.empty((len(proj_params), d, 4 * d), device=dev, dtype=torch.float32) if proj_params else None,

            # --- ЗАМЕНА ПОЛНЫХ МАТРИЦ (inv_*) НА НИЗКОРАНГОВЫЕ (V_* и D_*) ---
            "V_qkv": alloc_V_buf(len(qkv_params), d, rank),
            "D_qkv": alloc_D_buf(len(qkv_params), rank),
            
            "V_o":   alloc_V_buf(len(o_params), d, rank),
            "D_o":   alloc_D_buf(len(o_params), rank),
            
            "V_fc":  alloc_V_buf(len(fc_params), d, rank),
            "D_fc":  alloc_D_buf(len(fc_params), rank),

            # Для c_proj выделяем 4 блока векторов
            "V_proj4": torch.empty((len(proj_params), 4, d, rank), device=dev, dtype=torch.float32) if proj_params else None,
            "D_proj4": torch.empty((len(proj_params), 4, rank), device=dev, dtype=torch.float32) if proj_params else None,
            
            # Мы удалили tmp_proj_blocks и tmp_blocks_in, так как для Вудбери 
            # они не понадобятся в том виде, в котором использовались раньше.
        }
        self._apply_plan = plan

        # Раздаем ссылки на новые тензоры V и D в state параметров, 
        # чтобы _refresh_precond_all_batched_ мог писать прямо в них.
        if plan["V_qkv"] is not None:
            for i, p in enumerate(qkv_params):
                self.state[p]["precond_V"] = plan["V_qkv"][i]
                self.state[p]["precond_D"] = plan["D_qkv"][i]

        if plan["V_o"] is not None:
            for i, p in enumerate(o_params):
                self.state[p]["precond_V"] = plan["V_o"][i]
                self.state[p]["precond_D"] = plan["D_o"][i]

        if plan["V_fc"] is not None:
            for i, p in enumerate(fc_params):
                self.state[p]["precond_V"] = plan["V_fc"][i]
                self.state[p]["precond_D"] = plan["D_fc"][i]

        if plan["V_proj4"] is not None:
            for i, p in enumerate(proj_params):
                self.state[p]["precond_V"] = plan["V_proj4"][i]
                self.state[p]["precond_D"] = plan["D_proj4"][i]

        self._precond_ready = True

    @torch.no_grad()
    def _refresh_precond_all_batched_(self, do_inverse: bool, precond_ewma: float):
        if (not self._precond_attached) or (not self._precond_ready):
            return

        one_minus = 1.0 - float(precond_ewma)

        # 1. Обновление ковариации (EWMA) остается без изменений
        for p, stref in self._iter_params_with_stats_():
            st = self.state[p]
            kind = st["precond_kind"]

            cnt = stref["count"]
            w = (cnt > 0) * one_minus

            if kind in ("qkv", "o", "c_fc"):
                st["precond_cov"].lerp_(stref["accum"] / cnt.clamp_min(1.0), w)
            elif kind == "c_proj":
                st["precond_cov"].lerp_(stref["accum"] / cnt.clamp_min(1.0), w)

        if not do_inverse:
            return
        if self._refresh_K is None or not self._refresh_map:
            return

        K = self._refresh_K
        d = int(self._precond_d)

        # Собираем все матрицы K в один батч-тензор
        for i, (p, kind, sub) in enumerate(self._refresh_map):
            st = self.state[p]
            if kind in ("qkv", "o", "c_fc"):
                K[i].copy_(st["precond_cov"])
            else:
                K[i].copy_(st["precond_cov"][sub])

        # --- НАЧАЛО НОВОГО БЛОКА: Батчевое Randomized SVD ---
        # В идеале эти параметры передаются через __init__ оптимизатора
        rank = 128            # k: Целевой ранг (топ-компоненты)
        eps = 1e-5            # Шумовой пол (значительно меньше, чем старый ridge)
        n_power_iter = 1      # Количество степенных итераций
        oversample = 10       # Запас для стабильности rSVD
        
        B = K.size(0)
        k_opt = min(rank + oversample, d)
        
        # Генерация случайной матрицы Омега
        Omega = torch.randn(B, d, k_opt, device=K.device, dtype=K.dtype)
        
        # Степенные итерации (Power iterations)
        for _ in range(n_power_iter):
            Omega = torch.bmm(K, Omega)
            Omega, _ = torch.linalg.qr(Omega) # Батчевая ортогонализация
            
        # Проекция ковариации в малое подпространство: B_mat (B, k_opt, k_opt)
        B_mat = torch.bmm(Omega.mT, torch.bmm(K, Omega))
        
        # Точное спектральное разложение малых матриц
        L, W = torch.linalg.eigh(B_mat)
        
        # torch.linalg.eigh возвращает значения по возрастанию. 
        # Нам нужны топ-k (самые большие), поэтому берем срез с конца.
        L_topk = L[:, -rank:]             # (B, rank)
        W_topk = W[:, :, -rank:]          # (B, k_opt, rank)
        
        # Восстанавливаем главные векторы в исходном пространстве: V (B, d, rank)
        V = torch.bmm(Omega, W_topk)      
        
        # Вычисляем диагональ матрицы D для тождества Вудбери.
        # Математически: D = 1 / (eps * L^{-1} + 1). 
        # Алгебраический трюк для стабильности: D = L / (eps + L)
        D = L_topk / (eps + L_topk)       # (B, rank)
        # --- КОНЕЦ НОВОГО БЛОКА ---

        # Раздаем результаты (V и D) обратно в стейты параметров.
        for i, (p, kind, sub) in enumerate(self._refresh_map):
            st = self.state[p]
            # Внимание: В методе _finalize_precond_buffers_ 
            # нужно создать precond_V и precond_D вместо precond_inv_apply
            if kind in ("qkv", "o", "c_fc"):
                st["precond_V"].copy_(V[i])
                st["precond_D"].copy_(D[i])
            else:
                st["precond_V"][sub].copy_(V[i])
                st["precond_D"][sub].copy_(D[i])

    def step(self):
        do_refresh, precond_ewma, lr_mult = self._regime_schedule_(self.global_step)
        since = max(0, int(self.global_step) - int(self._regime_step))
        t = since + 1
        do_inverse = bool(self._precond_attached and do_refresh)

        if self._precond_attached and do_refresh:
            self._finalize_precond_buffers_()
            self._refresh_precond_all_batched_(do_inverse=do_inverse, precond_ewma=precond_ewma)
            for _, stref in self._iter_params_with_stats_():
                stref["accum"].zero_()
                stref["count"].zero_()

        self._apply_precond_all_grads_batched_()

        for group in self.param_groups:
            lr = group['lr'] * lr_mult
            momentum = group['momentum']
            steps = group['backend_steps']
            for p in group['params']:
                g = p.grad
                if g is None:
                    continue

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                if group['nesterov']:
                    g = g.add(buf, alpha=momentum)

                if g.size(0) == 3 * g.size(1):
                    g = torch.cat([zeropower_via_newtonschulz5(g1, steps=steps) for g1 in g.split(g.size(1))])
                    scale = g.size(1)**0.5
                else:
                    g = zeropower_via_newtonschulz5(g, steps=steps)
                    scale = max(g.size(0), g.size(1))**0.5
                p.data.add_(g, alpha=-lr * scale)

        self.global_step += 1

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos()
            self.sin_cached = freqs.sin()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

def rmsnorm(x0, eps=1e-6):
    x = x0.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return x.type_as(x0)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

        d = self.n_embd
        self.qkv_xtx_accum = nn.Buffer(torch.zeros(d, d, dtype=torch.float32), persistent=False)
        self.o_xtx_accum   = nn.Buffer(torch.zeros(d, d, dtype=torch.float32), persistent=False)
        self.xtx_tmp       = nn.Buffer(torch.empty(d, d, dtype=torch.float32), persistent=False)
        self.qkv_xtx_count = nn.Buffer(torch.zeros((), dtype=torch.float32), persistent=False)
        self.o_xtx_count   = nn.Buffer(torch.zeros((), dtype=torch.float32), persistent=False)

        self.c_attn.weight._stats_ref = {"kind": "qkv", "d": d, "accum": self.qkv_xtx_accum, "count": self.qkv_xtx_count}
        self.c_proj.weight._stats_ref = {"kind": "o",   "d": d, "accum": self.o_xtx_accum,   "count": self.o_xtx_count}

    def forward(self, x, precond_flag: bool = False):
        B, T, C = x.size()

        if precond_flag:
            x2d = x.flatten(0, -2)
            torch.ops.nanogpt.accum_xtx(x2d, self.qkv_xtx_accum, self.qkv_xtx_count, self.xtx_tmp)

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, self.head_dim)
        q = q.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        if precond_flag:
            y2d = y.flatten(0, -2)
            torch.ops.nanogpt.accum_xtx(y2d, self.o_xtx_accum, self.o_xtx_count, self.xtx_tmp)

        y = self.c_proj(y)
        return y

    def _apply(self, fn):
        super()._apply(fn)
        d = self.n_embd
        self.c_attn.weight._stats_ref = {"kind": "qkv", "d": d, "accum": self.qkv_xtx_accum, "count": self.qkv_xtx_count}
        self.c_proj.weight._stats_ref = {"kind": "o",   "d": d, "accum": self.o_xtx_accum,   "count": self.o_xtx_count}
        return self

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

        d = config.n_embd
        self.fc_xtx_accum  = nn.Buffer(torch.zeros(d, d, dtype=torch.float32), persistent=False)
        self.fc_xtx_tmp    = nn.Buffer(torch.empty(d, d, dtype=torch.float32), persistent=False)
        self.fc_xtx_count  = nn.Buffer(torch.zeros((), dtype=torch.float32), persistent=False)

        self.proj_xtx_accum = nn.Buffer(torch.zeros(4, d, d, dtype=torch.float32), persistent=False)
        self.proj_xtx_tmp   = nn.Buffer(torch.empty(4, d, d, dtype=torch.float32), persistent=False)
        self.proj_xtx_count = nn.Buffer(torch.zeros((), dtype=torch.float32), persistent=False)

        self.c_fc.weight._stats_ref = {"kind": "c_fc",   "d": d, "accum": self.fc_xtx_accum,   "count": self.fc_xtx_count}
        self.c_proj.weight._stats_ref = {"kind": "c_proj","d": d, "accum": self.proj_xtx_accum, "count": self.proj_xtx_count}

    def forward(self, x, precond_flag: bool = False):
        if precond_flag:
            x2d = x.flatten(0, -2)
            torch.ops.nanogpt.accum_xtx(x2d, self.fc_xtx_accum, self.fc_xtx_count, self.fc_xtx_tmp)

        x = self.c_fc(x)
        x = F.gelu(x)

        if precond_flag:
            z2d = x.flatten(0, -2)
            torch.ops.nanogpt.accum_xtx_blocks4(z2d, self.proj_xtx_accum, self.proj_xtx_count, self.proj_xtx_tmp)

        x = self.c_proj(x)
        return x

    def _apply(self, fn):
        super()._apply(fn)
        d = self.c_fc.weight.size(1)
        self.c_fc.weight._stats_ref = {"kind": "c_fc",   "d": d, "accum": self.fc_xtx_accum,   "count": self.fc_xtx_count}
        self.c_proj.weight._stats_ref = {"kind": "c_proj","d": d, "accum": self.proj_xtx_accum, "count": self.proj_xtx_count}
        return self

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        self.attn_scale = (1 / (2 * config.n_layer)**0.5)

    def forward(self, x, precond_flag: bool = False):
        x = x + self.attn_scale * self.attn(rmsnorm(x), precond_flag)
        x = x + self.mlp(rmsnorm(x), precond_flag)
        return x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 50257
    n_layer : int = 12
    n_head : int = 12
    n_embd : int = 768

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None, return_logits=True, precond_flag: bool = False):
        precond_flag = bool(precond_flag) and self.training

        x = self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x, precond_flag)
        x = rmsnorm(x)

        if targets is not None:
            logits = self.lm_head(x).float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :]).float()
            loss = None

        if not return_logits:
            logits = None
        return logits, loss

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
    if header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    assert header[1] == 1, "unsupported version"
    return int(header[2])

def _load_data_shard(filename):
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256*4), dtype=np.int32)
        assert header[0] == 20240520, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = int(header[2])
        tokens = np.frombuffer(f.read(), dtype=np.uint16)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens

class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += int(shard_ntok)
        self.ntok_total = ntok_total

        self.reset()

    def reset(self):
        self.current_shard = 0
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    input_bin : str = 'data/fineweb10B/fineweb_train_*.bin'
    input_val_bin : str = 'data/fineweb10B/fineweb_val_*.bin'
    batch_size : int = 8*64
    device_batch_size : int = 64
    sequence_length : int = 1024
    num_iterations : int = 6200
    learning_rate : float = 0.0040
    warmup_iters : int = 0
    warmdown_iters : int = 1800
    weight_decay : float = 0
    val_loss_every : int = 100
    val_tokens : int = 10485760
    save_every : int = 0
args = Hyperparameters()

assert torch.cuda.is_available()
ddp_rank = 0
ddp_world_size = 1
device = 'cuda:0'
torch.cuda.set_device(0)
print(f"using device: {device}")
master_process = True

B, T = args.device_batch_size, args.sequence_length
assert args.val_tokens % (B * T * ddp_world_size) == 0
val_steps = args.val_tokens // (B * T * ddp_world_size)
assert args.batch_size % (B * ddp_world_size) == 0
train_accumulation_steps = args.batch_size // (B * ddp_world_size)

train_loader = DistributedDataLoader(args.input_bin, B, T, ddp_rank, ddp_world_size)
val_loader = DistributedDataLoader(args.input_val_bin, B, T, ddp_rank, ddp_world_size)
if master_process:
    print(f"Training DataLoader: total number of tokens: {train_loader.ntok_total} across {len(train_loader.files)} files")
    print(f"Validation DataLoader: total number of tokens: {val_loader.ntok_total} across {len(val_loader.files)} files")
x, y = train_loader.next_batch()

num_vocab = 50257
model = GPT(GPTConfig(vocab_size=num_vocab, n_layer=12, n_head=12, n_embd=768)).cuda()
model = torch.compile(model)
raw_model = model
ctx = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)

optimizer1 = torch.optim.AdamW(raw_model.lm_head.parameters(), lr=args.learning_rate, betas=(0.9, 0.95),
                               weight_decay=args.weight_decay, fused=True)
optimizer2 = Muon(raw_model.transformer.h.parameters(), lr=0.1*args.learning_rate, momentum=0.95)
optimizer2.attach_preconditioner()
optimizers = [optimizer1, optimizer2]

def get_lr(it):
    assert it <= args.num_iterations
    if it < args.warmup_iters:
        return (it+1) / args.warmup_iters
    elif it < args.num_iterations - args.warmdown_iters:
        return 1.0
    else:
        return (args.num_iterations - it) / args.warmdown_iters

schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

if master_process:
    run_id = str(uuid.uuid4())
    os.makedirs('logs/%s/' % run_id, exist_ok=True)
    logfile = 'logs/%s.txt' % run_id
    with open(logfile, "w") as f:
        f.write('='*100 + '\n')
        f.write(code)
        f.write('='*100 + '\n')
        f.write(f"Running pytorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}\nnvidia-smi:\n")
        import subprocess
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        f.write(f'{result.stdout}\n')
        f.write('='*100 + '\n')

training_time_ms = 0
torch.cuda.synchronize()
t0 = time.time()

train_loader.reset()
for step in range(args.num_iterations + 1):
    last_step = (step == args.num_iterations)
    if step == 32:
        torch.cuda.synchronize()
        training_time_ms = 0
        t0 = time.time()
    timed_steps = float('nan') if step <= 33 else (step - 32) + 1

    if (last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0)):
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)

        model.eval()
        val_loader.reset()
        val_loss = 0.0
        for _ in range(val_steps):
            x_val, y_val = val_loader.next_batch()
            with torch.no_grad():
                _, loss = model(x_val, y_val, return_logits=False, precond_flag=False)
                val_loss += loss
        val_loss /= val_steps

        if master_process:
            print(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms')
            with open(logfile, "a") as f:
                f.write(f'step:{step}/{args.num_iterations} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/(timed_steps-1):.2f}ms\n')

        torch.cuda.synchronize()
        t0 = time.time()

    if master_process and (last_step or (args.save_every > 0 and step % args.save_every == 0)):
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.time() - t0)
        log = dict(step=step, code=code, model=raw_model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
        torch.save(log, 'logs/%s/state_step%06d.pt' % (run_id, step))
        torch.cuda.synchronize()
        t0 = time.time()

    if last_step:
        break

    # --------------- TRAINING SECTION BEGIN -----------------
    model.train()
    optimizer2.global_step = step
    precond_flag = optimizer2.precond_flag_for_step(step)

    for _ in range(train_accumulation_steps):
        with ctx:
            _, loss = model(x, y, return_logits=False, precond_flag=precond_flag)
            train_loss = loss.detach()
            loss = loss / train_accumulation_steps
        x, y = train_loader.next_batch()
        loss.backward()

    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
    model.zero_grad(set_to_none=True)
    # --------------- TRAINING SECTION END -------------------

    if master_process:
        approx_time = training_time_ms + 1000 * (time.time() - t0)
        print(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms")
        with open(logfile, "a") as f:
            f.write(f"step:{step+1}/{args.num_iterations} train_loss:{train_loss.item():.4f} train_time:{approx_time:.0f}ms step_avg:{approx_time/timed_steps:.2f}ms\n")

if master_process:
    print(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")
