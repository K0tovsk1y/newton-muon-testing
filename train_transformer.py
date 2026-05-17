"""Мини-трансформер на wikitext-2 (byte-level). Та же 5-арм ablation что и CNN/GNN.

- Архитектура: декодер-only, causal attention, n_layers слоёв с MLP.
- MatrixOptimizer применяем ко ВСЕМ Linear (QKV, attn out, MLP fc1, fc2, lm_head).
  Token embedding (1D index lookup) → AdamW.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

import time
import math
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import concurrent.futures

import torch
torch.set_num_threads(2)
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from optimizers_new import MatrixOptimizer

WIKITEXT_TRAIN = "/home/buka2004/data/datasets/wikitext/wikitext-2-v1/train-00000-of-00001.parquet"
WIKITEXT_VALID = "/home/buka2004/data/datasets/wikitext/wikitext-2-v1/validation-00000-of-00001.parquet"

CONFIG = {
    "ADAM_LR": 0.003,
    "NEWTON_MUON_LR": 0.003,
    "BATCH_SIZE": 64,
    "SEQ_LEN": 128,
    "STEPS": 2000,
    "EVAL_EVERY": 200,
    "WEIGHT_DECAY": 0.01,
    "DROPOUT": 0.0,

    "NM_UPDATE_THRESHOLD": 0.045,   # ≈ lr_avg(0.003*0.5) × refresh_interval(30)
    "NM_BETA": 0.85,
    "NM_GAMMA": 0.2,
    "NM_REFRESH_INTERVAL": 30,

    "VOCAB": 256,
    "DIM": 128,
    "N_LAYERS": 2,
    "N_HEADS": 4,
    "FF_MULT": 2,

    "PLOT_DIR": "experiments_plots_our",
}

METHOD_FLAGS = {
    "Muon":        (False, False, False),
    "NM-base":     (True,  False, False),
    "NM-gamma":    (True,  True,  False),
    "NM-interval": (True,  False, True),
    "NM-both":     (True,  True,  True),
}


_DATA_CACHE = {}
def load_corpus_bytes():
    """Загружаем wikitext-2 и кодируем в utf-8 bytes (vocab=256)."""
    if 'train' in _DATA_CACHE:
        return _DATA_CACHE['train'], _DATA_CACHE['valid']
    import pyarrow.parquet as pq
    def to_bytes(path):
        t = pq.read_table(path)
        texts = [s for s in t.column('text').to_pylist() if s.strip()]
        joined = '\n'.join(texts).encode('utf-8', errors='replace')
        return np.frombuffer(joined, dtype=np.uint8)
    _DATA_CACHE['train'] = to_bytes(WIKITEXT_TRAIN)
    _DATA_CACHE['valid'] = to_bytes(WIKITEXT_VALID)
    return _DATA_CACHE['train'], _DATA_CACHE['valid']


class MLP(nn.Module):
    def __init__(self, dim, ff_mult):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * ff_mult, bias=False)
        self.fc2 = nn.Linear(dim * ff_mult, dim, bias=False)
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class CausalAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.o = nn.Linear(dim, dim, bias=False)
    def forward(self, x):
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.o(y)


class Block(nn.Module):
    def __init__(self, dim, n_heads, ff_mult):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = CausalAttention(dim, n_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, ff_mult)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, vocab, dim, n_layers, n_heads, ff_mult, seq_len):
        super().__init__()
        self.token_emb = nn.Embedding(vocab, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, dim))
        self.blocks = nn.ModuleList([Block(dim, n_heads, ff_mult) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)
    def forward(self, idx):
        # idx: (B, T)
        x = self.token_emb(idx) + self.pos_emb[:, :idx.size(1), :]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        return self.lm_head(x)


def attach_z_hooks(model):
    """Для всех Linear: p.Z = (in_features, B*T)."""
    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            def make_h(_m):
                def h(m, inputs):
                    x = inputs[0].detach()
                    if x.ndim > 2:
                        x = x.reshape(-1, x.shape[-1])
                    m.weight.Z = x.T.float()
                return h
            mod.register_forward_pre_hook(make_h(mod))


def sample_batch(data, batch_size, seq_len, device, gen):
    """Случайно нарезаем (B, seq_len+1) фрагменты из data."""
    N = len(data)
    starts = torch.randint(0, N - seq_len - 1, (batch_size,), generator=gen)
    chunks = torch.stack([torch.from_numpy(data[s : s + seq_len + 1].astype(np.int64))
                          for s in starts.tolist()])
    chunks = chunks.to(device, non_blocking=True)
    return chunks[:, :-1], chunks[:, 1:]


@torch.no_grad()
def eval_loss(model, data, config, device, n_batches=20):
    model.eval()
    g = torch.Generator(); g.manual_seed(123)
    total, count = 0.0, 0
    for _ in range(n_batches):
        x, y = sample_batch(data, config["BATCH_SIZE"], config["SEQ_LEN"], device, g)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, config["VOCAB"]), y.reshape(-1))
        total += loss.item(); count += 1
    model.train()
    return total / count


def run_experiment(task_args):
    opt_name, seed, gpu_id, config = task_args
    start_time = time.time()

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')

    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

    data_train, data_valid = load_corpus_bytes()

    model = TinyTransformer(
        vocab=config["VOCAB"], dim=config["DIM"],
        n_layers=config["N_LAYERS"], n_heads=config["N_HEADS"],
        ff_mult=config["FF_MULT"], seq_len=config["SEQ_LEN"],
    ).to(device)
    attach_z_hooks(model)

    # 2D Linear weights → MatrixOptimizer. Остальное (embed, pos_emb, LayerNorm) → AdamW.
    matrix_params, other_params = [], []
    for name, p in model.named_parameters():
        if p.ndim == 2 and not name.startswith("token_emb"):
            matrix_params.append(p)
        else:
            other_params.append(p)

    opt_other = torch.optim.AdamW(other_params, lr=config["ADAM_LR"], weight_decay=config["WEIGHT_DECAY"])

    use_newton, new_gamma, new_interval = METHOD_FLAGS[opt_name]
    opt_matrix = MatrixOptimizer(
        matrix_params,
        lr=config["NEWTON_MUON_LR"],
        weight_decay=config["WEIGHT_DECAY"],
        use_newton=use_newton,
        nm_update_threshold=config["NM_UPDATE_THRESHOLD"],
        nm_refresh_interval=config["NM_REFRESH_INTERVAL"],
        nm_beta=config["NM_BETA"],
        nm_gamma=config["NM_GAMMA"],
        new_gamma=new_gamma,
        new_interval=new_interval,
    )

    eta_ratio = config.get("ETA_MIN_RATIO", 0.1)
    total_steps = config["STEPS"]
    sched_other = CosineAnnealingLR(opt_other, T_max=total_steps, eta_min=config["ADAM_LR"] * eta_ratio)
    sched_matrix = CosineAnnealingLR(opt_matrix, T_max=total_steps, eta_min=config["NEWTON_MUON_LR"] * eta_ratio)

    history = {'loss': [], 'val_loss': [], 'val_step': []}
    opt_step_time = 0.0
    opt_step_count = 0
    is_cuda = (device.type == 'cuda')
    skipped_first = False

    gen = torch.Generator(); gen.manual_seed(seed)

    log_every = max(1, total_steps // 5)
    model.train()
    for step in range(total_steps):
        x, y = sample_batch(data_train, config["BATCH_SIZE"], config["SEQ_LEN"], device, gen)
        opt_matrix.zero_grad(); opt_other.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, config["VOCAB"]), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        opt_matrix.step()
        if is_cuda:
            torch.cuda.synchronize()
        if skipped_first:
            opt_step_time += time.perf_counter() - t0
            opt_step_count += 1
        else:
            skipped_first = True

        opt_other.step()
        sched_matrix.step(); sched_other.step()
        history['loss'].append(loss.item())

        if (step + 1) % config["EVAL_EVERY"] == 0 or step == total_steps - 1:
            vl = eval_loss(model, data_valid, config, device)
            history['val_loss'].append(vl)
            history['val_step'].append(step + 1)
            if ((step + 1) % log_every == 0) or (step == total_steps - 1):
                print(f"  [{opt_name:11} s{seed:03d} gpu{gpu_id}] step {step+1:5d}/{total_steps} "
                      f"train={loss.item():.3f} val={vl:.3f}", flush=True)

    freqs, g_frobs, g_traces = [], [], []
    for p, st in opt_matrix.state.items():
        if 'update_count' in st and st['update_count'] > 0:
            freqs.append(total_steps / st['update_count'])
            gf = st.get('gamma_frob_t', None)
            gt = st.get('gamma_trace_t', None)
            g_frobs.append(gf.item() if torch.is_tensor(gf) else 0.0)
            g_traces.append(gt.item() if torch.is_tensor(gt) else 0.0)

    history['nm_freq'] = float(np.mean(freqs)) if freqs else 0.0
    history['nm_g_frob'] = float(np.mean(g_frobs)) if g_frobs else 0.0
    history['nm_g_trace'] = float(np.mean(g_traces)) if g_traces else 0.0
    history['opt_step_mean_ms'] = (opt_step_time / max(opt_step_count, 1)) * 1000.0

    elapsed = time.time() - start_time
    return opt_name, seed, history, elapsed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--steps', type=int, default=CONFIG["STEPS"])
    parser.add_argument('--methods', type=str, default=",".join(METHOD_FLAGS.keys()))
    parser.add_argument('--eta_min_ratio', type=float, default=0.1)
    parser.add_argument('--tag', type=str, default="")
    parser.add_argument('--gpus', type=str, default="0")
    parser.add_argument('--nm_gamma', type=float, default=None)
    parser.add_argument('--nm_refresh', type=int, default=None)
    parser.add_argument('--nm_threshold', type=float, default=None)
    parser.add_argument('--dim', type=int, default=CONFIG["DIM"])
    parser.add_argument('--n_layers', type=int, default=CONFIG["N_LAYERS"])
    parser.add_argument('--batch', type=int, default=CONFIG["BATCH_SIZE"])
    parser.add_argument('--seq', type=int, default=CONFIG["SEQ_LEN"])
    parser.add_argument('--lr', type=float, default=None)
    args = parser.parse_args()
    CONFIG["NUM_SEEDS"] = args.seeds
    CONFIG["STEPS"] = args.steps
    CONFIG["ETA_MIN_RATIO"] = args.eta_min_ratio
    CONFIG["TAG"] = args.tag
    CONFIG["DIM"] = args.dim
    CONFIG["N_LAYERS"] = args.n_layers
    CONFIG["BATCH_SIZE"] = args.batch
    CONFIG["SEQ_LEN"] = args.seq
    if args.nm_gamma is not None: CONFIG["NM_GAMMA"] = args.nm_gamma
    if args.nm_refresh is not None: CONFIG["NM_REFRESH_INTERVAL"] = args.nm_refresh
    if args.nm_threshold is not None: CONFIG["NM_UPDATE_THRESHOLD"] = args.nm_threshold
    if args.lr is not None: CONFIG["NEWTON_MUON_LR"] = args.lr
    gpu_ids = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
    CONFIG["NUM_WORKERS"] = len(gpu_ids)

    import torch.multiprocessing as mp
    try: mp.set_start_method('spawn', force=True)
    except RuntimeError: pass

    os.makedirs(CONFIG["PLOT_DIR"], exist_ok=True)
    # тёплая загрузка данных в main, чтобы parquet прочитался один раз — но spawn
    # workers заново читают свой кеш. Это OK, файл маленький.
    _ = load_corpus_bytes()

    methods = [m.strip() for m in args.methods.split(",") if m.strip() in METHOD_FLAGS]
    tasks = []
    for idx, (m, s) in enumerate([(m, s) for m in methods for s in range(CONFIG["NUM_SEEDS"])]):
        tasks.append((m, s, gpu_ids[idx % len(gpu_ids)], CONFIG))

    print(f"=== WikiText-2 Mini-Transformer | {len(methods)} arms: {methods} ===")
    print(f"Seeds: {CONFIG['NUM_SEEDS']} | Steps: {CONFIG['STEPS']} | "
          f"dim={CONFIG['DIM']} layers={CONFIG['N_LAYERS']} | GPUs: {gpu_ids}")
    print("-" * 95)

    processed = {m: {
        'loss':   [None] * CONFIG["NUM_SEEDS"],
        'val_loss': [None] * CONFIG["NUM_SEEDS"],
        'val_step': [None] * CONFIG["NUM_SEEDS"],
        'freq':   [None] * CONFIG["NUM_SEEDS"],
        'g_frob': [None] * CONFIG["NUM_SEEDS"],
        'g_trace':[None] * CONFIG["NUM_SEEDS"],
        'opt_ms': [None] * CONFIG["NUM_SEEDS"],
        'wall_t': [None] * CONFIG["NUM_SEEDS"],
    } for m in methods}

    with concurrent.futures.ProcessPoolExecutor(max_workers=CONFIG["NUM_WORKERS"]) as executor:
        futures = {executor.submit(run_experiment, task): task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            opt_name, seed, hist, elapsed = future.result()
            processed[opt_name]['loss'][seed] = hist['loss']
            processed[opt_name]['val_loss'][seed] = hist['val_loss']
            processed[opt_name]['val_step'][seed] = hist['val_step']
            processed[opt_name]['freq'][seed] = hist['nm_freq']
            processed[opt_name]['g_frob'][seed] = hist['nm_g_frob']
            processed[opt_name]['g_trace'][seed] = hist['nm_g_trace']
            processed[opt_name]['opt_ms'][seed] = hist['opt_step_mean_ms']
            processed[opt_name]['wall_t'][seed] = elapsed
            print(f"[{opt_name:11}] Seed {seed:03d} | wall {elapsed:>5.1f}s | "
                  f"opt {hist['opt_step_mean_ms']:>5.1f}ms/step | "
                  f"val={hist['val_loss'][-1]:.3f} | Freq: {hist['nm_freq']:.1f}st | "
                  f"γ {hist['nm_g_frob']:.1e}(F)/{hist['nm_g_trace']:.1e}(Tr)")

    print("-" * 95)

    # Метрика — минимальный validation loss (лучше = ниже).
    best_val = {m: np.min(np.array(processed[m]['val_loss']), axis=1) for m in methods}
    baseline = "NM-base" if "NM-base" in methods else methods[0]

    def calc_mean_ci(arr):
        arr = np.asarray(arr, dtype=float)
        return float(np.mean(arr)), float(1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr)))

    paired_lines = []
    print(f"\n[STAT] Paired diff in Min Val Loss vs {baseline} (отрицательное = лучше):")
    for m in methods:
        if m == baseline:
            continue
        diffs = best_val[m] - best_val[baseline]
        mean_d = float(np.mean(diffs))
        ci_d = float(1.96 * np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
        tstat = mean_d / (np.std(diffs, ddof=1) / np.sqrt(len(diffs)) + 1e-12)
        line = f"  {m:11} - {baseline}: {mean_d:+.4f} ± {ci_d:.4f}  (t={tstat:+.2f})"
        print(line); paired_lines.append(line)
    print()

    summary_stats = []
    for m in methods:
        v_min, v_min_ci = calc_mean_ci(best_val[m])
        v_final = np.array([v[-1] for v in processed[m]['val_loss']])
        v_fin_m, v_fin_ci = calc_mean_ci(v_final)
        opt_ms = np.array(processed[m]['opt_ms'])
        opt_ms_m, opt_ms_ci = calc_mean_ci(opt_ms)
        wall_t = np.array(processed[m]['wall_t'])
        wall_t_m, wall_t_ci = calc_mean_ci(wall_t)
        freq_m = float(np.mean(processed[m]['freq']))
        summary_stats.append([
            m,
            f"{v_min_ci:.4f}".replace("0.0", "0.0").rjust(0),  # placeholder
            f"{v_min:.4f} ± {v_min_ci:.4f}",
            f"{v_fin_m:.4f} ± {v_fin_ci:.4f}",
            f"{opt_ms_m:.2f} ± {opt_ms_ci:.2f}",
            f"{wall_t_m:.1f} ± {wall_t_ci:.1f}",
            f"every {freq_m:.1f} st",
        ])

    fig, axs = plt.subplots(2, 2, figsize=(18, 11))
    fig.suptitle(f"WikiText-2 Mini-Transformer | {CONFIG['NUM_SEEDS']} seeds", fontsize=18, fontweight='bold')

    colors = {
        'Muon':        '#7f7f7f',
        'NM-base':     '#1f77b4',
        'NM-gamma':    '#ff7f0e',
        'NM-interval': '#9467bd',
        'NM-both':     '#2ca02c',
    }

    ax_loss = axs[0, 0]; ax_val = axs[0, 1]
    ax_prm  = axs[1, 0]; ax_tbl = axs[1, 1]

    for m in methods:
        data = processed[m]
        loss_m = np.mean(data['loss'], axis=0)
        ax_loss.plot(np.arange(len(loss_m)), loss_m, label=m, color=colors[m], linewidth=1.5)
        val_steps = data['val_step'][0]
        val_arr = np.array(data['val_loss'])
        val_mean = val_arr.mean(0); val_std = val_arr.std(0)
        ax_val.plot(val_steps, val_mean, color=colors[m], label=m, linewidth=2)
        ax_val.fill_between(val_steps, val_mean-val_std, val_mean+val_std, color=colors[m], alpha=0.10)

    ax_loss.set_yscale('log'); ax_loss.set_title("Train CE"); ax_loss.set_xlabel("Step"); ax_loss.grid(alpha=0.5); ax_loss.legend()
    ax_val.set_title("Val CE"); ax_val.set_xlabel("Step"); ax_val.grid(alpha=0.5); ax_val.legend()

    ax_prm.axis('off')
    ax_prm.set_title(f"Config & Paired Diffs (vs {baseline})", fontsize=14, fontweight='bold')
    param_text = "\n".join([f"{k}: {v}" for k, v in CONFIG.items() if k != "PLOT_DIR"])
    param_text += "\n\n========================="
    param_text += "\nPaired Diff (Min Val Loss):"
    for ln in paired_lines:
        param_text += "\n" + ln
    param_text += "\n========================="
    ax_prm.text(0.5, 0.5, param_text, fontsize=10, fontfamily='monospace', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=1.5', facecolor='#f8f9fa', edgecolor='#dee2e6'))

    ax_tbl.axis('off')
    ax_tbl.set_title("Performance & Timing", fontsize=14, fontweight='bold')
    col_labels = ["Optimizer", "_", "Min Val Loss", "Final Val", "Opt step (ms)", "Wall (s)", "Refresh"]
    table = ax_tbl.table(cellText=summary_stats, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.1, 2.2)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold'); cell.set_facecolor('#e9ecef')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_part = f"_{CONFIG['TAG']}" if CONFIG.get("TAG") else ""
    filename = f"WT2_MiniTrans_d{CONFIG['DIM']}L{CONFIG['N_LAYERS']}{tag_part}_{timestamp}.png"
    filepath = os.path.join(CONFIG["PLOT_DIR"], filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n[+] Statistical plot saved to: {filepath}")
