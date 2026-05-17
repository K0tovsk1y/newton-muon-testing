import os
# Мягкое ограничение CPU-потоков, чтобы 5 воркеров не съели 30+ ядер
# (но не 1 поток — иначе cholesky/matmul тормозят).
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")
# Inductor не должен раздувать compile-фазу.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
import time
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import concurrent.futures

import torch
torch.set_num_threads(2)
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch.optim.lr_scheduler import CosineAnnealingLR

from optimizers_new import MatrixOptimizer

CONFIG = {
    "ADAM_LR": 0.01,
    "NEWTON_MUON_LR": 0.003,
    "NUM_WORKERS": 5,
    "NUM_LAYERS": 5,
    "HIDDEN_DIM": 128,
    "NUM_SEEDS": 150,
    "EPOCHS": 100,
    "WEIGHT_DECAY": 0.01,
    "DROPOUT": 0.2,

    "NM_UPDATE_THRESHOLD": 0.009,
    "NM_BETA": 0.85,
    "NM_GAMMA": 0.3,

    "NM_REFRESH_INTERVAL": 6,

    "TRACKING_INTERVAL": 5,
    "PLOT_DIR": "experiments_plots_our",
}

# (use_newton, new_gamma, new_interval)
METHOD_FLAGS = {
    "Muon":        (False, False, False),  # baseline без Newton-преобуславливателя
    "NM-base":     (True,  False, False),  # как в optimizers.py: trace-gamma + фиксированный интервал
    "NM-gamma":    (True,  True,  False),  # Frobenius-gamma, интервал фиксированный
    "NM-interval": (True,  False, True),   # trace-gamma, path-based интервал
    "NM-both":     (True,  True,  True),   # обе оптимизации
}

def attach_z_hooks(model: torch.nn.Module):
    attached_count = 0
    for name, module in model.named_modules():
        if isinstance(module, GCNConv):
            def pre_hook(mod, inputs):
                x = inputs[0].detach()
                for p in mod.parameters():
                    if p.ndim == 2:
                        p.Z = x.T.float()
            module.register_forward_pre_hook(pre_hook)
            attached_count += 1
    return attached_count

class DeepGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=CONFIG["DROPOUT"], training=self.training)
        return x

def run_experiment(task_args):
    opt_name, seed, config = task_args
    start_time = time.time()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
    data = dataset[0].to(device)

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = DeepGCN(dataset.num_features, config["HIDDEN_DIM"], dataset.num_classes, config["NUM_LAYERS"]).to(device)
    attach_z_hooks(model)

    m_params = [p for n, p in model.named_parameters() if p.ndim == 2]
    o_params = [p for n, p in model.named_parameters() if p.ndim != 2]

    tracked_param = random.choice(m_params)
    weight_snapshots = []

    opt_other = torch.optim.AdamW(o_params, lr=config["ADAM_LR"], weight_decay=config["WEIGHT_DECAY"])

    use_newton, new_gamma, new_interval = METHOD_FLAGS[opt_name]
    opt_matrix = MatrixOptimizer(
        m_params,
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
    sched_other = CosineAnnealingLR(opt_other, T_max=config["EPOCHS"], eta_min=config["ADAM_LR"] * eta_ratio)
    sched_matrix = CosineAnnealingLR(opt_matrix, T_max=config["EPOCHS"], eta_min=config["NEWTON_MUON_LR"] * eta_ratio)

    history = {'loss': [], 'val_acc': [], 'cond_nums': []}
    opt_step_time = 0.0     # суммарное время opt_matrix.step() (с GPU sync), без первой эпохи
    opt_step_count = 0
    is_cuda = (device.type == 'cuda')

    for epoch in range(config["EPOCHS"]):
        model.train()
        opt_matrix.zero_grad()
        opt_other.zero_grad()

        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Первая эпоха идёт без замера (там компилируется newton_schulz через torch.compile).
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        m_info = opt_matrix.step()
        if is_cuda:
            torch.cuda.synchronize()
        if epoch > 0:
            opt_step_time += time.perf_counter() - t0
            opt_step_count += 1

        opt_other.step()

        sched_other.step()
        sched_matrix.step()

        history['loss'].append(loss.item())

        if m_info and "cond_nums" in m_info and len(m_info["cond_nums"]) > 0:
            history['cond_nums'].append(np.mean(m_info["cond_nums"]))
        else:
            last_val = history['cond_nums'][-1] if len(history['cond_nums']) > 0 else 1.0
            history['cond_nums'].append(last_val)

        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(dim=-1)
            acc = (pred[data.val_mask] == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
            history['val_acc'].append(acc)

        if epoch % config["TRACKING_INTERVAL"] == 0 or epoch == config["EPOCHS"] - 1:
            weight_snapshots.append((epoch, tracked_param.detach().cpu().clone()))

    # ================= ИЗВЛЕЧЕНИЕ СТАТИСТИКИ ИЗ OPTIMIZER =================
    # gamma_*_t хранятся как тензоры в state — конвертим тут, чтобы не блокировать hot loop.
    freqs, g_frobs, g_traces = [], [], []
    for p, st in opt_matrix.state.items():
        if 'update_count' in st and st['update_count'] > 0:
            freqs.append(config["EPOCHS"] / st['update_count'])
            gf = st.get('gamma_frob_t', None)
            gt = st.get('gamma_trace_t', None)
            g_frobs.append(gf.item() if torch.is_tensor(gf) else 0.0)
            g_traces.append(gt.item() if torch.is_tensor(gt) else 0.0)

    history['nm_freq'] = np.mean(freqs) if freqs else 0.0
    history['nm_g_frob'] = np.mean(g_frobs) if g_frobs else 0.0
    history['nm_g_trace'] = np.mean(g_traces) if g_traces else 0.0
    # Время на 1 step (среднее по эпохам, без первой)
    history['opt_step_time'] = opt_step_time
    history['opt_step_mean_ms'] = (opt_step_time / max(opt_step_count, 1)) * 1000.0
    # ======================================================================

    w_final = weight_snapshots[-1][1]
    snap_values = {}
    for ep, w_t in weight_snapshots[:-1]:
        diff = w_t - w_final
        sigma_w = diff @ diff.T
        eigs = torch.linalg.eigvalsh(sigma_w.to(torch.float32))
        eigs = torch.relu(eigs)
        top1_ratio = eigs[-1] / (eigs.sum() + 1e-12)
        snap_values[ep] = top1_ratio.item()

    anisotropy_full = []
    curr_val = snap_values.get(0, 0.0)
    for epoch in range(config["EPOCHS"]):
        if epoch in snap_values:
            curr_val = snap_values[epoch]
        anisotropy_full.append(curr_val)

    history['anisotropy'] = anisotropy_full

    elapsed = time.time() - start_time
    return opt_name, seed, history, elapsed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, default=CONFIG["NUM_SEEDS"])
    parser.add_argument('--epochs', type=int, default=CONFIG["EPOCHS"])
    parser.add_argument('--workers', type=int, default=CONFIG["NUM_WORKERS"])
    parser.add_argument('--methods', type=str, default=",".join(METHOD_FLAGS.keys()),
                        help="comma-separated subset of " + ",".join(METHOD_FLAGS.keys()))
    parser.add_argument('--eta_min_ratio', type=float, default=0.1,
                        help="eta_min as a fraction of base lr in CosineAnnealingLR (default 0.1)")
    parser.add_argument('--tag', type=str, default="", help="suffix for plot filename")
    args = parser.parse_args()
    CONFIG["NUM_SEEDS"] = args.seeds
    CONFIG["EPOCHS"] = args.epochs
    CONFIG["NUM_WORKERS"] = args.workers
    CONFIG["ETA_MIN_RATIO"] = args.eta_min_ratio
    CONFIG["TAG"] = args.tag

    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    os.makedirs(CONFIG["PLOT_DIR"], exist_ok=True)

    methods = [m.strip() for m in args.methods.split(",") if m.strip() in METHOD_FLAGS]
    tasks = [(m, s, CONFIG) for m in methods for s in range(0, CONFIG["NUM_SEEDS"])]

    print(f"=== GNN Ablation | {len(methods)} arms: {methods} ===")
    print(f"Layers: {CONFIG['NUM_LAYERS']} | Seeds: {CONFIG['NUM_SEEDS']} | Epochs: {CONFIG['EPOCHS']}")
    print("-" * 95)

    processed = {m: {
        'acc':    [None] * CONFIG["NUM_SEEDS"],
        'loss':   [None] * CONFIG["NUM_SEEDS"],
        'cond':   [None] * CONFIG["NUM_SEEDS"],
        'anis':   [None] * CONFIG["NUM_SEEDS"],
        'freq':   [None] * CONFIG["NUM_SEEDS"],
        'g_frob': [None] * CONFIG["NUM_SEEDS"],
        'g_trace':[None] * CONFIG["NUM_SEEDS"],
        'opt_t':  [None] * CONFIG["NUM_SEEDS"],
        'opt_ms': [None] * CONFIG["NUM_SEEDS"],
        'wall_t': [None] * CONFIG["NUM_SEEDS"],
    } for m in methods}

    with concurrent.futures.ProcessPoolExecutor(max_workers=CONFIG["NUM_WORKERS"]) as executor:
        futures = {executor.submit(run_experiment, task): task for task in tasks}

        for future in concurrent.futures.as_completed(futures):
            opt_name, seed, hist, elapsed = future.result()

            processed[opt_name]['acc'][seed] = hist['val_acc']
            processed[opt_name]['loss'][seed] = hist['loss']
            processed[opt_name]['cond'][seed] = hist['cond_nums']
            processed[opt_name]['anis'][seed] = hist['anisotropy']
            processed[opt_name]['freq'][seed] = hist['nm_freq']
            processed[opt_name]['g_frob'][seed] = hist['nm_g_frob']
            processed[opt_name]['g_trace'][seed] = hist['nm_g_trace']
            processed[opt_name]['opt_t'][seed] = hist['opt_step_time']
            processed[opt_name]['opt_ms'][seed] = hist['opt_step_mean_ms']
            processed[opt_name]['wall_t'][seed] = elapsed

            f_acc = hist['val_acc'][-1]
            print(f"[{opt_name:11}] Seed {seed:03d} | wall {elapsed:>5.1f}s | opt {hist['opt_step_mean_ms']:>5.1f}ms/step | "
                  f"Acc: {f_acc:.4f} | Freq: {hist['nm_freq']:.1f}eps | γ {hist['nm_g_frob']:.1e}(F)/{hist['nm_g_trace']:.1e}(Tr)")

    print("-" * 95)
    print("Computing statistics and generating plot...")

    # ================= СТАТИСТИКА =================
    def calc_mean_ci(arr):
        arr = np.asarray(arr, dtype=float)
        mean = np.mean(arr)
        ci = 1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr))
        return mean, ci

    best_acc = {m: np.max(np.array(processed[m]['acc']), axis=1) for m in methods}

    # Paired diff каждой арки vs baseline
    paired_lines = []
    baseline = "NM-base"
    print(f"\n[STAT] Paired diff in Best Accuracy vs {baseline}:")
    for m in methods:
        if m == baseline:
            continue
        diffs = best_acc[m] - best_acc[baseline]
        mean_d = np.mean(diffs)
        ci_d = 1.96 * np.std(diffs, ddof=1) / np.sqrt(len(diffs))
        tstat = mean_d / (np.std(diffs, ddof=1) / np.sqrt(len(diffs)) + 1e-12)
        line = f"  {m:11} - {baseline}: {mean_d:+.4f} ± {ci_d:.4f}  (t={tstat:+.2f})"
        print(line)
        paired_lines.append(line)
    print()

    # Сводная таблица
    summary_stats = []
    for m in methods:
        acc_matrix = np.array(processed[m]['acc'])

        auc_per_seed = np.trapezoid(acc_matrix, axis=1) / (CONFIG["EPOCHS"] - 1)
        auc_m_, auc_ci = calc_mean_ci(auc_per_seed)

        best_per_seed = best_acc[m]
        best_m_, best_ci = calc_mean_ci(best_per_seed)

        final_per_seed = acc_matrix[:, -1]
        final_m_, final_ci = calc_mean_ci(final_per_seed)

        opt_ms = np.array(processed[m]['opt_ms'], dtype=float)
        opt_ms_m, opt_ms_ci = calc_mean_ci(opt_ms)
        wall_t = np.array(processed[m]['wall_t'], dtype=float)
        wall_t_m, wall_t_ci = calc_mean_ci(wall_t)

        freq_m_ = np.mean(processed[m]['freq'])

        summary_stats.append([
            m,
            f"{auc_m_:.4f} ± {auc_ci:.4f}",
            f"{best_m_:.4f} ± {best_ci:.4f}",
            f"{final_m_:.4f} ± {final_ci:.4f}",
            f"{opt_ms_m:.2f} ± {opt_ms_ci:.2f}",
            f"{wall_t_m:.1f} ± {wall_t_ci:.1f}",
            f"every {freq_m_:.1f} ep",
        ])

    # ================= ВИЗУАЛИЗАЦИЯ =================
    fig, axs = plt.subplots(2, 3, figsize=(24, 12))
    fig.suptitle(f"Newton-Muon 2x2 Ablation | {CONFIG['NUM_SEEDS']} seeds", fontsize=20, y=0.96, fontweight='bold')

    colors = {
        'Muon':        '#7f7f7f',  # серый
        'NM-base':     '#1f77b4',  # синий
        'NM-gamma':    '#ff7f0e',  # оранжевый
        'NM-interval': '#9467bd',  # фиолетовый
        'NM-both':     '#2ca02c',  # зеленый
    }
    epochs = np.arange(CONFIG["EPOCHS"])

    ax_loss = axs[0, 0]
    ax_acc  = axs[0, 1]
    ax_prm  = axs[0, 2]
    ax_cond = axs[1, 0]
    ax_anis = axs[1, 1]
    ax_tbl  = axs[1, 2]

    for m in methods:
        data = processed[m]
        acc_m_ = np.mean(data['acc'], axis=0)
        acc_s_ = np.std(data['acc'], axis=0)
        loss_m_ = np.mean(data['loss'], axis=0)
        cond_m_ = np.mean(data['cond'], axis=0)
        anis_m_ = np.mean(data['anis'], axis=0)
        anis_s_ = np.std(data['anis'], axis=0)

        ax_loss.plot(epochs, loss_m_, label=m, color=colors[m], linewidth=2)
        ax_acc.plot(epochs, acc_m_, color=colors[m], label=m, linewidth=2)
        ax_acc.fill_between(epochs, acc_m_-acc_s_, acc_m_+acc_s_, color=colors[m], alpha=0.10)

        tracked_epochs = np.arange(0, CONFIG["EPOCHS"], CONFIG["TRACKING_INTERVAL"])
        if len(tracked_epochs) > 0:
            ax_anis.plot(tracked_epochs, anis_m_[tracked_epochs], color=colors[m], label=m, linewidth=2, marker='o', markersize=4)
            ax_anis.fill_between(tracked_epochs,
                                 anis_m_[tracked_epochs] - anis_s_[tracked_epochs],
                                 anis_m_[tracked_epochs] + anis_s_[tracked_epochs],
                                 color=colors[m], alpha=0.08)

        if not np.all(cond_m_ == 1.0):
            ax_cond.plot(epochs, cond_m_, color=colors[m], label=fr'{m} $\kappa$', linewidth=2)

    ax_loss.set_yscale('log'); ax_loss.set_title("Train Loss", fontsize=14)
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Cross Entropy"); ax_loss.grid(True, alpha=0.5); ax_loss.legend()

    ax_acc.set_title("Validation Accuracy", fontsize=14)
    ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy"); ax_acc.grid(True, alpha=0.5); ax_acc.legend()

    ax_cond.set_yscale('log'); ax_cond.set_title(r"Condition Number $\kappa(K + \gamma I)$", fontsize=14)
    ax_cond.set_xlabel("Epoch"); ax_cond.set_ylabel(r"$\kappa$"); ax_cond.grid(True, alpha=0.5); ax_cond.legend()

    ax_anis.set_title(r"Weight Displacement Anisotropy ($\Sigma_W$)", fontsize=14)
    ax_anis.set_xlabel("Epoch"); ax_anis.set_ylabel("Top-1 Eigenvalue Ratio"); ax_anis.grid(True, alpha=0.5); ax_anis.legend()

    # Гиперпараметры + paired diffs
    ax_prm.axis('off')
    ax_prm.set_title("Config & Paired Diffs (vs NM-base)", fontsize=16, fontweight='bold', pad=10)
    param_text = "\n".join([f"{k}: {v}" for k, v in CONFIG.items() if k != "PLOT_DIR"])
    param_text += "\n\n========================="
    param_text += "\nPaired Diff (Best Acc):"
    for ln in paired_lines:
        param_text += "\n" + ln
    param_text += "\n========================="
    ax_prm.text(0.5, 0.5, param_text, fontsize=11, fontfamily='monospace', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=1.5', facecolor='#f8f9fa', edgecolor='#dee2e6'))

    # Сводная таблица
    ax_tbl.axis('off')
    ax_tbl.set_title("Performance & Timing", fontsize=16, fontweight='bold', pad=20)
    col_labels = ["Optimizer", "AUC", "Best Acc", "Final Acc", "Opt step (ms)", "Wall (s)", "Refresh"]

    table = ax_tbl.table(cellText=summary_stats, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 2.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e9ecef')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_part = f"_{CONFIG['TAG']}" if CONFIG.get("TAG") else ""
    filename = f"GNN_Ablation4_L{CONFIG['NUM_LAYERS']}{tag_part}_{timestamp}.png"
    filepath = os.path.join(CONFIG["PLOT_DIR"], filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n[+] Statistical plot saved to: {filepath}")
