"""CIFAR-10 ablation для Newton-Muon — 4 армы NM + Muon, paired-diff vs NM-base.

Структурно повторяет train_experimental.py, отличия:
- Архитектура — маленькая CNN.
- Pre-hook на Conv2d разворачивает вход через F.unfold и кладёт в p.Z как (in*kH*kW, B*L_out).
- Pre-hook на Linear ставит p.Z = x.T (как в GNN).
- optimizers_new.MatrixOptimizer теперь сам разворачивает 4D веса.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR

from optimizers_new import MatrixOptimizer

CONFIG = {
    "ADAM_LR": 0.003,
    "NEWTON_MUON_LR": 0.005,
    "NUM_WORKERS": 3,
    "BATCH_SIZE": 128,
    "NUM_SEEDS": 50,
    "EPOCHS": 20,
    "WEIGHT_DECAY": 5e-4,

    # Калибровка: при cosine lr 0.005 → 0.0005 средний lr ≈ 0.00275.
    # threshold ≈ lr_avg × refresh_interval → одинаковая частота refresh
    # у path-based и fixed-interval (50 шагов).
    "NM_UPDATE_THRESHOLD": 0.14,
    "NM_BETA": 0.85,
    "NM_GAMMA": 0.3,
    "NM_REFRESH_INTERVAL": 50,   # измеряется в opt.step() => мини-батчах

    "DATA_ROOT": "/tmp/CIFAR10",
    "PLOT_DIR": "experiments_plots_our",
}

# (use_newton, new_gamma, new_interval)
METHOD_FLAGS = {
    "Muon":        (False, False, False),
    "NM-base":     (True,  False, False),
    "NM-gamma":    (True,  True,  False),
    "NM-interval": (True,  False, True),
    "NM-both":     (True,  True,  True),
}


class SmallCNN(nn.Module):
    """3-conv + FC, без BatchNorm чтобы упростить какой параметр где. Размер по width."""
    def __init__(self, num_classes=10, width=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3,        width,   3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(width,    width*2, 3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(width*2,  width*4, 3, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width*4, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = self.pool(x)
        x = F.relu(self.conv2(x)); x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.gap(x).flatten(1)
        return self.fc(x)


def attach_z_hooks(model):
    """Conv2d: p.Z = (in*kH*kW, B*L_out) через F.unfold.
       Linear: p.Z = x.T (как в GNN)."""
    for mod in model.modules():
        if isinstance(mod, nn.Conv2d):
            kH, kW = mod.kernel_size
            stride = mod.stride
            padding = mod.padding
            dilation = mod.dilation

            def make_conv_hook(_mod):
                def hook(m, inputs):
                    x = inputs[0].detach()  # (B, in, H, W)
                    patches = F.unfold(x, kernel_size=m.kernel_size,
                                       stride=m.stride, padding=m.padding,
                                       dilation=m.dilation)
                    # patches: (B, in*kH*kW, L) → (in*kH*kW, B*L)
                    p_in = patches.size(1)
                    patches = patches.permute(1, 0, 2).reshape(p_in, -1)
                    m.weight.Z = patches.float()
                return hook
            mod.register_forward_pre_hook(make_conv_hook(mod))
        elif isinstance(mod, nn.Linear):
            def make_lin_hook(_mod):
                def hook(m, inputs):
                    x = inputs[0].detach()
                    if x.ndim > 2:
                        x = x.view(-1, x.shape[-1])
                    m.weight.Z = x.T.float()  # (in_features, B)
                return hook
            mod.register_forward_pre_hook(make_lin_hook(mod))


def get_loaders(config, seed):
    norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    if config.get("AUG", True):
        tf_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), norm,
        ])
    else:
        tf_train = transforms.Compose([transforms.ToTensor(), norm])
    tf_test = transforms.Compose([transforms.ToTensor(), norm])
    train = datasets.CIFAR10(root=config["DATA_ROOT"], train=True, download=True, transform=tf_train)
    test  = datasets.CIFAR10(root=config["DATA_ROOT"], train=False, download=True, transform=tf_test)
    subset_n = config.get("SUBSET", 0)
    if subset_n > 0 and subset_n < len(train):
        rng = np.random.default_rng(0)  # фикс. подвыборка, чтобы сравнение было честным между сидами
        idx = rng.choice(len(train), size=subset_n, replace=False)
        train = torch.utils.data.Subset(train, idx.tolist())
    g = torch.Generator(); g.manual_seed(seed)
    return (
        DataLoader(train, batch_size=config["BATCH_SIZE"], shuffle=True,
                   num_workers=0, generator=g, drop_last=True),
        DataLoader(test, batch_size=512, shuffle=False, num_workers=0),
    )


def run_experiment(task_args):
    opt_name, seed, gpu_id, config = task_args
    start_time = time.time()

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f'cuda:{gpu_id}')
    else:
        device = torch.device('cpu')

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    train_loader, test_loader = get_loaders(config, seed)

    model = SmallCNN(num_classes=10, width=config.get("WIDTH", 64)).to(device)
    attach_z_hooks(model)

    # 2D/4D веса → matrix-оптимизатор; bias и прочее (1D) → AdamW.
    m_params = [p for n, p in model.named_parameters() if p.ndim >= 2]
    o_params = [p for n, p in model.named_parameters() if p.ndim < 2]

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

    total_steps = config["EPOCHS"] * len(train_loader)
    eta_ratio = config.get("ETA_MIN_RATIO", 0.1)
    sched_other = CosineAnnealingLR(opt_other, T_max=total_steps, eta_min=config["ADAM_LR"] * eta_ratio)
    sched_matrix = CosineAnnealingLR(opt_matrix, T_max=total_steps, eta_min=config["NEWTON_MUON_LR"] * eta_ratio)

    history = {'loss': [], 'val_acc': []}
    opt_step_time = 0.0
    opt_step_count = 0
    is_cuda = (device.type == 'cuda')
    skipped_first = False

    log_every = max(1, config["EPOCHS"] // 5)  # ~5 промежуточных строк на сид
    for epoch in range(config["EPOCHS"]):
        ep_start = time.time()
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            opt_matrix.zero_grad(); opt_other.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
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
            sched_matrix.step()
            sched_other.step()

            epoch_loss += loss.item()
        history['loss'].append(epoch_loss / max(len(train_loader), 1))

        # eval
        model.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
                pred = model(xb).argmax(dim=-1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        acc = correct / total
        history['val_acc'].append(acc)

        if (epoch % log_every == 0) or (epoch == config["EPOCHS"] - 1):
            ep_dt = time.time() - ep_start
            print(f"  [{opt_name:11} s{seed:03d} gpu{gpu_id}] ep {epoch+1:02d}/{config['EPOCHS']} "
                  f"loss={history['loss'][-1]:.3f} acc={acc:.4f} dt={ep_dt:.1f}s", flush=True)

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
    parser.add_argument('--seeds', type=int, default=CONFIG["NUM_SEEDS"])
    parser.add_argument('--epochs', type=int, default=CONFIG["EPOCHS"])
    parser.add_argument('--methods', type=str, default=",".join(METHOD_FLAGS.keys()))
    parser.add_argument('--eta_min_ratio', type=float, default=0.1)
    parser.add_argument('--tag', type=str, default="")
    parser.add_argument('--gpus', type=str, default="0",
                        help="comma-separated GPU indices; one worker per GPU")
    parser.add_argument('--nm_gamma', type=float, default=None, help="override NM_GAMMA")
    parser.add_argument('--nm_refresh', type=int, default=None, help="override NM_REFRESH_INTERVAL")
    parser.add_argument('--nm_threshold', type=float, default=None, help="override NM_UPDATE_THRESHOLD")
    parser.add_argument('--width', type=int, default=64, help="CNN base width (channels)")
    parser.add_argument('--lr', type=float, default=None, help="override NEWTON_MUON_LR")
    parser.add_argument('--batch', type=int, default=128, help="batch size")
    parser.add_argument('--subset', type=int, default=0, help="train subset size (0 = full)")
    parser.add_argument('--no_aug', action='store_true', help="disable augmentation")
    args = parser.parse_args()
    CONFIG["NUM_SEEDS"] = args.seeds
    CONFIG["EPOCHS"] = args.epochs
    CONFIG["ETA_MIN_RATIO"] = args.eta_min_ratio
    CONFIG["TAG"] = args.tag
    if args.nm_gamma is not None:
        CONFIG["NM_GAMMA"] = args.nm_gamma
    if args.nm_refresh is not None:
        CONFIG["NM_REFRESH_INTERVAL"] = args.nm_refresh
    if args.nm_threshold is not None:
        CONFIG["NM_UPDATE_THRESHOLD"] = args.nm_threshold
    if args.lr is not None:
        CONFIG["NEWTON_MUON_LR"] = args.lr
    CONFIG["WIDTH"] = args.width
    CONFIG["BATCH_SIZE"] = args.batch
    CONFIG["SUBSET"] = args.subset
    CONFIG["AUG"] = not args.no_aug

    gpu_ids = [int(g.strip()) for g in args.gpus.split(",") if g.strip()]
    CONFIG["NUM_WORKERS"] = len(gpu_ids)  # 1 worker per GPU

    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    os.makedirs(CONFIG["PLOT_DIR"], exist_ok=True)
    # Чтобы воркеры не качали датасет конкурентно — однократный download в main.
    _ = datasets.CIFAR10(root=CONFIG["DATA_ROOT"], train=True, download=True)
    _ = datasets.CIFAR10(root=CONFIG["DATA_ROOT"], train=False, download=True)

    methods = [m.strip() for m in args.methods.split(",") if m.strip() in METHOD_FLAGS]
    # Round-robin tasks по GPU.
    tasks = []
    for idx, (m, s) in enumerate([(m, s) for m in methods for s in range(CONFIG["NUM_SEEDS"])]):
        gpu = gpu_ids[idx % len(gpu_ids)]
        tasks.append((m, s, gpu, CONFIG))

    print(f"=== CIFAR-10 Ablation | {len(methods)} arms: {methods} ===")
    print(f"Seeds: {CONFIG['NUM_SEEDS']} | Epochs: {CONFIG['EPOCHS']} | Batch: {CONFIG['BATCH_SIZE']} | GPUs: {gpu_ids}")
    print("-" * 95)

    processed = {m: {
        'acc': [None] * CONFIG["NUM_SEEDS"],
        'loss': [None] * CONFIG["NUM_SEEDS"],
        'freq': [None] * CONFIG["NUM_SEEDS"],
        'g_frob': [None] * CONFIG["NUM_SEEDS"],
        'g_trace': [None] * CONFIG["NUM_SEEDS"],
        'opt_ms': [None] * CONFIG["NUM_SEEDS"],
        'wall_t': [None] * CONFIG["NUM_SEEDS"],
    } for m in methods}

    with concurrent.futures.ProcessPoolExecutor(max_workers=CONFIG["NUM_WORKERS"]) as executor:
        futures = {executor.submit(run_experiment, task): task for task in tasks}
        for future in concurrent.futures.as_completed(futures):
            opt_name, seed, hist, elapsed = future.result()
            processed[opt_name]['acc'][seed] = hist['val_acc']
            processed[opt_name]['loss'][seed] = hist['loss']
            processed[opt_name]['freq'][seed] = hist['nm_freq']
            processed[opt_name]['g_frob'][seed] = hist['nm_g_frob']
            processed[opt_name]['g_trace'][seed] = hist['nm_g_trace']
            processed[opt_name]['opt_ms'][seed] = hist['opt_step_mean_ms']
            processed[opt_name]['wall_t'][seed] = elapsed

            print(f"[{opt_name:11}] Seed {seed:03d} | wall {elapsed:>5.1f}s | "
                  f"opt {hist['opt_step_mean_ms']:>5.1f}ms/step | "
                  f"Acc: {hist['val_acc'][-1]:.4f} | Freq: {hist['nm_freq']:.1f}st | "
                  f"γ {hist['nm_g_frob']:.1e}(F)/{hist['nm_g_trace']:.1e}(Tr)")

    print("-" * 95)

    def calc_mean_ci(arr):
        arr = np.asarray(arr, dtype=float)
        return float(np.mean(arr)), float(1.96 * np.std(arr, ddof=1) / np.sqrt(len(arr)))

    best_acc = {m: np.max(np.array(processed[m]['acc']), axis=1) for m in methods}
    baseline = "NM-base" if "NM-base" in methods else methods[0]

    paired_lines = []
    print(f"\n[STAT] Paired diff in Best Accuracy vs {baseline}:")
    for m in methods:
        if m == baseline:
            continue
        diffs = best_acc[m] - best_acc[baseline]
        mean_d = float(np.mean(diffs))
        ci_d = float(1.96 * np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
        tstat = mean_d / (np.std(diffs, ddof=1) / np.sqrt(len(diffs)) + 1e-12)
        line = f"  {m:11} - {baseline}: {mean_d:+.4f} ± {ci_d:.4f}  (t={tstat:+.2f})"
        print(line); paired_lines.append(line)
    print()

    summary_stats = []
    for m in methods:
        acc_matrix = np.array(processed[m]['acc'])
        auc_per_seed = np.trapezoid(acc_matrix, axis=1) / (CONFIG["EPOCHS"] - 1)
        auc_m_, auc_ci = calc_mean_ci(auc_per_seed)
        best_m_, best_ci = calc_mean_ci(best_acc[m])
        final_m_, final_ci = calc_mean_ci(acc_matrix[:, -1])
        opt_ms_m, opt_ms_ci = calc_mean_ci(processed[m]['opt_ms'])
        wall_t_m, wall_t_ci = calc_mean_ci(processed[m]['wall_t'])
        freq_m_ = float(np.mean(processed[m]['freq']))
        summary_stats.append([
            m,
            f"{auc_m_:.4f} ± {auc_ci:.4f}",
            f"{best_m_:.4f} ± {best_ci:.4f}",
            f"{final_m_:.4f} ± {final_ci:.4f}",
            f"{opt_ms_m:.2f} ± {opt_ms_ci:.2f}",
            f"{wall_t_m:.1f} ± {wall_t_ci:.1f}",
            f"every {freq_m_:.1f} st",
        ])

    fig, axs = plt.subplots(2, 2, figsize=(18, 11))
    fig.suptitle(f"CIFAR-10 SmallCNN Ablation | {CONFIG['NUM_SEEDS']} seeds", fontsize=18, fontweight='bold')

    colors = {
        'Muon':        '#7f7f7f',
        'NM-base':     '#1f77b4',
        'NM-gamma':    '#ff7f0e',
        'NM-interval': '#9467bd',
        'NM-both':     '#2ca02c',
    }
    epochs = np.arange(CONFIG["EPOCHS"])

    ax_loss = axs[0, 0]; ax_acc = axs[0, 1]
    ax_prm  = axs[1, 0]; ax_tbl = axs[1, 1]

    for m in methods:
        data = processed[m]
        acc_m_ = np.mean(data['acc'], axis=0); acc_s_ = np.std(data['acc'], axis=0)
        loss_m_ = np.mean(data['loss'], axis=0)
        ax_loss.plot(epochs, loss_m_, label=m, color=colors[m], linewidth=2)
        ax_acc.plot(epochs, acc_m_, color=colors[m], label=m, linewidth=2)
        ax_acc.fill_between(epochs, acc_m_-acc_s_, acc_m_+acc_s_, color=colors[m], alpha=0.10)

    ax_loss.set_yscale('log'); ax_loss.set_title("Train Loss"); ax_loss.set_xlabel("Epoch"); ax_loss.grid(alpha=0.5); ax_loss.legend()
    ax_acc.set_title("Test Accuracy"); ax_acc.set_xlabel("Epoch"); ax_acc.grid(alpha=0.5); ax_acc.legend()

    ax_prm.axis('off')
    ax_prm.set_title(f"Config & Paired Diffs (vs {baseline})", fontsize=14, fontweight='bold')
    param_text = "\n".join([f"{k}: {v}" for k, v in CONFIG.items() if k not in ("PLOT_DIR", "DATA_ROOT")])
    param_text += "\n\n========================="
    param_text += "\nPaired Diff (Best Acc):"
    for ln in paired_lines:
        param_text += "\n" + ln
    param_text += "\n========================="
    ax_prm.text(0.5, 0.5, param_text, fontsize=11, fontfamily='monospace', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=1.5', facecolor='#f8f9fa', edgecolor='#dee2e6'))

    ax_tbl.axis('off')
    ax_tbl.set_title("Performance & Timing", fontsize=14, fontweight='bold')
    col_labels = ["Optimizer", "AUC", "Best Acc", "Final Acc", "Opt step (ms)", "Wall (s)", "Refresh"]
    table = ax_tbl.table(cellText=summary_stats, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.1, 2.2)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold'); cell.set_facecolor('#e9ecef')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag_part = f"_{CONFIG['TAG']}" if CONFIG.get("TAG") else ""
    filename = f"CIFAR_Ablation_smallCNN{tag_part}_{timestamp}.png"
    filepath = os.path.join(CONFIG["PLOT_DIR"], filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n[+] Statistical plot saved to: {filepath}")
