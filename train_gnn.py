import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import concurrent.futures

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv

from optimizers import MatrixOptimizer

# ================= ЕДИНАЯ КОНФИГУРАЦИЯ =================
CONFIG = {
    # Архитектура и обучение
    "ADAM_LR": 0.01,
    "MUON_LR": 0.002,
    "NUM_WORKERS": 3,
    "NUM_LAYERS": 5,
    "HIDDEN_DIM": 128,
    "NUM_SEEDS": 50,
    "EPOCHS": 120,
    "WEIGHT_DECAY": 0.01,
    "DROPOUT": 0.2,
    
    # Специфичные параметры Newton-Muon
    "NM_REFRESH_INTERVAL": 6, # Как часто обновляем предобуславливатель (эпохи)
    "NM_BETA": 0.85,          # EMA коэффициент
    "NM_GAMMA": 0.1,          # Ridge penalty
    
    # Папка для графиков
    "TRACKING_INTERVAL": 5, 
    "PLOT_DIR": "experiments_plots"
}
# ========================================================

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
    weight_snapshots = [] # Храним кортежи (epoch, weight_matrix)
    
    opt_other = torch.optim.AdamW(o_params, lr=config["ADAM_LR"], weight_decay=config["WEIGHT_DECAY"])

    if opt_name == "AdamW":
        opt_matrix = torch.optim.AdamW(m_params, lr=config["ADAM_LR"], weight_decay=config["WEIGHT_DECAY"])
    else:
        opt_matrix = MatrixOptimizer(
            m_params, 
            lr=config["MUON_LR"], 
            weight_decay=config["WEIGHT_DECAY"],
            use_newton=(opt_name == "Newton-Muon"),
            nm_refresh_interval=config["NM_REFRESH_INTERVAL"], 
            nm_beta=config["NM_BETA"],           
            nm_gamma=config["NM_GAMMA"]            
        )

    history = {'loss': [], 'val_acc': [], 'cond_nums': []}
    
    for epoch in range(config["EPOCHS"]):
        model.train()
        opt_matrix.zero_grad()
        opt_other.zero_grad()
        
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        m_info = opt_matrix.step()
        opt_other.step()
        
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
            
        # СОХРАНЯЕМ ВЕСА ТОЛЬКО РАЗ В N ЭПОХ
        if epoch % config["TRACKING_INTERVAL"] == 0 or epoch == config["EPOCHS"] - 1:
            weight_snapshots.append((epoch, tracked_param.detach().cpu().clone()))
            
    # ================= ПОСТ-РАСЧЕТ АНИЗОТРОПИИ =================
    w_final = weight_snapshots[-1][1]
    snap_values = {}
    
    # Считаем тяжелую математику только для сохраненных эпох
    for ep, w_t in weight_snapshots[:-1]:
        diff = w_t - w_final
        sigma_w = diff @ diff.T
        
        eigs = torch.linalg.eigvalsh(sigma_w.to(torch.float32))
        eigs = torch.relu(eigs) 
        top1_ratio = eigs[-1] / (eigs.sum() + 1e-12)
        snap_values[ep] = top1_ratio.item()
        
    # Разворачиваем значения на все эпохи (Forward-fill) для графика
    anisotropy_full = []
    curr_val = snap_values.get(0, 0.0)
    for epoch in range(config["EPOCHS"]):
        if epoch in snap_values:
            curr_val = snap_values[epoch]
        anisotropy_full.append(curr_val)
        
    history['anisotropy'] = anisotropy_full
    # ============================================================

    elapsed = time.time() - start_time
    return opt_name, seed, history, elapsed

if __name__ == '__main__':
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    os.makedirs(CONFIG["PLOT_DIR"], exist_ok=True)

    methods = ["AdamW", "Muon", "Newton-Muon"]
    tasks = [(m, s, CONFIG) for m in methods for s in range(CONFIG["NUM_SEEDS"])]

    print(f"=== GNN Benchmark ===")
    print(f"Layers: {CONFIG['NUM_LAYERS']} | Seeds: {CONFIG['NUM_SEEDS']} | Epochs: {CONFIG['EPOCHS']}")
    print(f"Tracking Anisotropy every {CONFIG['TRACKING_INTERVAL']} epochs.")
    print("-" * 75)

    processed = {m: {'acc': [], 'loss': [], 'cond': [], 'anis': []} for m in methods}
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=CONFIG["NUM_WORKERS"]) as executor:
        futures = {executor.submit(run_experiment, task): task for task in tasks}
        
        for future in concurrent.futures.as_completed(futures):
            opt_name, seed, hist, elapsed = future.result()
            
            processed[opt_name]['acc'].append(hist['val_acc'])
            processed[opt_name]['loss'].append(hist['loss'])
            processed[opt_name]['cond'].append(hist['cond_nums'])
            processed[opt_name]['anis'].append(hist['anisotropy'])
            
            f_loss = hist['loss'][-1]
            f_acc = hist['val_acc'][-1]
            max_c = np.max(hist['cond_nums'])
            
            cond_str = f"| Max \u03BA: {max_c:.1e}" if opt_name == "Newton-Muon" else ""
            print(f"[{opt_name:12}] Seed {seed:02d} | {elapsed:>4.1f}s | Acc: {f_acc:.4f} | Loss: {f_loss:.4f} {cond_str}")

    print("-" * 75)
    print("Computing statistics and generating plot...")

    # ================= РАСЧЕТ СТАТИСТИКИ =================
    summary_stats = []
    def calc_mean_ci(data_array):
        mean = np.mean(data_array)
        ci = 1.96 * np.std(data_array) / np.sqrt(len(data_array))
        return mean, ci

    for opt in methods:
        acc_matrix = np.array(processed[opt]['acc'])
        
        auc_per_seed = np.trapz(acc_matrix, axis=1) / (CONFIG["EPOCHS"] - 1)
        auc_m, auc_ci = calc_mean_ci(auc_per_seed)
        
        best_per_seed = np.max(acc_matrix, axis=1)
        best_m, best_ci = calc_mean_ci(best_per_seed)
        
        final_per_seed = acc_matrix[:, -1]
        final_m, final_ci = calc_mean_ci(final_per_seed)
        
        summary_stats.append([
            opt,
            f"{auc_m:.4f} ± {auc_ci:.4f}",
            f"{best_m:.4f} ± {best_ci:.4f}",
            f"{final_m:.4f} ± {final_ci:.4f}"
        ])

    # ================= ВИЗУАЛИЗАЦИЯ (Сетка 2x3) =================
    fig, axs = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle(f"GNN Optimization Analysis | {CONFIG['NUM_SEEDS']} Seeds", fontsize=20, y=0.96, fontweight='bold')
    
    colors = {'AdamW': '#1f77b4', 'Muon': '#ff7f0e', 'Newton-Muon': '#2ca02c'}
    epochs = np.arange(CONFIG["EPOCHS"])
    
    ax_loss = axs[0, 0]
    ax_acc  = axs[0, 1]
    ax_prm  = axs[0, 2]
    ax_cond = axs[1, 0]
    ax_anis = axs[1, 1]
    ax_tbl  = axs[1, 2]

    for opt in methods:
        data = processed[opt]
        acc_m = np.mean(data['acc'], axis=0)
        acc_s = np.std(data['acc'], axis=0)
        loss_m = np.mean(data['loss'], axis=0)
        cond_m = np.mean(data['cond'], axis=0)
        anis_m = np.mean(data['anis'], axis=0)
        anis_s = np.std(data['anis'], axis=0)
        
        ax_loss.plot(epochs, loss_m, label=opt, color=colors[opt], linewidth=2)
        
        ax_acc.plot(epochs, acc_m, color=colors[opt], label=opt, linewidth=2)
        ax_acc.fill_between(epochs, acc_m-acc_s, acc_m+acc_s, color=colors[opt], alpha=0.15)
        
        # Строим график анизотропии только по контрольным точкам (чтобы график не выглядел ступенчатым)
        tracked_epochs = np.arange(0, CONFIG["EPOCHS"], CONFIG["TRACKING_INTERVAL"])
        if len(tracked_epochs) > 0:
            ax_anis.plot(tracked_epochs, anis_m[tracked_epochs], color=colors[opt], label=opt, linewidth=2, marker='o', markersize=4)
            ax_anis.fill_between(tracked_epochs, anis_m[tracked_epochs]-anis_s[tracked_epochs], anis_m[tracked_epochs]+anis_s[tracked_epochs], color=colors[opt], alpha=0.1)

        if opt == "Newton-Muon" and not np.all(cond_m == 1.0):
            ax_cond.plot(epochs, cond_m, color='#d62728', label='Newton-Muon $\kappa$', linewidth=2)

    ax_loss.set_yscale('log'); ax_loss.set_title("Train Loss", fontsize=14)
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Cross Entropy")
    ax_loss.grid(True, which="both", ls="--", alpha=0.5); ax_loss.legend()

    ax_acc.set_title("Validation Accuracy", fontsize=14)
    ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy")
    ax_acc.grid(True, ls="--", alpha=0.5); ax_acc.legend()

    ax_cond.set_yscale('log'); ax_cond.set_title("Condition Number $\kappa(Z Z^T + \gamma I)$", fontsize=14)
    ax_cond.set_xlabel("Epoch"); ax_cond.set_ylabel("$\kappa$")
    ax_cond.grid(True, which="both", ls="--", alpha=0.5); ax_cond.legend()

    ax_anis.set_title(f"Weight Displacement Anisotropy ($\Sigma_W$) [Step={CONFIG['TRACKING_INTERVAL']}]", fontsize=14)
    ax_anis.set_xlabel("Epoch"); ax_anis.set_ylabel("Top-1 Eigenvalue Ratio")
    ax_anis.grid(True, ls="--", alpha=0.5); ax_anis.legend()
    ax_anis.text(0.05, 0.95, "Ratio $\to 1$: Extreme Anisotropy (Cigar shape)\nRatio $\to 0$: Isotropic (Spherical, Paper's Assumption)", 
                 transform=ax_anis.transAxes, fontsize=10, va='top', bbox=dict(boxstyle="round", alpha=0.1, facecolor="gray"))

    ax_prm.axis('off')
    ax_prm.set_title("Hyperparameters Configuration", fontsize=16, fontweight='bold', pad=10)
    param_text = "\n".join([f"{k}: {v}" for k, v in CONFIG.items() if k != "PLOT_DIR"])
    ax_prm.text(0.5, 0.5, param_text, fontsize=13, fontfamily='monospace', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=1.5', facecolor='#f8f9fa', edgecolor='#dee2e6'))

    ax_tbl.axis('off')
    ax_tbl.set_title("Performance Metrics (95% CI)", fontsize=16, fontweight='bold', pad=20)
    col_labels = ["Optimizer", "AUC (Conv. Speed)", "Best Acc", "Final Acc"]
    table = ax_tbl.table(cellText=summary_stats, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 3.0) 
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e9ecef')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"GNN_L{CONFIG['NUM_LAYERS']}_Ep{CONFIG['EPOCHS']}_ALR{CONFIG['ADAM_LR']}_MLR{CONFIG['MUON_LR']}_{timestamp}.png"
    filepath = os.path.join(CONFIG["PLOT_DIR"], filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"\n[+] Statistical plot successfully saved to: {filepath}")