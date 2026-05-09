import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import numpy as np
import concurrent.futures
import time

from optimizers import MatrixOptimizer

# ================= КОНФИГУРАЦИЯ =================
ADAM_LR = 0.01       
MUON_LR = 0.002      
NUM_WORKERS = 3
NUM_LAYERS = 6      
NUM_SEEDS = 25        
EPOCHS = 100         
# ================================================

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
                x = F.dropout(x, p=0.2, training=self.training)
        return x

def run_experiment(config):
    opt_name, seed = config
    start_time = time.time()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
    data = dataset[0].to(device)
    
    torch.manual_seed(seed)
    model = DeepGCN(dataset.num_features, 128, dataset.num_classes, NUM_LAYERS).to(device)
    attach_z_hooks(model)
    
    m_params = [p for n, p in model.named_parameters() if p.ndim == 2]
    o_params = [p for n, p in model.named_parameters() if p.ndim != 2]
    
    # Нематричные параметры всегда обучаем AdamW
    opt_other = torch.optim.AdamW(o_params, lr=ADAM_LR)

    if opt_name == "AdamW":
        opt_matrix = torch.optim.AdamW(m_params, lr=ADAM_LR)
    else:
        opt_matrix = MatrixOptimizer(
            m_params, 
            lr=MUON_LR, 
            use_newton=(opt_name == "Newton-Muon"),
            nm_refresh_interval=32,  # Для Full-batch графов нужно частое обновление!
            nm_beta=0.95,           # Меньше инерция для быстрой реакции     
            nm_gamma=0.2            
        )

    history = {'loss':[], 'val_acc': [], 'cond_nums':[]}
    
    for epoch in range(EPOCHS):
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
            
    elapsed = time.time() - start_time
    return opt_name, seed, history, elapsed

if __name__ == '__main__':
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    methods = ["AdamW", "Muon", "Newton-Muon"]
    tasks = [(m, s) for m in methods for s in range(NUM_SEEDS)]

    print(f"=== GNN Benchmark ===")
    print(f"AdamW LR: {ADAM_LR} | Muon LR: {MUON_LR}")
    print(f"Workers: {NUM_WORKERS} | Layers: {NUM_LAYERS} | Seeds: {NUM_SEEDS}")
    print("-" * 75)

    processed = {m: {'acc': [], 'loss': [], 'cond':[]} for m in methods}
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(run_experiment, task): task for task in tasks}
        
        for future in concurrent.futures.as_completed(futures):
            opt_name, seed, hist, elapsed = future.result()
            
            processed[opt_name]['acc'].append(hist['val_acc'])
            processed[opt_name]['loss'].append(hist['loss'])
            processed[opt_name]['cond'].append(hist['cond_nums'])
            
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
        # 95% Confidence Interval
        mean = np.mean(data_array)
        ci = 1.96 * np.std(data_array) / np.sqrt(len(data_array))
        return mean, ci

    for opt in methods:
        acc_matrix = np.array(processed[opt]['acc']) # [NUM_SEEDS, EPOCHS]
        
        # 1. AUC (Normalized) - Насколько быстро растет график
        auc_per_seed = np.trapz(acc_matrix, axis=1) / (EPOCHS - 1)
        auc_m, auc_ci = calc_mean_ci(auc_per_seed)
        
        # 2. Best Accuracy - Пиковая точность
        best_per_seed = np.max(acc_matrix, axis=1)
        best_m, best_ci = calc_mean_ci(best_per_seed)
        
        # 3. Final Accuracy - Точность в конце (нет ли деградации)
        final_per_seed = acc_matrix[:, -1]
        final_m, final_ci = calc_mean_ci(final_per_seed)
        
        summary_stats.append([
            opt,
            f"{auc_m:.4f} ± {auc_ci:.4f}",
            f"{best_m:.4f} ± {best_ci:.4f}",
            f"{final_m:.4f} ± {final_ci:.4f}"
        ])

    # ================= ВИЗУАЛИЗАЦИЯ (Сетка 2x2) =================
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"GNN Benchmark ({NUM_LAYERS} Layers) | {NUM_SEEDS} Seeds", fontsize=18, y=0.95)
    
    colors = {'AdamW': '#1f77b4', 'Muon': '#ff7f0e', 'Newton-Muon': '#2ca02c'}
    epochs = np.arange(EPOCHS)
    
    ax_loss, ax_acc = axs[0, 0], axs[0, 1]
    ax_cond, ax_tbl = axs[1, 0], axs[1, 1]

    for opt in methods:
        data = processed[opt]
        acc_m = np.mean(data['acc'], axis=0)
        acc_s = np.std(data['acc'], axis=0)
        loss_m = np.mean(data['loss'], axis=0)
        cond_m = np.mean(data['cond'], axis=0)
        
        # Plot Loss
        ax_loss.plot(epochs, loss_m, label=opt, color=colors[opt], linewidth=2)
        
        # Plot Acc
        ax_acc.plot(epochs, acc_m, color=colors[opt], label=opt, linewidth=2)
        ax_acc.fill_between(epochs, acc_m-acc_s, acc_m+acc_s, color=colors[opt], alpha=0.15)
        
        # Plot Cond (Only Newton-Muon usually has non-trivial condition numbers)
        if opt == "Newton-Muon" and not np.all(cond_m == 1.0):
            ax_cond.plot(epochs, cond_m, color='red', label='Newton-Muon')

    # Formatting axes
    ax_loss.set_yscale('log')
    ax_loss.set_title("Train Loss", fontsize=14)
    ax_loss.set_xlabel("Epoch"); ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, which="both", ls="--", alpha=0.5)
    ax_loss.legend()

    ax_acc.set_title("Validation Accuracy", fontsize=14)
    ax_acc.set_xlabel("Epoch"); ax_acc.set_ylabel("Accuracy")
    ax_acc.grid(True, ls="--", alpha=0.5)
    ax_acc.legend()

    ax_cond.set_yscale('log')
    ax_cond.set_title(r'Condition Number $\kappa(Z Z^T + \gamma I)$', fontsize=14)
    ax_cond.set_xlabel("Epoch")
    ax_cond.grid(True, which="both", ls="--", alpha=0.5)
    ax_cond.legend()

    # Построение таблицы статистики
    ax_tbl.axis('off')
    ax_tbl.set_title("Performance Metrics (95% CI)", fontsize=14, pad=20)
    
    col_labels = ["Optimizer", "AUC (Conv. Speed)", "Best Acc", "Final Acc"]
    table = ax_tbl.table(cellText=summary_stats, 
                         colLabels=col_labels, 
                         loc='center', 
                         cellLoc='center')
    
    # Стилизация таблицы
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2.5) # Ширина, Высота ячеек
    
    # Выделяем заголовки таблицы жирным
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#f0f0f0')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    filename = f'gnn_results_stats_adam_{ADAM_LR}_muon_{MUON_LR}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n[+] Statistical plot successfully saved to {filename}")