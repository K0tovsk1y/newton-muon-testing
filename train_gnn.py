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
FIXED_LR = 0.02      # Заданный LR
NUM_WORKERS = 3      # Сколько параллельных процессов (чтобы не убить CPU)
NUM_LAYERS = 3      # Глубина сети (стресс-тест)
NUM_SEEDS = 12        # Сколько запусков на каждый оптимизатор
EPOCHS = 300         
# ================================================

# ПРАВИЛЬНЫЙ ХУК ДЛЯ PYTORCH GEOMETRIC
def attach_z_hooks(model: torch.nn.Module):
    attached_count = 0
    for name, module in model.named_modules():
        # Цепляемся напрямую к графовому слою
        if isinstance(module, GCNConv):
            def pre_hook(mod, inputs):
                # В GCNConv первый аргумент - это признаки узлов X
                x = inputs[0].detach()
                # Ищем 2D-веса внутри этого слоя и привязываем к ним Z
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
    
    device = torch.device('cuda:0')
    dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
    data = dataset[0].to(device)
    
    torch.manual_seed(seed)
    model = DeepGCN(dataset.num_features, 128, dataset.num_classes, NUM_LAYERS).to(device)
    attach_z_hooks(model)
    
    m_params =[p for n, p in model.named_parameters() if p.ndim == 2]
    o_params = [p for n, p in model.named_parameters() if p.ndim != 2]
    opt_other = torch.optim.AdamW(o_params, lr=0.01)

    if opt_name == "AdamW":
        opt_matrix = torch.optim.AdamW(m_params, lr=FIXED_LR)
    else:
        opt_matrix = MatrixOptimizer(
            m_params, 
            lr=FIXED_LR, 
            use_newton=(opt_name == "Newton-Muon"),
            nm_refresh_interval=5,  # Обновлять матрицу раз в 5 эпох (вернет скорость!)
            nm_beta=0.5,            # Легкое сглаживание, чтобы K не "дергалась"
            nm_gamma=2.0            # ОЧЕНЬ ВАЖНО: Увеличиваем ridge penalty!
                                    # Это не даст Newton-Muon'у усиливать шум в 1000 раз.
                                    # В статье (Figure 5) показано, что при росте gamma алгоритм
                                    # становится более стабильным и приближается к стандартоному Muon.
        )

    history = {'loss': [], 'val_acc': [], 'cond_nums':[]}
    
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
        
        # Если Newton-Muon вернул кондишн - сохраняем, иначе 1.0 (для AdamW и Muon)
        if m_info and "cond_nums" in m_info and len(m_info["cond_nums"]) > 0:
            history['cond_nums'].append(np.mean(m_info["cond_nums"]))
        else:
            history['cond_nums'].append(1.0)

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
    tasks =[(m, s) for m in methods for s in range(NUM_SEEDS)]

    print(f"=== GNN Benchmark (LR={FIXED_LR}) ===")
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
            print(f"[{opt_name:12}] Seed {seed} | {elapsed:>4.1f}s | Acc: {f_acc:.4f} | Loss: {f_loss:.4f} {cond_str}")

    print("-" * 75)
    print("Generating plot...")

    # --- ВИЗУАЛИЗАЦИЯ ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))
    fig.suptitle(f"GNN Benchmark (16 Layers) | Fixed LR = {FIXED_LR} | {NUM_SEEDS} Seeds", fontsize=16)
    
    colors = {'AdamW': 'blue', 'Muon': 'orange', 'Newton-Muon': 'green'}
    epochs = np.arange(EPOCHS)
    
    for opt in methods:
        data = processed[opt]
        acc_m = np.mean(data['acc'], axis=0)
        acc_s = np.std(data['acc'], axis=0)
        loss_m = np.mean(data['loss'], axis=0)
        cond_m = np.mean(data['cond'], axis=0)
        
        ax1.plot(epochs, loss_m, label=opt, color=colors[opt])
        ax1.set_yscale('log'); ax1.set_title("Train Loss")
        ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.legend()
        
        ax2.plot(epochs, acc_m, color=colors[opt], label=opt)
        ax2.fill_between(epochs, acc_m-acc_s, acc_m+acc_s, color=colors[opt], alpha=0.15)
        ax2.set_title("Val Accuracy")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.legend()
        
        # График Condition Number (только если это Newton-Muon и данные реальные)
        if opt == "Newton-Muon" and not np.all(cond_m == 1.0):
            ax3.plot(epochs, cond_m, color='red', label='Newton-Muon')
            ax3.set_yscale('log')
            ax3.set_title("Condition Number $kappa(Z Z^T + gamma I)$")
            ax3.set_xlabel("Epoch"); ax3.legend()

    plt.tight_layout()
    filename = f'gnn_results_seeds_{NUM_SEEDS}_layers_{NUM_LAYERS}_epochs_{EPOCHS}_lr_{FIXED_LR}.png'
    plt.savefig(filename, dpi=300)
    print(f"[+] Plot successfully saved to {filename}")