import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import numpy as np

from optimizers import MatrixOptimizer
from utils import attach_z_hooks

class DeepGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=6):
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
                # Уменьшим dropout, чтобы лучше видеть чистый loss
                x = F.dropout(x, p=0.2, training=self.training) 
        return x

def train_single_run(opt_name: str, lr: float, data, dataset, device):
    torch.manual_seed(42) # Фиксируем сид для честности
    model = DeepGCN(dataset.num_features, 128, dataset.num_classes, num_layers=6).to(device)
    attach_z_hooks(model)
    
    matrix_params = [p for n, p in model.named_parameters() if p.ndim == 2]
    other_params = [p for n, p in model.named_parameters() if p.ndim != 2]

    # AdamW для bias параметров
    optimizer_other = torch.optim.AdamW(other_params, lr=0.01, weight_decay=5e-4)

    if opt_name == "AdamW":
        optimizer_matrix = torch.optim.AdamW(matrix_params, lr=lr, weight_decay=5e-4)
    elif opt_name == "Muon":
        optimizer_matrix = MatrixOptimizer(matrix_params, lr=lr, use_newton=False)
    elif opt_name == "Newton-Muon":
        optimizer_matrix = MatrixOptimizer(matrix_params, lr=lr, use_newton=True)
    
    history = {'loss': [], 'val_acc': []}
    best_val_acc = 0.0

    model.train()
    for epoch in range(1, 201):
        optimizer_matrix.zero_grad()
        optimizer_other.zero_grad()
        
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        
        optimizer_matrix.step()
        optimizer_other.step()
        
        history['loss'].append(loss.item())
        
        # Считаем валидацию
        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(dim=-1)
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
        model.train()
        
        history['val_acc'].append(val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
    return history, best_val_acc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
    data = dataset[0].to(device)

    # Задаем сетки поиска LR
    grids = {
        "AdamW": [0.001, 0.005, 0.01, 0.05],
        # Muon и Newton-Muon часто требуют LR x5-x10 раз больше, чем Adam
        "Muon": [0.01, 0.05, 0.1, 0.2, 0.5],
        "Newton-Muon": [0.01, 0.05, 0.1, 0.2, 0.5]
    }

    best_results = {}

    for opt_name, lr_list in grids.items():
        print(f"\n--- Sweeping {opt_name} ---")
        best_acc = 0
        best_hist = None
        best_lr = 0
        
        for lr in lr_list:
            print(f"  Testing LR={lr}...")
            hist, acc = train_single_run(opt_name, lr, data, dataset, device)
            print(f"  -> Final Val Acc: {acc:.4f} (Final Loss: {hist['loss'][-1]:.4f})")
            
            if acc > best_acc:
                best_acc = acc
                best_hist = hist
                best_lr = lr
                
        best_results[opt_name] = {
            'hist': best_hist,
            'lr': best_lr,
            'acc': best_acc
        }
        print(f"[*] Best {opt_name}: LR={best_lr} with Val Acc={best_acc:.4f}")

    # --- РИСУЕМ ГРАФИКИ ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'AdamW': 'blue', 'Muon': 'orange', 'Newton-Muon': 'green'}

    for opt_name, res in best_results.items():
        hist = res['hist']
        label = f"{opt_name} (lr={res['lr']})"
        
        # Сглаживаем Loss для красоты (Exponential Moving Average)
        losses = np.array(hist['loss'])
        smoothed_losses = np.zeros_like(losses)
        smoothed_losses[0] = losses[0]
        for i in range(1, len(losses)):
            smoothed_losses[i] = 0.9 * smoothed_losses[i-1] + 0.1 * losses[i]
            
        ax1.plot(smoothed_losses, label=label, color=colors[opt_name], alpha=0.8)
        ax2.plot(hist['val_acc'], label=label, color=colors[opt_name], alpha=0.8)

    ax1.set_title("Train Loss (Smoothed)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_yscale('log') # Логарифмическая шкала для лосса очень полезна!
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gnn_results.png', dpi=300)
    print("\n[+] Plot saved to gnn_results.png")

if __name__ == '__main__':
    main()