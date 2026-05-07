import torch
import torch.nn as nn

def attach_z_hooks(model: nn.Module):
    """
    Рекурсивно проходит по модели и вешает pre-forward hook на все nn.Linear слои.
    Newton-Muon требует матрицу активаций входа (Z), чтобы посчитать (Z * Z.T)^-1.
    """
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Хук срабатывает прямо перед тем, как слой сделает y = x @ W^T
            def pre_hook(mod, inputs):
                x = inputs[0].detach() # Отвязываем от графа градиентов
                
                # Z в статье имеет размерность [Dim_in, N_samples]. 
                # inputs[0] обычно имеет размер [Batch, Dim_in]. 
                # Поэтому мы делаем reshape и транспонируем.
                if x.ndim > 2:
                    x = x.reshape(-1, x.shape[-1])
                
                # Сохраняем Z прямо в параметр весов этого слоя!
                mod.weight.Z = x.T 

            handle = module.register_forward_pre_hook(pre_hook)
            handles.append(handle)
            print(f"Hook attached to: {name} (shape: {module.weight.shape})")
    
    return handles