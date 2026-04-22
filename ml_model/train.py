import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import TrafficDataset
from stgcn import STGCN_Prototype


def get_congestion_level(speed):
    level = torch.ones_like(speed) * 3
    level[speed >= 20] = 2
    level[speed >= 40] = 1
    return level

def evaluate(model, loader, criterion, adj, mean=0.0, std=1.0):
    """Returns average MSE, MAE, RMSE, and congestion accuracy on a dataloader."""
    model.eval()
    total_mse = total_mae = 0.0
    correct_levels = 0
    total_items = 0
    n = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            preds = model(batch_x, adj)
            batch_n = batch_x.size(0)
            total_mse += criterion(preds, batch_y).item() * batch_n
            total_mae += torch.mean(torch.abs(preds - batch_y)).item() * batch_n
            n += batch_n
            
            # Congestion Accuracy logic
            preds_unnorm = preds * std + mean
            y_unnorm = batch_y * std + mean
            
            preds_lvl = get_congestion_level(preds_unnorm)
            y_lvl = get_congestion_level(y_unnorm)
            
            correct_levels += (preds_lvl == y_lvl).sum().item()
            total_items += y_lvl.numel()
            
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    avg_mse  = total_mse / n
    avg_mae  = total_mae / n
    avg_rmse = avg_mse ** 0.5
    accuracy = (correct_levels / total_items) * 100.0 if total_items > 0 else 0.0
    return avg_mse, avg_mae, avg_rmse, accuracy


def train():
    print("=" * 55)
    print("  GeoAI – STGCN Training Pipeline")
    print("=" * 55)
    
    # ── Paths ────────────────────────────────────────────────
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir    = os.path.join(current_dir, '..', 'data_collection')
    model_path  = os.path.join(current_dir, 'stgcn_model.pth')
    
    # ── Hyperparameters ──────────────────────────────────────
    # WINDOW_SIZE: 增加到 6 讓模型看 30 分鐘歷史（每步 5 分鐘），捕捉更多時序趨勢
    WINDOW_SIZE     = 6
    PREDICT_SIZE    = 1
    # BATCH_SIZE: 提高減少梯度估計噪聲，加快穩定收斂
    BATCH_SIZE      = 16
    # EPOCHS: 拉長讓 lr scheduler 有足夠空間逐步下降
    EPOCHS          = 100
    # LR: 降低初始學習率，避免在最優解周圍震盪
    LR              = 0.001
    # HIDDEN_CHANNELS: 加大模型容量以捕捉更複雜的空間-時間依賴
    HIDDEN_CHANNELS = 64
    TRAIN_RATIO     = 0.7
    VAL_RATIO       = 0.15
    # TEST_RATIO = 0.15 (remainder)
    
    # ── Dataset & split ──────────────────────────────────────
    dataset   = TrafficDataset(data_dir,
                               window_size=WINDOW_SIZE,
                               predict_size=PREDICT_SIZE)
    total     = len(dataset)
    n_train   = max(1, int(total * TRAIN_RATIO))
    n_val     = max(1, int(total * VAL_RATIO))
    n_test    = total - n_train - n_val
    
    print(f"\nDataset  : {total} samples  "
          f"(train={n_train}, val={n_val}, test={n_test})")
    
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)
    
    # ── Model ────────────────────────────────────────────────
    num_nodes = dataset.features.shape[0]
    adj       = dataset.adj
    
    model = STGCN_Prototype(
        num_nodes=num_nodes,
        in_channels=1,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=PREDICT_SIZE,
        time_steps=WINDOW_SIZE
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model    : STGCN  ({total_params:,} parameters)")
    print(f"Nodes    : {num_nodes}  |  Adj non-zero: {(adj > 0).sum().item()}")
    print(f"Speed    : mean={dataset.mean:.1f} km/h, std={dataset.std:.1f}\n")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    # ReduceLROnPlateau: patience 加大到 8，factor 調小讓 lr 緩速下降
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6
    )

    # ── Training loop ─────────────────────────────────────────
    best_val_mse  = float('inf')
    best_state    = None
    # early stopping patience 放寬到 15，給模型足夠時間突破 plateau
    patience      = 15
    epochs_no_impr= 0
    
    print(f"{'Epoch':>5} | {'Train MSE':>10} | {'Val MSE':>10} | "
          f"{'Val RMSE':>9} | {'Val Acc':>8}")
    print("-" * 65)
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x, adj)
            loss  = criterion(preds, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(batch_x)
        
        train_mse = train_loss / n_train
        val_mse, val_mae, val_rmse, val_acc = evaluate(model, val_loader, criterion, adj, dataset.mean, dataset.std)
        
        scheduler.step(val_mse)
        
        # Save best model and Early Stopping check
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            epochs_no_impr = 0
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_impr += 1
        
        print(f"{epoch:>5} | {train_mse:>10.4f} | {val_mse:>10.4f} | "
              f"{val_rmse:>9.4f} | {val_acc:>7.2f}%")
              
        if epochs_no_impr >= patience:
            print(f"\n[Early Stopping] Triggered at epoch {epoch}. No improvement in Val MSE for {patience} epochs.")
            break
    
    # ── Test evaluation ──────────────────────────────────────
    print("\n" + "=" * 55)
    model.load_state_dict(best_state)
    test_mse, test_mae, test_rmse, test_acc = evaluate(model, test_loader, criterion, adj, dataset.mean, dataset.std)
    
    # Denormalize RMSE back to km/h
    rmse_kmh = test_rmse * dataset.std
    mae_kmh  = test_mae  * dataset.std
    
    print(f"Test MSE  (normalized): {test_mse:.4f}")
    print(f"Test RMSE (normalized): {test_rmse:.4f}")
    print(f"Test MAE  (normalized): {test_mae:.4f}")
    print(f"Test RMSE (km/h)      : {rmse_kmh:.2f}")
    print(f"Test MAE  (km/h)      : {mae_kmh:.2f}")
    print(f"Test Acc (Congestion) : {test_acc:.2f}%")
    
    # ── Save ─────────────────────────────────────────────────
    save_dict = {
        'model_state_dict': best_state,
        'hyperparams': {
            'num_nodes':       num_nodes,
            'in_channels':     1,
            'hidden_channels': HIDDEN_CHANNELS,
            'out_channels':    PREDICT_SIZE,
            'time_steps':      WINDOW_SIZE,
            'window_size':     WINDOW_SIZE,
            'predict_size':    PREDICT_SIZE,
        },
        'normalization': {
            'mean': dataset.mean,
            'std':  dataset.std,
        },
        'link_ids': dataset.link_ids,
        'adj': adj,
    }
    torch.save(save_dict, model_path)
    print(f"\nModel saved → {model_path}")
    print("=" * 55)


if __name__ == '__main__':
    train()
