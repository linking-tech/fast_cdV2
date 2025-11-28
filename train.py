import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import h5py
import numpy as np
import os
import torch.jit

# ====================================================================
# A. 模型结构: 融合了 Positional Encoding 和 Skip Connections 的 RNDF 风格 MLP
# ====================================================================
class RNDFMLP(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=256, num_layers=5, skip_layer=2):
        super().__init__()
        
        # 1. Positional Encoding (Sin/Cos Feature Map)
        self.input_layer = nn.Linear(input_dim, hidden_dim) 
        self.pos_encoding_layer = nn.Linear(input_dim * 2, hidden_dim)
        
        # 2. 网络主体 (Body)
        layers = []
        current_dim = hidden_dim
        
        for i in range(num_layers):
            # 处理跳跃连接的拼接输入: 输入维度变为 (当前维度 + 跳跃连接维度)
            if i == skip_layer:
                layers.append(nn.Linear(current_dim + hidden_dim, hidden_dim)) 
            else:
                layers.append(nn.Linear(current_dim, hidden_dim))
            
            layers.append(nn.LeakyReLU(0.1)) 
            current_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)
        
        # 3. 输出层
        self.output_layer = nn.Linear(hidden_dim, 1)

        self.num_layers = num_layers
        self.skip_layer = skip_layer
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0.0)

    def forward(self, q):
        # q: [batch_size, 6] 关节配置
        
        # 1. Positional Encoding: q_in = [sin(q), cos(q)]
        q_pos_encoded = torch.cat([torch.sin(q), torch.cos(q)], dim=-1)
        
        # x_skip: 跳跃连接携带的特征，维度 [batch_size, hidden_dim] (256)
        x_skip = self.pos_encoding_layer(q_pos_encoded)
        x = x_skip # 初始化隐藏层输入 [batch_size, 256]

        # 2. 隐藏层前向传播
        for i, layer in enumerate(self.hidden_layers):
            # 修复点：仅在目标线性层 (偶数索引) 之前进行拼接
            if i == self.skip_layer * 2: 
                x = torch.cat([x, x_skip], dim=-1)
            x = layer(x)

        return self.output_layer(x).squeeze(-1)


# ====================================================================
# B. 损失函数
# ====================================================================
# def collision_type_loss(pred, target):
#     Bceloss = nn.BCEWithLogitsLoss()

def weighted_mse_loss(pred, target, beta=5.0): 
    weight = torch.exp(-beta * torch.abs(target))
    return (weight * (pred - target) ** 2).mean()

# ====================================================================
# C. 评估函数
# ====================================================================
def evaluate_metrics(pred, target, threshold=0.05):
    pred = np.array(pred)
    target = np.array(target)
    # Collision recall: true collision (target<0) detected as risky (pred<threshold)
    true_collision = target < 0
    pred_risky = pred < threshold
    if true_collision.sum() == 0:
        recall = 1.0
    else:
        recall = np.mean(pred_risky[true_collision])
    # False positive rate
    true_safe = target > 0.2
    fp = np.mean(pred_risky[true_safe]) if true_safe.sum() > 0 else 0.0
    # MAE near boundary
    near_boundary = (target >= -0.05) & (target <= 0.05)
    mae_boundary = np.mean(np.abs(pred[near_boundary] - target[near_boundary])) if near_boundary.any() else 0.0
    return recall, fp, mae_boundary

# ====================================================================
# D. 主训练函数
# ====================================================================
def main():
    # 路径 (假设您已生成 sdf_dataset_train.h5)
    train_hdf5_path = "dataset/sdf_dataset_train.h5" 
    
    # 检查文件是否存在并加载
    if not os.path.exists(train_hdf5_path):
        print(f"Error: Training dataset not found at {train_hdf5_path}. Please run data generation first.")
        return

    with h5py.File(train_hdf5_path, "r") as f:
        X = f["joint_configs"][:]
        y = f["sdf_values"][:].flatten()

    print(f"Loaded {len(X)} training samples.")

    # 拆分训练/验证集 (使用整个文件作为训练集，然后内部拆分)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

    # 转换为 Tensor 和 DataLoader
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()),
        batch_size=512, shuffle=True 
    )
    X_val_t = torch.from_numpy(X_val).float()
    y_val_t = torch.from_numpy(y_val).float()
    
    # === 模型和优化器 (使用新的 RNDFMLP) ===
    # 增加模型容量: hidden_dim=256, num_layers=5
    model = RNDFMLP(input_dim=6, hidden_dim=256, num_layers=5, skip_layer=2)
    optimizer = optim.Adam(model.parameters(), lr=5e-4) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_recall = 0.0
    patience = 20 
    no_improve = 0
    max_epochs = 200

    print("🚀 Training started with RNDFMLP...")
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        
        # 学习率调度 
        if epoch == max_epochs // 2:
             for param_group in optimizer.param_groups:
                 param_group['lr'] *= 0.1
                 print(f"--- Learning rate decayed to {param_group['lr']} ---")

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = weighted_mse_loss(pred, y_batch, beta=5.0) 
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t.to(device)).cpu().numpy()
        recall, fp, mae_boundary = evaluate_metrics(val_pred, y_val)

        print(f"Epoch {epoch+1:3d} | Loss: {epoch_loss/len(train_loader):.4f} | "
              f"Recall: {recall:.3f} | FP: {fp:.3f} | MAE@0: {mae_boundary:.4f} | Best Recall: {best_recall:.3f}")

        # Save best model by recall
        if recall > best_recall:
            best_recall = recall
            no_improve = 0
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), "models/best_sdf_model.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping!")
                break

    # === 加载最佳模型并导出 ===
    model.load_state_dict(torch.load("models/best_sdf_model.pth", map_location=device))
    model.eval()

    # TorchScript export
    example_input = torch.randn(1, 6).to(device)
    traced_model = torch.jit.trace(model.cpu(), example_input.cpu()) 
    traced_model.save("models/fast_sdf_model.pt")
    print("✅ Model exported to models/fast_sdf_model.pt")

    # === FIX: Move model back to the GPU for final evaluation ===
    model.to(device) 

    # Final metrics (在整个训练集上报告最终指标)
    final_pred = model(torch.from_numpy(X).float().to(device)).cpu().detach().numpy()
    recall, fp, mae_boundary = evaluate_metrics(final_pred, y)
    print("\n📊 Final Metrics on Full Training Dataset (Best Model):")
    print(f"   Collision Recall (threshold=0.05): {recall:.4f}")
    print(f"   False Positive Rate (safe>0.2):   {fp:.4f}")
    print(f"   MAE near SDF=0:                  {mae_boundary:.4f}")

if __name__ == "__main__":
    main()