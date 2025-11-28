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
#    - 修改点：输出维度变为 2 (SDF, Classification Logit)
# ====================================================================
class RNDFMLP(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=256, num_layers=5, skip_layer=2):
        super().__init__()
        
        # 1. Positional Encoding (Sin/Cos Feature Map)
        self.input_layer = nn.Linear(input_dim, hidden_dim) 
        self.pos_encoding_layer = nn.Linear(input_dim * 2, hidden_dim)
        
        # 2. 网络主体 (Body) - 保持不变
        layers = []
        current_dim = hidden_dim
        
        for i in range(num_layers):
            if i == skip_layer:
                layers.append(nn.Linear(current_dim + hidden_dim, hidden_dim)) 
            else:
                layers.append(nn.Linear(current_dim, hidden_dim))
            
            layers.append(nn.LeakyReLU(0.1)) 
            current_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)
        
        # 3. 输出层 - 拆分成两个独立的头
        self.sdf_output_layer = nn.Linear(hidden_dim, 1)      # SDF 值 (回归)
        self.class_output_layer = nn.Linear(hidden_dim, 1)    # 分类 Logit (二分类)

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
            if i == self.skip_layer * 2: 
                x = torch.cat([x, x_skip], dim=-1)
            x = layer(x)
        
        # 3. 双头输出
        # pred_sdf: [batch_size] SDF 回归预测
        pred_sdf = self.sdf_output_layer(x).squeeze(-1)
        # pred_logit: [batch_size] 碰撞分类 Logit 预测
        pred_logit = self.class_output_layer(x).squeeze(-1)

        return pred_sdf, pred_logit # 返回两个值


# ====================================================================
# B. 损失函数
#    - 修改点：新增 BCEWithLogitsLoss，并创建混合损失函数
# ====================================================================
def weighted_mse_loss(pred, target, beta=5.0): 
    # 回归损失，更关注接近 SDF=0 的点
    weight = torch.exp(-beta * torch.abs(target))
    return (weight * (pred - target) ** 2).mean()

# 使用 PyTorch 内置的 BCEWithLogitsLoss，它更稳定
bce_loss_fn = nn.BCEWithLogitsLoss()

def mixed_loss(pred_sdf, pred_logit, target_sdf, target_class, lambda_bce=1.0, beta_mse=5.0):
    # 1. 回归损失 (Weighted MSE)
    loss_sdf = weighted_mse_loss(pred_sdf, target_sdf, beta=beta_mse)
    
    # 2. 分类损失 (BCE)
    # 目标类别 (0.0=安全/边界, 1.0=碰撞)
    loss_bce = bce_loss_fn(pred_logit, target_class)
    
    # 3. 混合损失
    total_loss = loss_sdf + lambda_bce * loss_bce
    return total_loss, loss_sdf, loss_bce


# ====================================================================
# C. 评估函数 (保持不变，只使用 SDF 预测进行评估)
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
#    - 修改点：准备分类标签，使用混合损失
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
        y_sdf = f["sdf_values"][:].flatten()

    print(f"Loaded {len(X)} training samples.")

    # **新增：根据 SDF 值生成二分类标签**
    # 碰撞 (SDF <= 0) = 1.0, 安全/边界 (SDF > 0) = 0.0
    y_class = (y_sdf <= 0).astype(np.float32)

    # 拆分训练/验证集 (现在拆分 X, y_sdf, y_class)
    X_train, X_val, y_sdf_train, y_sdf_val, y_class_train, y_class_val = train_test_split(
        X, y_sdf, y_class, test_size=0.1, random_state=42
    )

    # 转换为 Tensor 和 DataLoader
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train).float(), 
            torch.from_numpy(y_sdf_train).float(),
            torch.from_numpy(y_class_train).float() # 新增分类标签
        ),
        batch_size=512, shuffle=True 
    )
    X_val_t = torch.from_numpy(X_val).float()
    y_sdf_val_t = torch.from_numpy(y_sdf_val).float()
    
    # === 模型和优化器 ===
    model = RNDFMLP(input_dim=6, hidden_dim=256, num_layers=5, skip_layer=2)
    optimizer = optim.Adam(model.parameters(), lr=5e-4) 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 损失权重参数
    LAMBDA_BCE = 1.0 # 交叉熵损失权重
    BETA_MSE = 5.0   # MSE 损失的 beta 参数

    best_recall = 0.0
    patience = 20 
    no_improve = 0
    max_epochs = 200

    print(f"🚀 Training started with RNDFMLP (Mixed Loss: SDF + {LAMBDA_BCE}*BCE)...")
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_loss_sdf = 0.0
        epoch_loss_bce = 0.0
        
        # 学习率调度 
        if epoch == max_epochs // 2:
             for param_group in optimizer.param_groups:
                 param_group['lr'] *= 0.1
                 print(f"--- Learning rate decayed to {param_group['lr']} ---")

        # 迭代器现在输出 x_batch, y_sdf_batch, y_class_batch
        for x_batch, y_sdf_batch, y_class_batch in train_loader:
            x_batch = x_batch.to(device)
            y_sdf_batch = y_sdf_batch.to(device)
            y_class_batch = y_class_batch.to(device) # 新增

            optimizer.zero_grad()
            
            # 模型现在返回两个输出
            pred_sdf, pred_logit = model(x_batch)
            
            # 计算混合损失
            loss, loss_sdf, loss_bce = mixed_loss(
                pred_sdf, pred_logit, y_sdf_batch, y_class_batch, 
                lambda_bce=LAMBDA_BCE, beta_mse=BETA_MSE
            )
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_loss_sdf += loss_sdf.item()
            epoch_loss_bce += loss_bce.item()

        # Validation
        model.eval()
        with torch.no_grad():
            # 验证时只使用 SDF 预测 (第一个输出)
            val_pred_sdf, _ = model(X_val_t.to(device))
            val_pred = val_pred_sdf.cpu().numpy()

        recall, fp, mae_boundary = evaluate_metrics(val_pred, y_sdf_val)

        print(f"Epoch {epoch+1:3d} | Total Loss: {epoch_loss/len(train_loader):.4f} "
              f"(SDF: {epoch_loss_sdf/len(train_loader):.4f}, BCE: {epoch_loss_bce/len(train_loader):.4f}) | "
              f"Recall: {recall:.3f} | FP: {fp:.3f} | MAE@0: {mae_boundary:.4f} | Best Recall: {best_recall:.3f}")

        # Save best model by recall
        if recall > best_recall:
            best_recall = recall
            no_improve = 0
            os.makedirs("models", exist_ok=True)
            # 保存整个模型，因为结构发生了变化
            torch.save(model.state_dict(), "models/best_sdf_class_model.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping!")
                break

    # === 加载最佳模型并导出 ===
    model.load_state_dict(torch.load("models/best_sdf_class_model.pth", map_location=device))
    model.eval()

    # TorchScript export (只导出 SDF 预测部分)
    class SDFPredictor(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, q):
            # 只需要 SDF 预测
            pred_sdf, _ = self.model(q)
            return pred_sdf

    sdf_predictor = SDFPredictor(model.cpu()) # 先将模型移到 CPU 进行 tracing
    example_input = torch.randn(1, 6)
    traced_model = torch.jit.trace(sdf_predictor, example_input) 
    traced_model.save("models/fast_sdf_model_mixed.pt")
    print("✅ SDF Predictor Model exported to models/fast_sdf_model_mixed.pt")

    # === Final evaluation ===
    model.to(device)
    # 在整个数据集上创建最终的 Tensor
    X_full_t = torch.from_numpy(X).float().to(device)
    y_full_sdf = y_sdf
    
    # Final metrics 
    with torch.no_grad():
        final_pred_sdf, _ = model(X_full_t)
        final_pred = final_pred_sdf.cpu().detach().numpy()
        
    recall, fp, mae_boundary = evaluate_metrics(final_pred, y_full_sdf)
    print("\n📊 Final Metrics on Full Training Dataset (Best Model):")
    print(f"   Collision Recall (threshold=0.05): {recall:.4f}")
    print(f"   False Positive Rate (safe>0.2):   {fp:.4f}")
    print(f"   MAE near SDF=0:                  {mae_boundary:.4f}")

if __name__ == "__main__":
    main()