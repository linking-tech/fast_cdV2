import torch
import numpy as np
import mujoco
import h5py
import os
import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import seaborn as sns 
from scipy.stats import gaussian_kde # 用于 KDE 图
# ====================================================================
# I. MuJoCo 环境初始化与 SDF 计算 (用于 Ground Truth 基准测试)
# 必须复制 generate_data_IK.py 中的核心逻辑来计算 SDF
# ====================================================================

# 全局变量占位符 (将在初始化函数中填充)
MUJOCO_ENV = {}

def initialize_mujoco_environment():
    """初始化 MuJoCo 模型、数据和 SDF 计算所需的参数"""
    global MUJOCO_ENV
    
    # 路径使用您在 generate_data_IK.py 中使用的路径
    xml_path = r"D:\fast_cdV2\model\universal_robots_ur10e\scene_with_spheres.xml" 
    if not os.path.exists(xml_path):
        print(f"Error: XML file not found at {xml_path}. Cannot perform MuJoCo benchmark.")
        return False

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # 1. 关节信息
    joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                   "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    joint_ids = [model.joint(jname).id for jname in joint_names]
    joint_qpos_addrs = [model.jnt_qposadr[jid] for jid in joint_ids]

    mujoco.mj_forward(model, data)
    
    # 2. 障碍物信息 (FIXED: 使用 data.geom_xpos 获取世界坐标)
    obstacle_spheres = []
    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE:
            pos = data.geom_xpos[i].copy()
            radius = model.geom_size[i][0]
            obstacle_spheres.append((pos, radius))

    ur10e_bodies = {"base", "shoulder_link", "upper_arm_link", "forearm_link", 
                    "wrist_1_link", "wrist_2_link", "wrist_3_link"}
    ur10e_body_ids = {model.body(name).id for name in ur10e_bodies}
    
    if not obstacle_spheres:
        print("Warning: No spherical obstacles found for benchmark!")

    MUJOCO_ENV = {
        "model": model,
        "data": data,
        "joint_qpos_addrs": joint_qpos_addrs,
        "obstacle_spheres": obstacle_spheres,
        "ur10e_body_ids": ur10e_body_ids
    }
    return True

def set_qpos_mujoco(q):
    """设置关节位置并更新运动学"""
    model = MUJOCO_ENV["model"]
    data = MUJOCO_ENV["data"]
    joint_qpos_addrs = MUJOCO_ENV["joint_qpos_addrs"]
    
    data.qpos[:] = 0
    for idx, addr in enumerate(joint_qpos_addrs):
        data.qpos[addr] = q[idx]
    mujoco.mj_forward(model, data)

def compute_sdf_mujoco(q):
    """计算 Ground Truth SDF"""
    set_qpos_mujoco(q)
    
    model = MUJOCO_ENV["model"]
    data = MUJOCO_ENV["data"]
    obstacle_spheres = MUJOCO_ENV["obstacle_spheres"]
    ur10e_body_ids = MUJOCO_ENV["ur10e_body_ids"]
    
    min_dist = 10.0
    
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_CAPSULE:
            continue
        
        body_id = model.geom_bodyid[gid]
        if body_id not in ur10e_body_ids:
            continue 
        
        gpos = data.geom_xpos[gid]
        gmat = data.geom_xmat[gid].reshape(3, 3)
        r_link = model.geom_size[gid][0]
        half_len = model.geom_size[gid][1]
        axis = gmat[:, 2] 
        
        p1 = gpos - half_len * axis
        p2 = gpos + half_len * axis
        ab = p2 - p1
        ab_sq = np.dot(ab, ab) + 1e-8

        for center, r_obs in obstacle_spheres:
            ap = center - p1
            t = np.dot(ap, ab) / ab_sq
            t = np.clip(t, 0.0, 1.0)
            
            closest_point_on_segment = p1 + t * ab
            
            dist = np.linalg.norm(center - closest_point_on_segment) - (r_link + r_obs)
            
            if dist < min_dist:
                min_dist = dist
                
    return min_dist


# ====================================================================
# II. 数据加载与模型推理
# ====================================================================

def load_dataset(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        X = f["joint_configs"][:]
        y_true = f["sdf_values"][:].flatten()
    return X, y_true

def load_model(model_path):
    # TorchScript 模型可以直接加载到 CPU 或 GPU，不需要定义类结构
    model = torch.jit.load(model_path)
    model.eval()
    return model

def predict_sdf(model, X):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    X_t = torch.from_numpy(X).float().to(device)
    with torch.no_grad():
        # 模型的输出是 (N,) 维度，而不是 (N, 1)
        pred = model(X_t).cpu().numpy().flatten()
    return pred

# ====================================================================
# III. 指标计算与绘图 (与原 infer.py 逻辑一致)
# ====================================================================

def compute_metrics(pred, true, threshold=0.05):
    # Collision recall: true collision (true < 0) → pred < threshold
    true_collision = true < 0
    pred_risky = pred < threshold
    recall = np.mean(pred_risky[true_collision]) if true_collision.sum() > 0 else 1.0

    # False positive rate: true safe (>0.2) → pred < threshold
    true_safe = true > 0.2
    pred_risky = pred < threshold
    fp_rate = np.mean(pred_risky[true_safe]) if true_safe.sum() > 0 else 0.0

    # MAE near boundary
    near_boundary = (true >= -0.05) & (true <= 0.05)
    mae_boundary = np.mean(np.abs(pred[near_boundary] - true[near_boundary])) if near_boundary.any() else 0.0

    return {
        "recall": recall,
        "fp_rate": fp_rate,
        "mae_boundary": mae_boundary,
        "mean_error": np.mean(np.abs(pred - true)),
        "max_error": np.max(np.abs(pred - true))
    }

def plot_roc_curve(pred, true, save_path="plots/roc_curve.png"):
    y_true_binary = (true < 0).astype(int)
    y_pred_proba = -pred  # 负 SDF 意味着更高的风险
    fpr, tpr, _ = roc_curve(y_true_binary, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Collision Detection')
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close() # 关闭图形以节省内存

def plot_scatter(pred, true, save_path="plots/scatter_plot.png"):
    plt.figure(figsize=(8, 8))
    plt.scatter(true, pred, alpha=0.6, s=10, c='blue')
    
    # 确保 red line 覆盖数据范围
    min_val = min(true.min(), pred.min())
    max_val = max(true.max(), pred.max())
    if np.isnan(min_val) or np.isnan(max_val): 
         min_val = -0.6
         max_val = 0.6
    
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel("Ground Truth SDF")
    plt.ylabel("Predicted SDF")
    plt.title("Scatter Plot: Predicted vs Ground Truth")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
def plot_sdf_and_gradient_slice(model, X_test, y_true, joint_idx=1, n_points=500, save_path="plots/sdf_gradient_slice.png"):
    """
    绘制 SDF 值和其关于单个关节的梯度切片图。
    灵感来源：Li et al. - 2024 (RDF) Fig. 3(b)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 1. 确定切片基准 (使用一个随机的配置作为切片平面)
    base_idx = np.random.randint(len(X_test))
    base_q = X_test[base_idx].copy()
    
    # 2. 定义扫描范围
    q_scan = base_q.copy()
    # 假设关节限制在 [-pi, pi] 附近, 这里我们扫描一个合理的范围
    scan_range = np.linspace(-np.pi, np.pi, n_points)
    
    X_slice = np.tile(base_q, (n_points, 1)).astype(np.float32)
    X_slice[:, joint_idx] = scan_range

    # 3. 计算预测 SDF 和梯度
    X_t = torch.from_numpy(X_slice).float().to(device)
    X_t.requires_grad_(True)
    
    with torch.no_grad():
        y_pred_slice = model(X_t).cpu().numpy().flatten()
    
    # 重新启用梯度以计算导数
    y_pred_t_grad = model(X_t)
    
    # 我们只关心 SDF 的平均值，而不是每个样本的 min(SDF)
    # 这里的模型输出是 min(d_k(q,p_i))，我们假设模型学习了一个复合距离函数
    # 且我们希望知道这个函数相对于 q_j 的变化
    
    # 为了简化，我们计算每个输出相对于 q_scan 的梯度，然后取平均或最小值梯度
    gradients_list = []
    for i in range(n_points):
        # 计算当前点的所有输出对所有输入的梯度
        # 由于 traced_model 只有一个输出，我们直接计算该输出对输入的梯度
        grads_all = torch.autograd.grad(y_pred_t_grad[i], X_t, retain_graph=True, allow_unused=True)[0]
        # 提取当前关节的梯度值
        gradients_list.append(grads_all[i, joint_idx].item())
    
    gradients = np.array(gradients_list)
    
    # 4. 可视化
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # --- SDF 曲线 ---
    axes[0].plot(scan_range, y_pred_slice, label=f'Predicted SDF (q{joint_idx+1} Slice)', color='crimson', linewidth=2)
    
    # 在原始数据中找到最接近这个切片的真实值进行绘制 (可选，但很有挑战性)
    # 寻找真实值是一个挑战，因为我们切片在一个特定高维配置周围，
    # 而真实值数据点在整个高维空间中是稀疏的。因此，我们只绘制预测值。
    
    axes[0].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    axes[0].set_title(f'SDF Value vs. Joint Angle $q_{{{joint_idx+1}}}$ (Slice at Fixed $q$)', fontsize=14)
    axes[0].set_ylabel('Predicted SDF Value', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # --- 梯度曲线 ---
    axes[1].plot(scan_range, gradients, label=f'Predicted Gradient $\\partial SDF/\\partial q_{{{joint_idx+1}}}$', color='royalblue', linewidth=2)
    axes[1].axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    axes[1].set_title(f'SDF Gradient vs. Joint Angle $q_{{{joint_idx+1}}}$ (Slice)', fontsize=14)
    axes[1].set_xlabel(f'Joint Angle $q_{{{joint_idx+1}}}$ (radians)', fontsize=12)
    axes[1].set_ylabel('SDF Gradient Value', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
def plot_large_error_diff(pred, true, error_threshold=0.01, max_points=1000):
    """
    Filters and plots true vs predicted SDF values only for samples where the absolute error > threshold.
    Simulates the 'Error Curve' visualization requested by the user.
    """
    save_path=f"plots/large_error_diff_curve_{error_threshold}.png"
    absolute_error = np.abs(pred - true)
    
    # 1. Filter indices where error is large
    large_error_indices = np.where(absolute_error > error_threshold)[0]
    
    if len(large_error_indices) == 0:
        print(f"Warning: No samples found with absolute error > {error_threshold}. Skipping this plot.")
        return

    # 2. Sample a subset of these indices if there are too many points
    if len(large_error_indices) > max_points:
        # Randomly sample to avoid overly dense plot
        sample_indices = np.random.choice(large_error_indices, size=max_points, replace=False)
        subtitle = f" (Showing random {max_points} of {len(large_error_indices)} points where Error > {error_threshold})"
    else:
        sample_indices = large_error_indices
        subtitle = f" (Showing all {len(large_error_indices)} points where Error > {error_threshold})"

    # 3. Extract sampled data
    true_sampled = true[sample_indices]
    pred_sampled = pred[sample_indices]
    
    # Sort by Ground Truth value for a cleaner, monotonic curve-like visualization (helps simulate the look of the user's image)
    sort_indices = np.argsort(true_sampled)
    true_sampled = true_sampled[sort_indices]
    pred_sampled = pred_sampled[sort_indices]
    
    # 4. Plotting
    plt.figure(figsize=(12, 6))
    
    # Plotting True values as a reference line (sorted, should look like a curve)
    plt.plot(true_sampled, label='Ground Truth SDF', color='blue', linestyle='-', linewidth=1.5, alpha=0.7)
    
    # Plotting Predicted values (dotted or distinct line/points)
    plt.plot(pred_sampled, label='Predicted SDF', color='red', linestyle='--', linewidth=1.5, alpha=0.9)
    
    # Also plot the difference/error bars or shaded area if needed, but the two lines are usually enough for visual comparison
    
    plt.title(f'True vs. Predicted SDF for High-Error Samples{subtitle}', fontsize=14)
    plt.xlabel(f'Sample Index (Sorted by True SDF Value)', fontsize=12)
    plt.ylabel('SDF Value', fontsize=12)
    plt.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.7)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_error_and_value_distribution(pred, true, save_path_base="plots/"):
    """
    绘制预测值与真实值的核密度估计(KDE)和绝对误差直方图。
    灵感来源：Zhu et al. - 2024 (SDF-SC) Fig. 3(a)
    """
    
    # 1. KDE Plot: Predicted vs Ground Truth SDF
    plt.figure(figsize=(10, 6))
    
    # 使用 seaborn/matplotlib 绘制 KDE
    sns.kdeplot(true, label='Ground Truth SDF', fill=True, alpha=.5, linewidth=1.5, color='green')
    sns.kdeplot(pred, label='Predicted SDF', fill=True, alpha=.5, linewidth=1.5, color='blue')
    
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Collision Boundary')
    plt.title('Distribution of SDF Values (Kernel Density Estimation)', fontsize=14)
    plt.xlabel('SDF Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path_base}sdf_kde_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Absolute Error Histogram
    absolute_error = np.abs(pred - true)
    
    plt.figure(figsize=(10, 6))
    
    # 绘制直方图，限制 X 轴以排除极端离群值
    max_err_plot = np.percentile(absolute_error, 99.5) 
    sns.histplot(absolute_error[absolute_error <= max_err_plot], bins=50, kde=True, 
                 color='orange', edgecolor='black', alpha=0.7)
    
    mean_err = np.mean(absolute_error)
    plt.axvline(mean_err, color='red', linestyle='-', linewidth=2, label=f'Mean Error: {mean_err:.4f}')
    
    plt.title('Distribution of Absolute Prediction Error', fontsize=14)
    plt.xlabel(f'Absolute Error |Predicted - True| (Capped at {max_err_plot:.4f})', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_path_base}abs_error_histogram.png", dpi=150, bbox_inches='tight')
    plt.close()

# ====================================================================
# IV. 基准测试函数 (MLP vs. MuJoCo)
# ====================================================================

def benchmark_inference_speed(model, X, num_runs=1000):
    """测量 MLP 模型推理速度 (单样本)"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Benchmarking MLP inference on device: {device}")

    model.to(device)
    
    times = []
    # 预热 (Warmup)
    dummy_input = torch.from_numpy(X[:1]).float().to(device)
    for _ in range(100):
         _ = model(dummy_input)
    if device == "cuda":
        torch.cuda.synchronize()

    # 正式计时
    for _ in range(num_runs):
        X_t = torch.from_numpy(X[np.random.randint(len(X)):np.random.randint(len(X))+1]).float().to(device)
        
        start_event = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
        end_event = torch.cuda.Event(enable_timing=True) if device == "cuda" else None

        if start_event is not None:
            start_event.record()
            with torch.no_grad():
                _ = model(X_t)
            end_event.record()
            torch.cuda.synchronize() 
            time_ms = start_event.elapsed_time(end_event)
            times.append(time_ms)
        else:
            start = time.time()
            with torch.no_grad():
                _ = model(X_t)
            end = time.time()
            times.append((end - start) * 1000)

    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"✅ MLP Inference Speed: {avg_time:.4f} ms ± {std_time:.4f} ms per sample")
    return avg_time

def benchmark_mujoco_sdf(X_test_subset, num_runs=100):
    """测量 MuJoCo SDF 计算速度 (单样本, CPU)"""
    if not MUJOCO_ENV:
        print("❌ MuJoCo environment not initialized. Skipping MuJoCo benchmark.")
        return 9999.0 # 返回一个大值

    print(f"🚀 Benchmarking MuJoCo Ground Truth SDF calculation ({num_runs} samples)...")
    
    mujoco_times = []
    
    # 预热 (Warmup)
    for _ in range(10):
        _ = compute_sdf_mujoco(X_test_subset[0])

    # 正式计时
    for i in tqdm(range(num_runs), desc="MuJoCo Timing"):
        q = X_test_subset[i % len(X_test_subset)]
        start = time.time()
        _ = compute_sdf_mujoco(q)
        end = time.time()
        mujoco_times.append((end - start) * 1000) # 转换为毫秒

    avg_time = np.mean(mujoco_times)
    std_time = np.std(mujoco_times)
    print(f"✅ MuJoCo SDF Time (CPU): {avg_time:.4f} ms ± {std_time:.4f} ms per sample")
    return avg_time
def plot_trajectory_error_curve(y_pred, y_true, threshold=0.01, num_steps=200):
    """
    Plots the absolute prediction error over a mock sequential trajectory,
    highlighting points that exceed the specified error threshold.
    """
    
    # 1. Simulate a trajectory (use the first N points as time steps)
    N = len(y_pred)
    num_steps = min(N, num_steps)
    
    y_pred_traj = y_pred[:num_steps]
    y_true_traj = y_true[:num_steps]
    
    absolute_error = np.abs(y_pred_traj - y_true_traj)
    time_steps = np.arange(num_steps)
    
    # 2. Identify high-error points
    high_error_mask = absolute_error > threshold
    
    # 3. Visualization
    plt.figure(figsize=(10, 6))
    
    # Plot the full error curve (as background/context)
    plt.plot(time_steps, absolute_error, 
             label='Absolute Error $|y_{pred} - y_{true}|$', 
             color='gray', linewidth=1.5, alpha=0.7)
    
    # Plot the high error points (as scatter points, mirroring the user's image style)
    plt.scatter(time_steps[high_error_mask], absolute_error[high_error_mask], 
                label=f'Error > {threshold:.2f} (Count: {high_error_mask.sum()})', 
                color='red', s=20, zorder=5)
    
    # Plot the threshold line
    plt.axhline(threshold, color='crimson', linestyle='--', linewidth=1.5, 
                label=f'Error Threshold: {threshold:.2f}', zorder=3)
    
    plt.title(f'Absolute Prediction Error along a Mock Trajectory (Highlighting Error > {threshold:.2f})', fontsize=14)
    plt.xlabel('Time Step (Mock Trajectory Index)', fontsize=12)
    plt.ylabel('Absolute Error (SDF Value)', fontsize=12)
    plt.ylim(0, np.percentile(absolute_error, 99.8) * 1.1) # Set Y limit dynamically
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = "plots/error_over_trajectory_curve.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path

# ====================================================================
# V. 主函数 (修改点：更新模型路径)
# ====================================================================

def main():
    # 路径
    test_hdf5_path = "dataset/sdf_dataset_test.h5"
    # *** 修改点：更新为新的模型文件名 ***
    model_path = "models/fast_sdf_model_mixed.pt" 

    # 检查文件是否存在
    if not os.path.exists(test_hdf5_path):
        print(f"Error: Test dataset not found at {test_hdf5_path}. Please check data generation output.")
        return
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}. Please run train.py first.")
        # 补充：如果新模型不存在，但旧模型存在，可以尝试使用旧模型进行推理，以便继续绘图和评估。
        old_model_path = "models/fast_sdf_model.pt"
        if os.path.exists(old_model_path):
            print(f"Warning: New model {model_path} not found. Using old model {old_model_path} instead.")
            model_path = old_model_path
        else:
             return

    # 1. 加载数据和模型
    print("🔍 Loading dataset and model...")
    X_test, y_true = load_dataset(test_hdf5_path)
    print(f"Dataset loaded: {len(X_test)} test samples")

    model = load_model(model_path)

    # 2. 推理
    print("⚡ Running MLP inference on test data...")
    y_pred = predict_sdf(model, X_test)

    # 3. 性能指标
    metrics = compute_metrics(y_pred, y_true)
    print("\n📊 Model Performance Metrics:")
    for k, v in metrics.items():
        print(f"   {k}: {v:.4f}")

    # 4. 可视化
    os.makedirs("plots", exist_ok=True)
    plot_roc_curve(y_pred, y_true)
    plot_scatter(y_pred, y_true)
    
    # --- 新增的可视化 ---
    # 4a. SDF 值和梯度切片图 (选择第一个关节 q1)
    if X_test.shape[1] > 0:
        plot_sdf_and_gradient_slice(model, X_test, y_true, joint_idx=0)
        print("✅ SDF and Gradient Slice Plot saved to 'plots/' directory!")
    
    # 4b. 分布图
    plot_error_and_value_distribution(y_pred, y_true)
   
    print("✅ Distribution plots saved to 'plots/' directory!")
    # --- 结束新增 ---
    
    print("\n✅ ROC Curve and Scatter Plot saved to 'plots/' directory!")

    plot_large_error_diff(y_pred, y_true,error_threshold=0.01,max_points = 100000)
    plot_large_error_diff(y_pred, y_true,error_threshold=0.05,max_points = 100000)
    plot_large_error_diff(y_pred, y_true,error_threshold=0.1,max_points = 100000)

    print("\n✅ Large error differences Plot saved to 'plots/' directory!")
    plot_trajectory_error_curve(y_pred, y_true,threshold=0.01)
    print("\n✅ Large error CURVE Plot saved to 'plots/' directory!")
    # 5. 速度基准测试 (MuJoCo vs. MLP)
    
    # MLP 速度
    num_benchmark_samples = min(5000, len(X_test)) # 限制样本数以加速测试
    mlp_time_ms = benchmark_inference_speed(model, X_test, num_runs=num_benchmark_samples)

    # MuJoCo 速度
    if initialize_mujoco_environment():
        mujoco_time_ms = benchmark_mujoco_sdf(X_test[:num_benchmark_samples], num_runs=num_benchmark_samples)
        
        print("\n=======================================================")
        print("           ⏱️  ACCELERATION COMPARISON")
        print("=======================================================")
        print(f"MuJoCo SDF Time (CPU): {mujoco_time_ms:.4f} ms/sample")
        print(f"MLP Inference Time ({'GPU' if torch.cuda.is_available() else 'CPU'}): {mlp_time_ms:.4f} ms/sample")
        
        if mlp_time_ms > 0:
            speedup = mujoco_time_ms / mlp_time_ms
            print(f"🚀 加速比 (MuJoCo / MLP): {speedup:.2f}x")
        print("=======================================================")
        
    else:
        print("\n❌ 无法进行 MuJoCo 基准测试，请检查 XML 文件路径。")


if __name__ == "__main__":
    # 配置 matplotlib 使用非交互式后端以确保在无显示环境也能运行
    plt.switch_backend('Agg')
    main()