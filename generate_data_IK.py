import numpy as np
import mujoco
import mujoco.viewer  
import h5py
import os
from tqdm import tqdm

def main(train_percentage=0.8):
    # === 配置路径 ===
    # 假设你的 XML 文件在正确的位置，这里使用相对路径作为示例
    xml_path = r"D:\fast_cdV2\model\universal_robots_ur10e\scene_with_spheres.xml" # 使用上传的相对路径

    if not os.path.exists(xml_path):
        print(f"Error: XML file not found at {xml_path}")
        return

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # === 1. 获取 UR10e 关节信息 ===
    joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                   "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    
    joint_ids = [model.joint(jname).id for jname in joint_names]
    joint_qpos_addrs = [model.jnt_qposadr[jid] for jid in joint_ids]

    qpos_lb, qpos_ub = [], []
    for jid in joint_ids:
        lb, ub = model.jnt_range[jid]
        if lb == 0 and ub == 0:
            lb, ub = -2 * np.pi, 2 * np.pi
        qpos_lb.append(lb)
        qpos_ub.append(ub)
    qpos_lb = np.array(qpos_lb)
    qpos_ub = np.array(qpos_ub)

    mujoco.mj_forward(model, data)
    
    # === 2. 提取障碍物信息 (FIXED: 使用 data.geom_xpos 获取世界坐标) ===
    obstacle_spheres = []
    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE:
            # FIX: 必须使用 data.geom_xpos 获取球体的世界坐标
            pos = data.geom_xpos[i].copy()
            radius = model.geom_size[i][0]
            obstacle_spheres.append((pos, radius))
    
    if not obstacle_spheres:
        print("Error: No spherical obstacles found!")
        return

    print(f"✅ Found {len(obstacle_spheres)} spherical obstacles.")

    ur10e_bodies = {
        "base", "shoulder_link", "upper_arm_link", "forearm_link", 
        "wrist_1_link", "wrist_2_link", "wrist_3_link"
    }
    ur10e_body_ids = {model.body(name).id for name in ur10e_bodies}

    # === 3. 核心计算函数 (已包含 qpos 索引修正) ===
    def set_qpos(q):
        data.qpos[:] = 0
        for idx, addr in enumerate(joint_qpos_addrs):
            data.qpos[addr] = q[idx]
        mujoco.mj_forward(model, data)

    def get_ee_pos():
        site_id = model.site("attachment_site").id
        return data.site_xpos[site_id].copy()

    def compute_sdf(q):
        """
        计算关节角 q 下，机器人所有 Capsule 与环境 Sphere 的最小带符号距离。
        """
        set_qpos(q)
        min_dist = 10.0 # 初始设为一个较大的安全距离

        for gid in range(model.ngeom):
            # 仅处理机器人的 Capsule
            if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_CAPSULE:
                continue
            
            body_id = model.geom_bodyid[gid]
            if body_id not in ur10e_body_ids:
                continue 
            
            # 获取 Capsule 参数 (世界坐标系)
            gpos = data.geom_xpos[gid]
            gmat = data.geom_xmat[gid].reshape(3, 3)
            r_link = model.geom_size[gid][0] # Capsule 半径
            half_len = model.geom_size[gid][1]
            axis = gmat[:, 2] 
            
            p1 = gpos - half_len * axis
            p2 = gpos + half_len * axis
            ab = p2 - p1
            ab_sq = np.dot(ab, ab) + 1e-8

            # 遍历所有障碍球
            for center, r_obs in obstacle_spheres:
                ap = center - p1
                t = np.dot(ap, ab) / ab_sq
                t = np.clip(t, 0.0, 1.0)
                
                closest_point_on_segment = p1 + t * ab
                
                # FIX: 距离 = (线段上最近点到球心的距离) - (胶囊半径 + 球半径)
                dist = np.linalg.norm(center - closest_point_on_segment) - (r_link + r_obs)
                
                if dist < min_dist:
                    min_dist = dist
                    
        return min_dist
    def ik_solve(target_pos, initial_q=None, max_iter=500, err_lim=0.1):
        if initial_q is None:
            q = np.random.uniform(qpos_lb, qpos_ub)
        else:
            q = initial_q.copy()

        site_id = model.site("attachment_site").id
        success = False

        for _ in range(max_iter):
            set_qpos(q)
            current_pos = data.site_xpos[site_id].copy()
            err = target_pos - current_pos
            err_norm = np.linalg.norm(err)
            
            if err_norm < err_lim: # 2cm 误差允许
                success = True
                break

            jac = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jac, None, site_id)
            J = jac[:, joint_ids]

            # 阻尼最小二乘法 (Damped Least Squares) 比转置法更稳定
            lambda_val = 0.01
            dq = J.T @ np.linalg.solve(J @ J.T + lambda_val * np.eye(3), err)
            
            q += dq
            q = np.clip(q, qpos_lb, qpos_ub)
       # print(f"IK 误差: {err_norm}")

        return q, success


    # === 4. 数据生成流程 ===
    TOTAL_SAMPLES = 150000 # 减少到 10000
    
    buffer_collision = []
    buffer_boundary  = []
    buffer_safe      = []

    TARGET_RATIOS = {
        "collision": 0.3,
        "boundary": 0.4,
        "safe": 0.3
    }

    print(f"🚀 Starting generation for {TOTAL_SAMPLES} samples...")
    
    # --- 阶段 1: 随机全空间采样 (构建基础池) ---
    # 用来快速填充 Safe 和 Deep Collision
    print("🎲 Phase 1: Random Sampling...")
    random_batch_size = 20000
    pbar = tqdm(total=random_batch_size)
    for _ in range(random_batch_size):
        q = np.random.uniform(qpos_lb, qpos_ub)
        sdf = compute_sdf(q)
        #print(sdf)
        if sdf < -0.01:
            buffer_collision.append((q, sdf))
        elif sdf > 0.05:
            buffer_safe.append((q, sdf))
        else:
            buffer_boundary.append((q, sdf))
        pbar.update(1)
    pbar.close()

    # --- 阶段 2: 基于 IK 的主动探测 (丰富 Collision/Boundary) ---
    print("🔧 Phase 2: IK Proximity Sampling...")
    ik_attempts = 60000
    pbar = tqdm(total=ik_attempts)
    rng = np.random.default_rng()
    
    for _ in range(ik_attempts):
        # 随机选一个障碍物
        obs_idx = rng.integers(0, len(obstacle_spheres))
        center, radius = obstacle_spheres[obs_idx]
        
        # 在障碍物内部或表面采样
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        # 采样距离：从穿透很深(-r) 到 离开一段距离(+0.1)
        dist_offset = rng.uniform(-radius * 0.8, 0.1) 
        target_pos = center + (radius + dist_offset) * direction
        
        q_init = np.random.uniform(qpos_lb, qpos_ub)
        q_sol, success = ik_solve(target_pos, q_init, err_lim = 0.02)
        
        if success:
            # 给 IK 解加一点高斯噪声，模拟非精确控制，增加数据多样性
            q_sol += np.random.normal(0, 0.05, size=6) 
            q_sol = np.clip(q_sol, qpos_lb, qpos_ub)
            
            sdf = compute_sdf(q_sol)
            if sdf < -0.01:
                buffer_collision.append((q_sol, sdf))
            elif sdf <= 0.05:
                buffer_boundary.append((q_sol, sdf))
            else:
                buffer_safe.append((q_sol, sdf))
        pbar.update(1)
    pbar.close()

    # --- 阶段 3: 边界插值采样 (Boundary Interpolation) ---
    # 这是生成高质量 SDF 数据的关键：在 Collision 和 Safe 之间插值
    print("⚖️ Phase 3: Boundary Interpolation (Bisection Method)...")
    
    target_boundary = int(TOTAL_SAMPLES * TARGET_RATIOS["boundary"])
    pbar = tqdm(total=target_boundary - len(buffer_boundary))
    
    # 为了防止死循环，设置最大尝试次数
    max_attempts = target_boundary * 10
    attempts = 0
    rng = np.random.default_rng()  # seed可选

    while len(buffer_boundary) < target_boundary and attempts < max_attempts:
        attempts += 1
        
        # === 1. 确保池子里有种子数据 ===
        # 如果没有 collision 数据，必须强制生成，不能只靠随机
        if not buffer_collision:
            # 尝试利用 IK 强制找一个碰撞点（借用 Phase 2 的逻辑）
            obs_idx = rng.integers(0, len(obstacle_spheres))
            center, radius = obstacle_spheres[obs_idx]
            # 故意生成在球心附近（深度碰撞）
            target_pos = center + rng.normal(size=3) * 0.01 
            q_sol, success = ik_solve(target_pos, max_iter=100,err_lim = 0.1)
            if success:
                sdf = compute_sdf(q_sol)
                if sdf < 0: buffer_collision.append((q_sol, sdf))
            # 如果还不行，就跳过本次循环继续试
            continue
            
        if not buffer_safe:
            # Safe 很容易找，随机一个就行
            q = np.random.uniform(qpos_lb, qpos_ub)
            sdf = compute_sdf(q)
            if sdf > 0: buffer_safe.append((q, sdf))
            continue

        # === 2. 选取端点 ===
        # 随机取一对 (Collision, Safe)
        idx_c = rng.integers(0, len(buffer_collision))
        idx_s = rng.integers(0, len(buffer_safe))
        
        q_start = buffer_collision[idx_c][0] # SDF < 0
        q_end   = buffer_safe[idx_s][0]      # SDF > 0
        
        # === 3. 二分查找 (核心逻辑) ===
        # 我们在 q_start 和 q_end 之间找 SDF=0
        # 定义 alpha 的范围 [0, 1]
        low = 0.0
        high = 1.0
        
        found_boundary = False
        
        # 迭代 10 次通常足够将精度收敛到非常小
        for _ in range(10):
            mid = (low + high) / 2.0
            q_mid = q_start * (1 - mid) + q_end * mid
            sdf_mid = compute_sdf(q_mid)
            
            if -0.01 <= sdf_mid <= 0.05:
                # 找到了！
                buffer_boundary.append((q_mid, sdf_mid))
                pbar.update(1)
                found_boundary = True
                break
            
            # 更新二分区间
            if sdf_mid < 0:
                # 中点还是碰撞，说明边界在 [mid, high]
                low = mid
                # 顺便把这个新的碰撞点加回去，丰富样本多样性
                if len(buffer_collision) < TOTAL_SAMPLES * TARGET_RATIOS["collision"]:
                    buffer_collision.append((q_mid, sdf_mid))
            else:
                # 中点是安全，说明边界在 [low, mid]
                high = mid
        
        if not found_boundary:
            # 如果二分也没找到满足条件的（极少见，可能是迭代次数不够或区间太窄）
            # 可以选择放弃或把最后的结果硬塞进去（这里选择放弃以保质量）
            pass

    pbar.close()

    # === 5. 组装最终数据集 ===
    print("📦 Assembling Final Dataset...")
    
    # 截断各个列表以符合比例 (如果不够就全部用上)
    n_bound = int(TOTAL_SAMPLES * TARGET_RATIOS["boundary"])
    n_coll  = int(TOTAL_SAMPLES * TARGET_RATIOS["collision"])
    n_safe  = int(TOTAL_SAMPLES * TARGET_RATIOS["safe"])
    
    # 确保不越界
    final_data = []
    final_data.extend(buffer_boundary[:n_bound])
    final_data.extend(buffer_collision[:n_coll])
    final_data.extend(buffer_safe[:n_safe])
    
    # 如果样本不够 TOTAL_SAMPLES，用随机数据补齐
    current_len = len(final_data)
    if current_len < TOTAL_SAMPLES:
        print(f"⚠️ Warning: Generated {current_len} samples, filling remaining {TOTAL_SAMPLES - current_len} with random.")
        for _ in range(TOTAL_SAMPLES - current_len):
            q = np.random.uniform(qpos_lb, qpos_ub)
            final_data.append((q, compute_sdf(q)))

    # 转换为 Numpy 数组并打乱
    rng.shuffle(final_data)
    
    
    train_q = np.array([item[0] for item in final_data[:int(TOTAL_SAMPLES * train_percentage)]], dtype=np.float32)
    train_sdf = np.array([item[1] for item in final_data[:int(TOTAL_SAMPLES * train_percentage)]], dtype=np.float32).reshape(-1, 1)
    
    test_q = np.array([item[0] for item in final_data[int(TOTAL_SAMPLES * train_percentage):]], dtype=np.float32)
    test_sdf = np.array([item[1] for item in final_data[int(TOTAL_SAMPLES * train_percentage):]], dtype=np.float32).reshape(-1, 1)

    # === 6. 保存 ===
    os.makedirs("dataset", exist_ok=True)
    train_path = "dataset/sdf_dataset_train.h5"
    with h5py.File(train_path, "w") as f:
        f.create_dataset("joint_configs", data=train_q)
        f.create_dataset("sdf_values", data=train_sdf)
        f.attrs["num_obstacles"] = len(obstacle_spheres)
        f.attrs["description"] = "Balanced SDF Train dataset: 30% Collision, 40% Boundary, 30% Safe"
    
    test_path = "dataset/sdf_dataset_test.h5"
    with h5py.File(test_path, "w") as f:
        f.create_dataset("joint_configs", data=test_q)
        f.create_dataset("sdf_values", data=test_sdf)
        f.attrs["num_obstacles"] = len(obstacle_spheres)
        f.attrs["description"] = "Balanced SDF Test dataset: 30% Collision, 40% Boundary, 30% Safe"
    # 统计信息
    train_neg_count = np.sum(train_sdf < 0)
    train_boundary_count = np.sum((train_sdf >= -0.02) & (train_sdf <= 0.05))
    train_safe_count = np.sum(train_sdf > 0.05)
    test_neg_count = np.sum(test_sdf < 0)
    test_boundary_count = np.sum((test_sdf >= -0.02) & (test_sdf <= 0.05))
    test_safe_count = np.sum(test_sdf > 0.05)
    print(f"\n🎉 Done! Dataset saved to Train: {train_path}, Test: {test_path}")
    print(f"Total Samples: {len(train_q)+len(test_q)}")
    print(f"  Collision (SDF < 0): Train:{train_neg_count} ({100*train_neg_count/len(train_q):.1f}%),Test:{test_neg_count} ({100*train_neg_count/len(test_q):.1f}%)")
    print(f"  Boundary (-0.02 < SDF < 0.05): Train: {train_boundary_count} ({100*train_boundary_count/len(train_q):.1f}%),Test:{test_boundary_count} ({100*test_boundary_count/len(test_q):.1f}%)")
    print(f"  Safe (SDF > 0.05):Train: {train_safe_count} ({100*train_safe_count/len(train_q):.1f}%), Test: {test_safe_count} ({100*test_safe_count/len(test_q):.1f}%)")
    print(f"  Data validity check: {train_neg_count + train_boundary_count + train_safe_count == len(train_q) and test_safe_count + test_boundary_count + test_neg_count == len(test_q)} ")
    print(f"  Min SDF: {train_sdf.min():.4f}")
    print(f"  Max SDF: {train_sdf.max():.4f}")
    print(f"  Mean SDF: {train_sdf.mean():.4f}")

if __name__ == "__main__":
    main()