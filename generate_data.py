# generate_dataset_with_critical_sampling.py
import numpy as np
import mujoco
import h5py
import os
from tqdm import tqdm

def main():
    xml_path = r"D:\fast_cdV2\model\universal_robots_ur10e\scene_with_spheres.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # === 1. 获取 UR10e 关节 ID 和限位 ===
    joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                   "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    joint_ids = [model.joint(jname).id for jname in joint_names]

    qpos_lb, qpos_ub = [], []
    for jid in joint_ids:
        lb, ub = model.jnt_range[jid]
        if lb == 0 and ub == 0:
            lb, ub = -2 * np.pi, 2 * np.pi
        qpos_lb.append(lb)
        qpos_ub.append(ub)
    qpos_lb = np.array(qpos_lb)
    qpos_ub = np.array(qpos_ub)

    # === 2. 获取碰撞胶囊和球体障碍物 ===
    collision_geom_ids = [i for i in range(model.ngeom) if model.geom_group[i] == 3]
    obstacle_spheres = [
        (model.geom_pos[i].copy(), model.geom_size[i][0])
        for i in range(model.ngeom)
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE
    ]

    def point_to_segment_distance(p, a, b):
        ab = b - a
        ap = p - a
        t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-8)
        t = np.clip(t, 0.0, 1.0)
        return np.linalg.norm(p - (a + t * ab))

    def get_capsule_pose_and_size(gid):
        pos = data.geom_xpos[gid]
        mat = data.geom_xmat[gid].reshape(3, 3)
        radius, half_len = model.geom_size[gid][0], model.geom_size[gid][1]
        axis = mat[:, 2]
        return pos - half_len * axis, pos + half_len * axis, radius

    def compute_sdf(qpos):
        data.qpos[:] = 0
        for idx, jid in enumerate(joint_ids):
            data.qpos[jid] = qpos[idx]
        mujoco.mj_forward(model, data)

        min_dist = np.inf
        for gid in collision_geom_ids:
            p1, p2, r_link = get_capsule_pose_and_size(gid)
            for center, r_obs in obstacle_spheres:
                dist = point_to_segment_distance(center, p1, p2) - r_link - r_obs
                if dist < min_dist:
                    min_dist = dist
        return min_dist

    # === 3. 分阶段采样：先随机，再聚焦临界区 ===
    total_samples = 50
    critical_zone = (-0.1, 0.2)  # 关注 SDF ∈ [-0.1, 0.2]
    target_critical_ratio = 0.6  # 60% 样本来自临界区

    # Step 1: 随机采样初步数据集
    print("🔄 Step 1: Random sampling...")
    rng = np.random.default_rng(42)
    N_random = int(total_samples * 0.3)
    q_candidates = rng.uniform(qpos_lb, qpos_ub, size=(N_random * 5, 6))  # 过采样候选
    sdf_candidates = np.array([compute_sdf(q) for q in tqdm(q_candidates, desc="Random SDF")])

    # 初始数据集
    selected_q = []
    selected_sdf = []

    # 先加入所有碰撞样本（SDF < 0）
    mask_collide = sdf_candidates < 0
    selected_q.extend(q_candidates[mask_collide])
    selected_sdf.extend(sdf_candidates[mask_collide])

    # 再加入部分安全样本
    safe_indices = np.where(~mask_collide)[0]
    n_safe_needed = N_random - len(selected_q)
    if n_safe_needed > 0:
        chosen = rng.choice(safe_indices, size=min(n_safe_needed, len(safe_indices)), replace=False)
        selected_q.extend(q_candidates[chosen])
        selected_sdf.extend(sdf_candidates[chosen])

    # Step 2: 主动采样临界区域
    print("🔍 Step 2: Critical region sampling...")
    n_critical_target = int(total_samples * target_critical_ratio)
    n_already_critical = sum(sdf < critical_zone[1] for sdf in selected_sdf)
    n_more_needed = max(0, n_critical_target - n_already_critical)

    attempts = 0
    while len(selected_q) < total_samples and attempts < 20000:
        q = rng.uniform(qpos_lb, qpos_ub)
        sdf = compute_sdf(q)
        print(sdf)
        attempts += 1

        # 接受条件：在临界区，或当前临界样本不足
        if sdf < critical_zone[1] or len(selected_q) < total_samples * 0.4:
            selected_q.append(q)
            selected_sdf.append(sdf)
        if len(selected_q) >= total_samples:
            break

    # 补足至 total_samples（以防未满）
    if len(selected_q) < total_samples:
        needed = total_samples - len(selected_q)
        extra_q = rng.uniform(qpos_lb, qpos_ub, size=(needed, 6))
        extra_sdf = [compute_sdf(q) for q in extra_q]
        selected_q.extend(extra_q)
        selected_sdf.extend(extra_sdf)

    selected_q = np.array(selected_q[:total_samples]).astype(np.float32)
    selected_sdf = np.array(selected_sdf[:total_samples]).astype(np.float32).reshape(-1, 1)

    # 统计
    critical_count = np.sum((selected_sdf.flatten() >= critical_zone[0]) & (selected_sdf.flatten() <= critical_zone[1]))
    print(f"✅ Final dataset: {len(selected_q)} samples")
    print(f"   Critical zone [{critical_zone[0]}, {critical_zone[1]}]: {critical_count} ({100*critical_count/len(selected_q):.1f}%)")

    # === 4. 保存 ===
    os.makedirs("dataset", exist_ok=True)
    with h5py.File("dataset/sdf_dataset_critical.h5", "w") as f:
        f.create_dataset("joint_configs", data=selected_q)
        f.create_dataset("sdf_values", data=selected_sdf)
        f.attrs["input_dim"] = 6
        f.attrs["output_dim"] = 1
        f.attrs["critical_zone"] = critical_zone


    print("💾 Dataset saved to dataset/sdf_dataset_critical.h5")
    import matplotlib.pyplot as plt
    plt.hist(selected_sdf.flatten(), bins=100, alpha=0.7)
    plt.axvline(critical_zone[0], color='r', linestyle='--')
    plt.axvline(critical_zone[1], color='r', linestyle='--')
    plt.title("SDF Distribution (Final Dataset)")
    plt.savefig("plots/sdf_distribution.png")


if __name__ == "__main__":
    main()