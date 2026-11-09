import torch
from PIL.TiffImagePlugin import DATE_TIME


def get_random_avrpsdp_problems(batch_size, node_size, params):
    int_min = params['int_min']
    int_max = params['int_max']
    scaler = params['scaler']
    capacity = params['capacity']
    d_low = params['demand_low']
    d_high = params['demand_high']
    p_low = params['pickup_low']
    p_high = params['pickup_high']
    asym_bias = params['asym_bias']
    customer_size = node_size - 1

    mat = torch.randint(low=int_min, high=int_max, size=(batch_size, node_size, node_size))
    mat[:, torch.arange(node_size), torch.arange(node_size)] = 0


    mat[:, torch.arange(node_size), torch.arange(node_size)] = 0

    while True:
        old_mat = mat.clone()
        mat, _ = (mat[:, :, None, :] + mat[:, None, :, :].transpose(2, 3)).min(dim=3)
        if (mat == old_mat).all():
            break

    mat_T = mat.transpose(1, 2)
    mask = mat > mat_T
    mat = torch.where(mask, mat * (1 + asym_bias), mat)
    mat = mat / scaler

    dp_demand = torch.zeros(batch_size, node_size, 2, dtype=torch.float32)
    dp_demand[:, 1:, 0] = torch.randint(d_low, d_high, size=(batch_size, customer_size)).float()
    dp_demand[:, 1:, 1] = torch.randint(p_low, p_high, size=(batch_size, customer_size)).float()
    dp_demand = dp_demand / capacity

    dp_demand_sym = dp_demand.clone()
    dp_demand_sym = dp_demand_sym[..., [1, 0]]

    d_exp = dp_demand[:, None, :, 0].expand(-1, node_size, -1)
    p_exp = dp_demand[:, None, :, 1].expand(-1, node_size, -1)
    problems = torch.stack([mat, d_exp, p_exp], dim=-1)

    d_exp_sym = dp_demand_sym[:, None, :, 0].expand(-1, node_size, -1)
    p_exp_sym = dp_demand_sym[:, None, :, 1].expand(-1, node_size, -1)
    problems_sym = torch.stack([mat, d_exp_sym, p_exp_sym], dim=-1)

    total_delivery = dp_demand[:, :, 0].sum(dim=1)
    total_delivery_sym = dp_demand_sym[:, :, 0].sum(dim=1)

    total_pickup = dp_demand[:, :, 1].sum(dim=1)
    total_pickup_sym = dp_demand_sym[:, :, 1].sum(dim=1)

    pd_diff = dp_demand[:, 1:, 1] - dp_demand[:, 1:, 0]
    global_pd_max = pd_diff.max(dim=1).values
    safe_ratio = 1.0 - global_pd_max

    pd_diff_sym = dp_demand_sym[:, 1:, 1] - dp_demand_sym[:, 1:, 0]
    global_pd_max_sym = pd_diff_sym.max(dim=1).values
    safe_ratio_sym = 1.0 - global_pd_max_sym


    return (
        problems,
        problems_sym,
        dp_demand,
        dp_demand_sym,
        total_delivery,
        total_delivery_sym,
        total_pickup,
        total_pickup_sym,
        safe_ratio,
        safe_ratio_sym
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    params = {
        "int_min": 0,
        "int_max": 1000*1000,
        "scaler": 1000*1000,
        "capacity": 30,
        "demand_low": 1,
        "demand_high": 10,
        "pickup_low": 1,
        "pickup_high": 10,
        "asym_bias": 0.2
    }
    batch = 2
    n_node = 20
    out = get_random_avrpsdp_problems(batch, n_node, params)
    problems, sym_problems, dp_demand, sym_dp_demand, total_delivery, sym_total_delivery, total_pickup, sym_total_pickup, safe_ratio, sym_safe_ratio = out

    print("="*60)
    print(f"problems shape: {problems.shape} [B,N,N,3]")
    print(f"sym_problems shape: {sym_problems.shape}")
    print("="*60)

    dist_mat = problems[0, :, :, 0]
    print("对角线是否全部为0:", torch.allclose(torch.diag(dist_mat), torch.zeros(n_node)))

    i, j = 1, 2
    d_ij = dist_mat[i, j]
    d_ji = dist_mat[j, i]
    print(f"\n抽查 D[{i},{j}]={d_ij:.4f}, D[{j},{i}]={d_ji:.4f}")

    diff_abs = torch.abs(dist_mat - dist_mat.T)
    mask_off_diag = ~torch.eye(n_node, dtype=torch.bool)
    diff_off = diff_abs[mask_off_diag]
    print("\n===== 全局双向距离不对称统计 =====")
    print(f"双向差值均值: {diff_off.mean().item():.6f}")
    print(f"双向差值最大值: {diff_off.max().item():.6f}")
    print(f"双向差值总和: {diff_off.sum().item():.6f}")
    asym_edge_count = torch.sum(diff_off > 1e-6).item()
    total_edge = diff_off.numel()
    print(f"不对称边数量：{asym_edge_count} / {total_edge}")

    sym_dist = sym_problems[0, :, :, 0]
    print("\nsym_problems距离是否与原D一致:", torch.allclose(dist_mat, sym_dist))
    print("原需求样本0 (d, p):", dp_demand[0])
    print("对偶需求样本0 (p, d):", sym_dp_demand[0])
    print("总送货需求(样本0):", total_delivery[0].item())
    print("总取货需求(样本0):", total_pickup[0].item())
    print("safe_ratio:", safe_ratio[0].item())
    print("sym_safe_ratio:", sym_safe_ratio[0].item())