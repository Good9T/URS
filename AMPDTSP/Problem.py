import torch
from PIL.TiffImagePlugin import DATE_TIME


def get_random_ampdtsp_problems(batch_size, node_size, params):
    int_min = params['int_min']
    int_max = params['int_max']
    scaler = params['scaler']
    capacity = params['capacity']
    demand_low = params['demand_low']
    demand_high = params['demand_high']
    asym_bias = params['asym_bias']
    customer_size = node_size - 1
    pickup_size = customer_size // 2

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

    depot_demand = torch.zeros(batch_size, 1, dtype=torch.float32)
    pickup_demand = torch.randint(demand_low, demand_high, (batch_size, pickup_size)).float() / capacity
    delivery_demand = -pickup_demand
    node_demand = torch.cat([depot_demand, pickup_demand, delivery_demand], dim=1)

    node_demand_matrix = node_demand.unsqueeze(1).expand(batch_size, node_size, node_size)
    problems = torch.stack([mat, node_demand_matrix], dim=-1)

    node_demand_sym = - node_demand
    node_demand_matrix_sym = node_demand_sym.unsqueeze(1).expand(batch_size, node_size, node_size)
    problems_sym = torch.stack([mat, node_demand_matrix_sym], dim=-1)

    return (
        problems,
        problems_sym,
        node_demand,
        node_demand_sym,
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
        "asym_bias": 0.2
    }
    batch = 2
    n_node = 21
    problems, sym_problems, node_demand, sym_node_demand = get_random_ampdtsp_problems(batch, n_node, params)

    print("="*60)
    print(f"problems shape: {problems.shape} [B,N,N,2]")
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
    print("原需求样本0:", node_demand[0])
    print("对偶需求样本0:", sym_node_demand[0])
    print("需求取反校验:", torch.allclose(node_demand[0], -sym_node_demand[0]))

    print("\n取货点数量:", (node_demand[0] > 0).sum().item())
    print("送货点数量:", (node_demand[0] < 0).sum().item())