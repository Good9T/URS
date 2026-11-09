import torch


def get_capacity(node_size):
    if node_size <= 20:
        return 30
    elif node_size <= 50:
        return 40
    elif node_size <= 100:
        return 50
    elif node_size <= 200:
        return 60
    elif node_size <= 500:
        return 100
    else:
        return 150


def load_rrnco_problems_avrpsdp(params, dist_batch):
    device = dist_batch.device
    batch_size, node_size, _ = dist_batch.shape
    customer_size = node_size - 1

    d_low = params['demand_low']
    d_high = params['demand_high']
    p_low = params['pickup_low']
    p_high = params['pickup_high']
    capacity = get_capacity(node_size)

    # 距离归一化
    max_dist = dist_batch.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    scaler = max_dist
    dist_norm = dist_batch / (scaler + 1e-8)

    # 每个客户节点独立生成 delivery / pickup 需求
    dp_demand = torch.zeros(batch_size, node_size, 2, dtype=torch.float32, device=device)
    dp_demand[:, 1:, 0] = torch.randint(
        d_low, d_high, size=(batch_size, customer_size), device=device
    ).float()
    dp_demand[:, 1:, 1] = torch.randint(
        p_low, p_high, size=(batch_size, customer_size), device=device
    ).float()
    dp_demand = dp_demand / capacity

    # 对称视图：交换 delivery / pickup 两个通道
    dp_demand_sym = dp_demand.clone()[..., [1, 0]]

    # 广播成 (B, N, N)
    d_exp = dp_demand[:, None, :, 0].expand(-1, node_size, -1)
    p_exp = dp_demand[:, None, :, 1].expand(-1, node_size, -1)
    problems = torch.stack([dist_norm, d_exp, p_exp], dim=-1)

    d_exp_sym = dp_demand_sym[:, None, :, 0].expand(-1, node_size, -1)
    p_exp_sym = dp_demand_sym[:, None, :, 1].expand(-1, node_size, -1)
    problems_sym = torch.stack([dist_norm, d_exp_sym, p_exp_sym], dim=-1)

    # 需求总量
    total_delivery = dp_demand[:, :, 0].sum(dim=1)
    total_delivery_sym = dp_demand_sym[:, :, 0].sum(dim=1)
    total_pickup = dp_demand[:, :, 1].sum(dim=1)
    total_pickup_sym = dp_demand_sym[:, :, 1].sum(dim=1)

    # safe_ratio：保证回仓装 delivery 时不超容量
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
        safe_ratio_sym,
        scaler,
    )