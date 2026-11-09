import torch

def get_random_vrpsdp_problems(batch_size, node_size, params):
    scaler = params["scaler"]
    capacity = params["capacity"]
    customer_size = node_size - 1
    d_low = params["d_low"]
    d_high = params["d_high"]
    p_low = params["p_low"]
    p_high = params["p_high"]

    node_coords = torch.rand(batch_size, node_size, 2)
    coord_diff = node_coords.unsqueeze(2) - node_coords.unsqueeze(1)
    distance_matrix = torch.norm(coord_diff, p=2, dim=-1)
    distance_matrix[:, torch.arange(node_size), torch.arange(node_size)] = 0.0
    scaled_distance_matrix = distance_matrix / scaler

    dp_demand = torch.zeros(batch_size, node_size, 2, dtype=torch.float32)
    dp_demand[:, 1:, 0] = torch.randint(d_low, d_high, size=(batch_size, customer_size)).float()
    dp_demand[:, 1:, 1] = torch.randint(p_low, p_high, size=(batch_size, customer_size)).float()
    dp_demand = dp_demand / capacity

    sym_dp_demand = dp_demand.clone()
    sym_dp_demand = sym_dp_demand[..., [1, 0]]

    d_exp = dp_demand[:, None, :, 0].expand(-1, node_size, -1)
    p_exp = dp_demand[:, None, :, 1].expand(-1, node_size, -1)
    problems = torch.stack([scaled_distance_matrix, d_exp, p_exp], dim=-1)

    sym_d_exp = sym_dp_demand[:, None, :, 0].expand(-1, node_size, -1)
    sym_p_exp = sym_dp_demand[:, None, :, 1].expand(-1, node_size, -1)
    sym_problems = torch.stack([scaled_distance_matrix, sym_d_exp, sym_p_exp], dim=-1)

    total_delivery = dp_demand[:, :, 0].sum(dim=1)
    sym_total_delivery = sym_dp_demand[:, :, 0].sum(dim=1)

    pd_diff = dp_demand[:, 1:, 1] - dp_demand[:, 1:, 0]
    global_pd_max = pd_diff.max(dim=1).values
    safe_ratio = 1.0 - global_pd_max

    sym_pd_diff = sym_dp_demand[:, 1:, 1] - sym_dp_demand[:, 1:, 0]
    sym_global_pd_max = sym_pd_diff.max(dim=1).values
    sym_safe_ratio = 1.0 - sym_global_pd_max

    return (
        problems,
        sym_problems,
        node_coords,
        dp_demand,
        sym_dp_demand,
        total_delivery,
        sym_total_delivery,
        safe_ratio,
        sym_safe_ratio
    )


if __name__ == "__main__":
    test_params = {
        "scaler": 10.0,
        "capacity": 30,
        "d_low": 1,
        "d_high": 10,
        "p_low": 1,
        "p_high": 10
    }
    batch = 4
    node = 20
    out = get_random_vrpsdp_problems(batch, node, test_params)
    names = [
        "problems", "sym_problems", "node_coords",
        "dp_demand", "sym_dp_demand",
        "total_delivery", "sym_total_delivery",
        "safe_capacity", "sym_safe_ratio"
    ]
    print("==== DVRS Multi-view Problem Output Shape Check ====")
    for name, tensor in zip(names, out):
        print(f"{name:18s} | shape: {tensor.shape}")

    print("\nDemand swap check sample[0,1]:")
    print("origin d,p:", out[3][0, 1])
    print("sym   d,p:", out[4][0, 1])