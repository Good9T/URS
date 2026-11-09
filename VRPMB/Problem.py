import torch

def get_random_vrpmb_problems(batch_size, node_size, params):
    scaler = params["scaler"]
    capacity = params["capacity"]
    customer_size = node_size - 1
    demand_low = params["demand_low"]
    demand_high = params["demand_high"]
    linehaul_customer_size = params["linehaul_customer_size"]

    node_coordinates = torch.rand(batch_size, node_size, 2)

    random_permutation = torch.rand(batch_size, customer_size).argsort(dim=-1)
    linehaul_mask = torch.zeros(batch_size, customer_size, dtype=torch.bool)
    linehaul_mask.scatter_(1, random_permutation[:, :linehaul_customer_size], True)

    absolute_demand = torch.randint(demand_low, demand_high, (batch_size, customer_size)).float() / capacity
    customer_demand = torch.where(linehaul_mask, absolute_demand, -absolute_demand)

    node_demand = torch.zeros(batch_size, node_size)
    node_demand[:, 1:] = customer_demand
    sym_node_demand = -node_demand

    linehaul_only_demand = torch.where(node_demand > 0, node_demand, 0.0)
    backhaul_only_demand = torch.where(node_demand < 0, -node_demand, 0.0)
    total_linehaul_demand = linehaul_only_demand.sum(dim=-1)
    total_backhaul_demand = backhaul_only_demand.sum(dim=-1)

    coordinate_diff = node_coordinates.unsqueeze(2) - node_coordinates.unsqueeze(1)
    distance_matrix = torch.norm(coordinate_diff, p=2, dim=-1)
    distance_matrix[:, torch.arange(node_size), torch.arange(node_size)] = 0.0
    scaled_distance_matrix = distance_matrix / scaler
    node_demand_matrix = node_demand.unsqueeze(1).expand(batch_size, node_size, node_size)
    problems = torch.stack((scaled_distance_matrix, node_demand_matrix), dim=-1)

    sym_node_demand_matrix = sym_node_demand.unsqueeze(1).expand(batch_size, node_size, node_size)
    sym_problems = torch.stack((scaled_distance_matrix.clone(), sym_node_demand_matrix), dim=-1)

    return problems, sym_problems, node_coordinates, node_demand, sym_node_demand, total_linehaul_demand, total_backhaul_demand


if __name__ == "__main__":
    test_config = {
        "scaler": 1,
        "capacity": 30,
        "demand_low": 1,
        "demand_high": 10,
        "linehaul_customer_size": 10
    }
    batch_sample_count = 4
    total_node_count = 20
    problems, sym_problems, node_coordinates, node_demand, sym_node_demand, total_linehaul_demand, total_backhaul_demand = get_random_vrpmb_problems(batch_sample_count, total_node_count, test_config)

    # 维度校验
    print("problems shape:", problems.shape)
    print("sym_problems shape:", sym_problems.shape)
    print("node_coords shape:", node_coordinates.shape)
    print("node_demand shape:", node_demand.shape)
    print("sym_node_demand shape:", sym_node_demand.shape)
    print("total_linehaul_demand shape:", total_linehaul_demand.shape)
    print("total_backhaul_demand shape:", total_backhaul_demand.shape)

    # 基础逻辑校验
    assert torch.all(node_demand[:, 0] == 0)
    assert torch.all(sym_node_demand[:, 0] == 0)
    assert torch.allclose(sym_node_demand, -node_demand)
    assert torch.all(total_linehaul_demand >= 0)
    assert torch.all(total_backhaul_demand >= 0)
    assert torch.all((node_demand > 0).sum(-1) == test_config["linehaul_customer_size"])
    print("✅ 全部校验通过")