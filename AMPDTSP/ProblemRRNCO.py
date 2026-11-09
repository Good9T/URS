import torch
from PIL.TiffImagePlugin import DATE_TIME

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

def load_rrnco_problems(params, dist_batch):
    device = dist_batch.device
    batch_size, node_size, _ = dist_batch.shape
    customer_size = node_size - 1
    demand_low = params['demand_low']
    demand_high = params['demand_high']
    capacity = get_capacity(node_size)
    if customer_size % 2 == 1:
        actual_customer_size = customer_size - 1
        dist_batch = dist_batch[:, :actual_customer_size + 1, :actual_customer_size + 1]
        batch_size, node_size, _ = dist_batch.shape
        customer_size = actual_customer_size

    pickup_size = customer_size // 2

    max_dist = dist_batch.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    scaler = max_dist
    dist_norm = dist_batch / (scaler + 1e-8)

    depot_demand = torch.zeros(batch_size, 1, dtype=torch.float32)
    pickup_demand = torch.randint(demand_low, demand_high, (batch_size, pickup_size)).float() / capacity
    delivery_demand = -pickup_demand
    node_demand = torch.cat([depot_demand, pickup_demand, delivery_demand], dim=1)

    node_demand_matrix = node_demand.unsqueeze(1).expand(batch_size, node_size, node_size)
    problems = torch.stack([dist_norm, node_demand_matrix], dim=-1)

    node_demand_sym = - node_demand
    node_demand_matrix_sym = node_demand_sym.unsqueeze(1).expand(batch_size, node_size, node_size)
    problems_sym = torch.stack([dist_norm, node_demand_matrix_sym], dim=-1)

    return (
        problems,
        problems_sym,
        node_demand,
        node_demand_sym,
        scaler,
    )
