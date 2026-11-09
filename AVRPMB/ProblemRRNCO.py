# ProblemRRNCO.py
import torch
import numpy as np

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


# ProblemRRNCO.py
def load_rrnco_problems(params, dist_batch):
    device = dist_batch.device
    batch_size, node_size, _ = dist_batch.shape
    customer_size = node_size - 1

    demand_low = params['demand_low']
    demand_high = params['demand_high']
    linehaul_ratio = params['linehaul_ratio']
    linehaul_customer_size = int(customer_size * linehaul_ratio)
    capacity = get_capacity(node_size)

    # 归一化
    max_dist = dist_batch.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
    scaler = max_dist
    dist_norm = dist_batch / (scaler + 1e-8)

    # 生成需求，放到同一个device
    random_permutation = torch.rand(batch_size, customer_size, device=device).argsort(dim=-1)
    linehaul_mask = torch.zeros(batch_size, customer_size, dtype=torch.bool, device=device)
    linehaul_mask.scatter_(1, random_permutation[:, :linehaul_customer_size], True)

    absolute_demand = torch.randint(demand_low, demand_high + 1, (batch_size, customer_size), device=device).float() / capacity
    customer_demand = torch.where(linehaul_mask, absolute_demand, -absolute_demand)

    node_demand = torch.zeros(batch_size, node_size, device=device)
    node_demand[:, 1:] = customer_demand
    node_demand_sym = -node_demand

    node_demand_matrix = node_demand.unsqueeze(1).expand(batch_size, node_size, node_size)
    problems = torch.stack((dist_norm, node_demand_matrix), dim=-1)

    node_demand_matrix_sym = node_demand_sym.unsqueeze(1).expand(batch_size, node_size, node_size)
    problems_sym = torch.stack((dist_norm, node_demand_matrix_sym), dim=-1)

    linehaul_only = torch.where(node_demand > 0, node_demand, torch.zeros_like(node_demand))
    backhaul_only = torch.where(node_demand < 0, -node_demand, torch.zeros_like(node_demand))
    total_linehaul = linehaul_only.sum(dim=-1)
    total_backhaul = backhaul_only.sum(dim=-1)
    return (
        problems,
        problems_sym,
        node_demand,
        node_demand_sym,
        total_linehaul,
        total_backhaul,
        scaler
    )