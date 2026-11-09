import torch
import numpy as np
import pandas as pd

def get_random_problems(batch_size, customer_size, problem_gen_params):
    scaler = problem_gen_params['scaler']
    capacity = problem_gen_params['capacity']
    demand_min = problem_gen_params['demand_min']
    demand_max = problem_gen_params['demand_max']

    depot_size = 1
    node_num = depot_size + customer_size * 2

    depot_xy = torch.rand(batch_size, depot_size, 2)
    pick_xy = torch.rand(batch_size, customer_size, 2)
    delivery_xy = torch.rand(batch_size, customer_size, 2)

    pick_demand = torch.randint(demand_min, demand_max, (batch_size, customer_size))
    delivery_demand = -pick_demand

    pick_demand = pick_demand / capacity
    delivery_demand = delivery_demand / capacity

    depot_demand = torch.zeros(batch_size, 1)
    node_demand = torch.cat([depot_demand, pick_demand, delivery_demand], dim=1)  # (B, N)

    node_coords = torch.cat([depot_xy, pick_xy, delivery_xy], dim=1)
    node_coords_dual = torch.cat([depot_xy, delivery_xy, pick_xy], dim=1)

    coord_diff = node_coords.unsqueeze(2) - node_coords.unsqueeze(1)
    dist = torch.norm(coord_diff, p=2, dim=-1)
    dist[:, torch.arange(node_num), torch.arange(node_num)] = 0.0
    scaled_dist = dist.float() / scaler

    coord_diff_d = node_coords_dual.unsqueeze(2) - node_coords_dual.unsqueeze(1)
    dist_d = torch.norm(coord_diff_d, p=2, dim=-1)
    dist_d[:, torch.arange(node_num), torch.arange(node_num)] = 0.0
    scaled_dist_d = dist_d.float() / scaler

    demand_mat = node_demand.unsqueeze(1).expand(batch_size, node_num, node_num)
    demand_mat_d = node_demand.unsqueeze(1).expand(batch_size, node_num, node_num)

    problems = torch.stack((scaled_dist, demand_mat), dim=-1)
    dual_problems = torch.stack((scaled_dist_d, demand_mat_d), dim=-1)

    return problems, dual_problems, node_coords, node_demand


