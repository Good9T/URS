from dataclasses import dataclass
import torch
from Problem import get_random_vrpmb_problems

@dataclass
class Reset_State:
    problems: torch.Tensor
    node_demand: torch.Tensor
    start_node_idx: torch.Tensor

@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    selected_count: int = None
    load: torch.Tensor = None
    current_node: torch.Tensor = None
    state_mask: torch.Tensor = None
    finished: torch.Tensor = None


class Env:
    def __init__(self, **env_params):
        self.env_params = env_params
        self.customer_size = env_params["customer_size"]
        self.node_size = self.customer_size + 1
        self.capacity = env_params["problem_gen_params"]["capacity"]

        self.problems = None
        self.node_coords = None
        self.node_demand = None
        self.total_line_demand = None
        self.remain_line_demand = None
        self.dist = None

        self.load = None
        self.at_depot = None

        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        self.pomo_size = None
        self.selected_count = None
        self.current_node = None
        self.selected_node_list = None
        self.step_state = None
        self.mask = None
        self.finished = None
        self.visited_flag = None

    def load_problems(self, batch_size, aug_enable=False, aug_factor=1):
        self.batch_size = batch_size
        gen_params = self.env_params["problem_gen_params"]
        self.problems, self.node_coords, self.node_demand, self.total_line_demand = get_random_vrpmb_problems(
            self.batch_size, self.node_size, gen_params
        )
        if aug_enable:
            self.problems = self.problems.repeat(aug_factor, 1, 1)
            self.node_demand = self.node_demand.repeat(aug_factor, 1)
            self.total_line_demand = self.total_line_demand.repeat(aug_factor)
            self.batch_size = self.batch_size * aug_factor
        self.pomo_size = gen_params["linehaul_customer_size"]

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
        self.node_coords_expanded = self.node_coords.repeat(aug_factor, 1, 1)

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.empty((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.load = torch.zeros(size=(self.batch_size, self.pomo_size))
        self.visited_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.node_size))
        self.mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.node_size))
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)

        self.at_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)

        self.remain_line_demand = self.total_line_demand[:, None].expand(-1, self.pomo_size).clone()
        start_mask = self.node_demand > 1e-6
        _, node_idx = torch.where(start_mask)
        start_node_idx = node_idx.reshape(self.batch_size, -1)


        coord_diff = self.node_coords_expanded.unsqueeze(2) - self.node_coords_expanded.unsqueeze(1)
        self.dist = torch.norm(coord_diff, p=2, dim=-1)
        self.dist = self.dist[:, None, :, :].expand(
            self.batch_size, self.pomo_size, self.node_size, self.node_size
        )

        reward = None
        done = False
        return Reset_State(
            problems=self.problems,
            node_demand=self.node_demand,
            start_node_idx=start_node_idx
        ), reward, done

    def pre_step(self):
        reward = None
        done = False
        self.step_state.load = self.load
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.state_mask = torch.zeros(self.batch_size, self.pomo_size, self.node_size)
        self.step_state.finished = self.finished
        return self.step_state, reward, done

    def step(self, selected):
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)

        self.at_depot = (selected == 0)
        demand_list = self.node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        gathering_index = selected[:, :, None]
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        self.load -= selected_demand

        line_selected_flag = selected_demand > 0
        subtract = torch.where(line_selected_flag, selected_demand, torch.zeros_like(selected_demand))
        self.remain_line_demand -= subtract

        reset_load = torch.min(torch.ones_like(self.remain_line_demand), self.remain_line_demand)
        self.load[self.at_depot] = reset_load[self.at_depot]

        self.visited_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        self.visited_flag[:, :, 0][~self.at_depot] = 0
        self.mask = self.visited_flag.clone()

        round_error_epsilon = 0.00001
        demand_unfeasible = self.load[:, :, None] + round_error_epsilon < demand_list
        overload_unfeasible = self.load[:, :, None] - demand_list - round_error_epsilon > 1.0
        self.mask[demand_unfeasible | overload_unfeasible] = float('-inf')

        new_finished = (self.visited_flag == float('-inf')).all(dim=2)
        self.finished = self.finished | new_finished
        self.mask[:, :, 0][self.finished] = 0

        self._update_step_state()

        done = self.finished.all()
        if done:
            reward = -self._get_total_cost()
        else:
            reward = None
        return self.step_state, reward, done

    def _update_step_state(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.state_mask = self.mask
        self.step_state.finished = self.finished

    def _get_total_cost(self):
        index_to_gather = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        node_coords = self.problems[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        seq_ordered = node_coords.gather(dim=2, index=index_to_gather)
        seq_rolled = seq_ordered.roll(dims=2, shifts=-1)
        segment_lengths = ((seq_ordered - seq_rolled) ** 2).sum(3).sqrt()
        travel_distances = segment_lengths.sum(2)
        return travel_distances

    def get_cur_feature(self):
        if self.current_node is None:
            return None, None
        current_node = self.current_node[:, :, None, None].expand(self.batch_size, self.pomo_size, 1,
                                                                  self.node_size)

        # Compute the relative distance
        cur_dist = torch.take_along_dim(
            self.dist,  # 不再加 [:, None, :, :]
            current_node,
            dim=2).squeeze(2)
        # shape: (batch, multi, problem)
        return cur_dist