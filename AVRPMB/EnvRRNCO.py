# EnvRRNCO.py
from dataclasses import dataclass
import torch
from ProblemRRNCO import load_rrnco_problems


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


class EnvRRNCO:
    def __init__(self, **env_params):
        self.env_params = env_params
        self.node_size = None
        self.customer_size = None

        self.problems = None
        self.problems_sym = None
        self.node_demand = None
        self.node_demand_sym = None

        self.total_linehaul_demand = None
        self.total_backhaul_demand = None

        self.remain_linehaul_demand = None

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
        self.state_mask = None
        self.finished = None
        self.visited_flag = None

        self.symmetry_flag = False
        self.scaler = None

    def load_problems(self, batch_size, aug_enable=False, aug_factor=1, dist_batch=None):
        from ProblemRRNCO import load_rrnco_problems

        self.batch_size = batch_size
        gen_out = load_rrnco_problems(self.env_params, dist_batch)
        (
            self.problems,
            self.problems_sym,
            self.node_demand,
            self.node_demand_sym,
            self.total_linehaul_demand,
            self.total_backhaul_demand,
            self.scaler
        ) = gen_out

        self.node_size = self.problems.shape[1]
        self.customer_size = self.node_size - 1

        if aug_enable and aug_factor > 1:
            problems_repeated = self.problems.repeat(aug_factor, 1, 1, 1)
            node_demand_repeated = self.node_demand.repeat(aug_factor, 1)
            problems_repeated_sym = self.problems_sym.repeat(aug_factor, 1, 1, 1)
            node_demand_repeated_sym = self.node_demand_sym.repeat(aug_factor, 1)

            total = self.batch_size * aug_factor
            rand = torch.rand(total, self.customer_size)
            customer_perms = rand.argsort(dim=1) + 1
            perms = torch.cat([torch.zeros(total, 1, dtype=torch.long), customer_perms], dim=1)

            idx = torch.arange(total)
            problems_aug = problems_repeated[
                idx[:, None, None, None],
                perms[:, :, None, None],
                perms[:, None, :, None],
                torch.arange(2)[None, None, None, :]
            ]
            node_demand_aug = node_demand_repeated[idx[:, None], perms]
            problems_aug_sym = problems_repeated_sym[
                idx[:, None, None, None],
                perms[:, :, None, None],
                perms[:, None, :, None],
                torch.arange(2)[None, None, None, :]
            ]
            node_demand_aug_sym = node_demand_repeated_sym[idx[:, None], perms]

            self.problems = problems_aug
            self.problems_sym = problems_aug_sym
            self.node_demand = node_demand_aug
            self.node_demand_sym = node_demand_aug_sym
            self.total_linehaul_demand = self.total_linehaul_demand.repeat(aug_factor)
            self.total_backhaul_demand = self.total_backhaul_demand.repeat(aug_factor)
            self.scaler = self.scaler.repeat(aug_factor, 1, 1)
            self.batch_size = self.batch_size * aug_factor

        self.pomo_size = (self.node_demand[0] > 0).sum().item()
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
        self.symmetry_flag = False

    def load_sym(self):
        # 对称视图用backhaul作为起点
        backhaul_count = (self.node_demand_sym[0] > 0).sum().item()
        self.pomo_size = backhaul_count
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
        self.symmetry_flag = True

        self.problems = self.problems_sym
        self.node_demand = self.node_demand_sym

        temp_total_linehaul_demand = self.total_linehaul_demand.clone()
        self.total_linehaul_demand = self.total_backhaul_demand
        self.total_backhaul_demand = temp_total_linehaul_demand

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.empty((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.load = torch.zeros(size=(self.batch_size, self.pomo_size))
        self.visited_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.node_size))
        self.state_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.node_size))
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        self.at_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)

        current_node_demand = self.node_demand
        current_total_linehaul_demand = self.total_linehaul_demand
        start_mask = current_node_demand > 1e-6

        self.remain_linehaul_demand = current_total_linehaul_demand[:, None].expand(-1, self.pomo_size).clone()
        _, node_index = torch.where(start_mask)
        start_node_idx = node_index.reshape(self.batch_size, -1)

        reward = None
        done = False
        reset_state = Reset_State(
            problems=self.problems,
            node_demand=current_node_demand,
            start_node_idx=start_node_idx,
        )
        return reset_state, reward, done

    def pre_step(self):
        reward = None
        done = False
        self.step_state.load = self.load
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.state_mask = self.state_mask
        self.step_state.finished = self.finished
        return self.step_state, reward, done

    def step(self, selected):
        self.selected_count += 1
        selected = selected[:, :self.pomo_size]
        self.current_node = selected
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)

        self.at_depot = (selected == 0)
        demand_list = self.node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        gathering_index = selected[:, :, None]
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        self.load -= selected_demand

        line_selected_flag = selected_demand > 0
        subtract_value = torch.where(line_selected_flag, selected_demand, torch.zeros_like(selected_demand))
        self.remain_linehaul_demand -= subtract_value

        reset_load_value = torch.min(self.remain_linehaul_demand, torch.ones_like(self.remain_linehaul_demand))
        self.load[self.at_depot] = reset_load_value[self.at_depot]

        self.visited_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        self.visited_flag[:, :, 0][~self.at_depot] = 0
        self.state_mask = self.visited_flag.clone()

        round_error_epsilon = 0.00001
        demand_unfeasible = self.load[:, :, None] + round_error_epsilon < demand_list
        overload_unfeasible = self.load[:, :, None] - demand_list - round_error_epsilon > 1.0
        self.state_mask[demand_unfeasible | overload_unfeasible] = float('-inf')

        new_finished = (self.visited_flag == float('-inf')).all(dim=2)
        self.finished = self.finished | new_finished
        self.state_mask[:, :, 0][self.finished] = 0
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
        self.step_state.state_mask = self.state_mask
        self.step_state.finished = self.finished

    def _get_total_cost(self):
        node_from = self.selected_node_list
        node_to = self.selected_node_list.roll(dims=2, shifts=-1)
        batch_index = self.BATCH_IDX[:, :, None].expand(self.batch_size, self.pomo_size, self.selected_count)
        selected_cost_norm = self.problems[batch_index, node_from, node_to, 0]
        scaler_exp = self.scaler.expand(self.batch_size, self.pomo_size, self.selected_count)
        total_cost = (selected_cost_norm * scaler_exp).sum(2)
        return total_cost