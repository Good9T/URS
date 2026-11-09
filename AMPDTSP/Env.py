from dataclasses import dataclass
import torch
from Problem import get_random_ampdtsp_problems

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
        self.pickup_size = self.customer_size // 2
        self.capacity = env_params["capacity"]

        self.problems = None
        self.problems_sym = None
        self.node_demand = None
        self.node_demand_sym = None

        self.load = None
        self.at_the_pick = None

        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        self.pomo_size = None

        self.selected_count = None
        self.current_node = None
        self.selected_node_list = None

        self.lock = None
        self.step_state = None
        self.state_mask = None
        self.finished = None
        self.visited_flag = None

        self.sym_mode = False
        self.original_cost = None

    def load_problems(self, batch_size, aug_enable=False, aug_factor=1):
        self.batch_size = batch_size
        gen_out = get_random_ampdtsp_problems(self.batch_size, self.node_size, self.env_params)
        (
            self.problems,
            self.problems_sym,
            self.node_demand,
            self.node_demand_sym,
        ) = gen_out

        if aug_enable and aug_factor > 1:
            problems_repeated = self.problems.repeat(aug_factor, 1, 1, 1)
            node_demand_repeated = self.node_demand.repeat(aug_factor, 1)
            problems_repeated_sym = self.problems_sym.repeat(aug_factor, 1, 1, 1)
            node_demand_repeated_sym = self.node_demand_sym.repeat(aug_factor, 1)

            total = self.batch_size * aug_factor
            rand = torch.rand(total, self.pickup_size)
            perm_p = rand.argsort(dim=1) + 1
            perm_d = perm_p + self.pickup_size
            perms = torch.cat([torch.zeros(total, 1, dtype=torch.long), perm_p, perm_d], dim=1)

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
            self.batch_size = self.batch_size * aug_factor

        self.pomo_size = self.pickup_size
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
        self.sym_mode = False
        self.original_cost = self.problems[..., 0].clone()

    def load_sym(self):
        self.pomo_size = self.pickup_size
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.problems = self.problems_sym
        self.node_demand = self.node_demand_sym
        self.sym_mode = True

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.empty((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.load = torch.zeros(size=(self.batch_size, self.pomo_size))
        self.visited_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.node_size))
        self.lock = torch.zeros(size=(self.batch_size, self.pomo_size, self.node_size))
        self.lock[:, :, 1 + self.pickup_size:] = float('-inf')
        self.state_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.node_size))
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)

        start_node_idx = torch.arange(1, 1 + self.pickup_size)[None, :].expand(self.batch_size, -1)

        reward = None
        done = False
        reset_state = Reset_State(
            problems=self.problems,
            node_demand=self.node_demand,
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
        self.current_node = selected
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)

        self.at_the_pick = (selected < 1 + self.pickup_size) * (selected > 0)

        demand_list = self.node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        selected_demand = demand_list.gather(dim=2, index=selected[:, :, None]).squeeze(dim=2)
        self.load += selected_demand

        unlock = selected.clone()
        unlock[self.at_the_pick] += self.pickup_size
        self.lock[self.BATCH_IDX, self.POMO_IDX, unlock] = 0

        self.visited_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        self.state_mask = self.visited_flag.clone() + self.lock.clone()

        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] - round_error_epsilon + demand_list > 1.0
        self.state_mask[demand_too_large] = float('-inf')

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
        route = self.selected_node_list
        if self.sym_mode:
            route = route.flip(dims=[2])  # 反转路线
        node_from = route
        node_to = route.roll(dims=2, shifts=-1)
        batch_index = self.BATCH_IDX[:, :, None].expand(self.batch_size, self.pomo_size, self.selected_count)
        selected_cost = self.original_cost[batch_index, node_from, node_to]
        total_cost = selected_cost.sum(2)
        return total_cost
