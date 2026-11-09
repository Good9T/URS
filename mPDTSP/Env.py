from dataclasses import dataclass
import torch

from Problem import get_random_problems

@dataclass
class Reset_State:
    problems: torch.Tensor

@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    selected_count: int = None
    current_node: torch.Tensor = None
    state_mask: torch.Tensor = None
    finished: torch.Tensor = None

class Env:
    def __init__(self, **env_params):
        self.env_params = env_params
        self.pomo_size = env_params['pomo_size']
        self.customer_size = env_params['customer_size']
        self.node_size = 1 + self.customer_size * 2

        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None

        self.problems = None
        self.dual_problems = None
        self.node_coords = None
        self.node_demand = None

        self.selected_count = None
        self.current_node = None
        self.selected_node_list = None
        self.visited_flag = None
        self.mask = None
        self.lock = None
        self.load = None
        self.finished = None
        self.step_state = None

    def load_problems(self, batch_size, aug_enable=False, aug_factor=1):
        self.batch_size = batch_size
        problem_gen_params = self.env_params['problem_gen_params']

        self.problems, self.dual_problems, self.node_coords, self.node_demand = get_random_problems(
            self.batch_size, self.customer_size, problem_gen_params
        )
        if aug_enable:
            self.problems = self.problems.repeat(aug_factor, 1, 1, 1)
            self.dual_problems = self.dual_problems.repeat(aug_factor, 1, 1, 1)
            self.node_coords = self.node_coords.repeat(aug_factor, 1, 1)
            self.node_demand = self.node_demand.repeat(aug_factor, 1)
            self.batch_size = self.batch_size * aug_factor

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

    def load_dual_problems(self):
        self.problems = self.dual_problems

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.empty((self.batch_size, self.pomo_size, 0), dtype=torch.long)

        self.visited_flag = torch.zeros((self.batch_size, self.pomo_size, self.node_size))
        self.lock = torch.zeros((self.batch_size, self.pomo_size, self.node_size))
        self.mask = torch.zeros((self.batch_size, self.pomo_size, self.node_size))
        self.load = torch.zeros(size=(self.batch_size, self.pomo_size))
        self.lock[:, :, 1 + self.customer_size:] = float('-inf')
        self.finished = torch.zeros((self.batch_size, self.pomo_size), dtype=torch.bool)

        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        return Reset_State(problems=self.problems), None, False

    def pre_step(self):
        reward = None
        done = False
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.state_mask = self.mask
        self.step_state.finished = self.finished
        return self.step_state, reward, done

    def step(self, selected):
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)

        self.at_the_pick = (selected < 1 + self.customer_size) * (selected > 0)
        demand_list = self.node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        self.load += self.node_demand[self.BATCH_IDX, selected]

        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] - round_error_epsilon + demand_list > 1.0

        self.visited_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')

        unlock = selected.clone()
        unlock[self.at_the_pick] += self.customer_size
        self.lock[self.BATCH_IDX, self.POMO_IDX, unlock] = 0

        self.mask = self.visited_flag.clone() + self.lock.clone()
        self.mask[demand_too_large] = float('-inf')
        self.mask = self.mask.clone()

        new_finished = (self.visited_flag == float('-inf')).all(dim=2)
        self.finished = self.finished + new_finished

        self.mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.current_node = self.current_node
        self.step_state.state_mask = self.mask
        self.step_state.finished = self.finished

        done = self.finished.all()
        reward = -self._get_total_cost() if done else None
        return self.step_state, reward, done

    def _get_total_cost(self):
        node_from = self.selected_node_list
        node_to = self.selected_node_list.roll(dims=2, shifts=-1)
        batch_index = self.BATCH_IDX[:, :, None].expand(self.batch_size, self.pomo_size, self.selected_count)
        selected_cost = self.problems[batch_index, node_from, node_to, 0].squeeze(dim=-1)
        total_cost = selected_cost.sum(2)
        return total_cost