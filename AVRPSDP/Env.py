from dataclasses import dataclass
import torch
from Problem import get_random_avrpsdp_problems

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
    current_dist: torch.Tensor = None

class Env:
    def __init__(self, **env_params):
        self.env_params = env_params
        self.customer_size = env_params["customer_size"]
        self.node_size = self.customer_size + 1
        self.capacity = env_params["capacity"]

        self.problems = None
        self.problems_sym = None
        self.node_demand = None
        self.node_demand_sym = None

        self.total_delivery = None
        self.total_pickup = None
        self.total_delivery_sym = None
        self.total_pickup_sym = None

        self.remain_delivery = None
        self.remain_pickup = None

        self.load = None
        self.load_d = None
        self.load_p = None
        self.at_depot = None
        self.safe_ratio = None
        self.safe_ratio_sym = None

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

        self.sym_mode = False
        self.original_cost = None

    def load_problems(self, batch_size, aug_enable=False, aug_factor=1):
        self.batch_size = batch_size
        gen_out = get_random_avrpsdp_problems(self.batch_size, self.node_size, self.env_params)
        (
            self.problems,
            self.problems_sym,
            self.node_demand,
            self.node_demand_sym,
            self.total_delivery,
            self.total_delivery_sym,
            self.total_pickup,
            self.total_pickup_sym,
            self.safe_ratio,
            self.safe_ratio_sym
        ) = gen_out

        if aug_enable and aug_factor > 1:
            problems_repeated = self.problems.repeat(aug_factor, 1, 1, 1)
            node_demand_repeated = self.node_demand.repeat(aug_factor, 1, 1)
            problems_repeated_sym = self.problems_sym.repeat(aug_factor, 1, 1, 1)
            node_demand_repeated_sym = self.node_demand_sym.repeat(aug_factor, 1, 1)

            total = self.batch_size * aug_factor
            rand = torch.rand(total, self.node_size - 1)
            customer_perms = rand.argsort(dim=1) + 1
            perms = torch.cat([torch.zeros(total, 1, dtype=torch.long), customer_perms], dim=1)
            # (total, node_size)

            idx = torch.arange(total)
            problems_aug = problems_repeated[
                idx[:, None, None, None],
                perms[:, :, None, None],
                perms[:, None, :, None],
                torch.arange(3)[None, None, None, :]
            ]
            node_demand_aug = node_demand_repeated[
                idx[:, None, None],
                perms[:, :, None],
                torch.arange(2)[None, None, :]
            ]
            problems_aug_sym = problems_repeated_sym[
                idx[:, None, None, None],
                perms[:, :, None, None],
                perms[:, None, :, None],
                torch.arange(3)[None, None, None, :]
            ]
            node_demand_aug_sym = node_demand_repeated_sym[
                idx[:, None, None],
                perms[:, :, None],
                torch.arange(2)[None, None, :]
            ]

            self.problems = problems_aug
            self.problems_sym = problems_aug_sym
            self.node_demand = node_demand_aug
            self.node_demand_sym = node_demand_aug_sym
            self.total_delivery = self.total_delivery.repeat(aug_factor)
            self.total_delivery_sym = self.total_delivery_sym.repeat(aug_factor)
            self.total_pickup = self.total_pickup.repeat(aug_factor)
            self.total_pickup_sym = self.total_pickup_sym.repeat(aug_factor)
            self.safe_ratio = self.safe_ratio.repeat(aug_factor)
            self.safe_ratio_sym = self.safe_ratio_sym.repeat(aug_factor)
            self.batch_size = self.batch_size * aug_factor

        self.pomo_size = self.customer_size
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
        self.original_cost = self.problems[..., 0].clone()
        self.sym_mode = False

    def load_sym(self):
        self.pomo_size = self.customer_size
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.problems = self.problems_sym
        self.node_demand = self.node_demand_sym

        temp_total_delivery = self.total_delivery.clone()
        self.total_delivery = self.total_delivery_sym
        self.total_delivery_sym = temp_total_delivery

        temp_total_pickup = self.total_pickup.clone()
        self.total_pickup = self.total_pickup_sym
        self.total_pickup_sym = temp_total_pickup

        temp_safe_ratio = self.safe_ratio.clone()
        self.safe_ratio = self.safe_ratio_sym
        self.safe_ratio_sym = temp_safe_ratio

        self.sym_mode = True


    def reset(self):
        self.selected_count = 0
        self.current_node = None
        self.selected_node_list = torch.empty((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)

        self.load_d = torch.zeros((self.batch_size, self.pomo_size))
        self.load_p = torch.zeros((self.batch_size, self.pomo_size))
        self.load = self.load_d + self.load_p

        self.visited_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.node_size))
        self.state_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.node_size))
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        self.at_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)

        self.remain_delivery = self.total_delivery[:, None].expand(-1, self.pomo_size).clone()
        self.remain_pickup = self.total_pickup[:, None].expand(-1, self.pomo_size).clone()

        start_node_idx = torch.arange(1, self.node_size)[None, :].expand(self.batch_size, -1)
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
        self.step_state.state_mask = self.state_mask
        self.step_state.finished = self.finished
        return self.step_state, reward, done

    def step(self, selected):
        self.selected_count += 1
        self.current_node = selected
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)

        self.at_depot = (selected == 0)
        demand_list = self.node_demand[:, None, :, :].expand(self.batch_size, self.pomo_size, -1, -1)
        gathering_index = selected[:, :, None, None].expand(-1, -1, -1, 2)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        selected_d = selected_demand[..., 0]
        selected_p = selected_demand[..., 1]

        self.load_d -= selected_d
        self.load_p += selected_p
        self.load = self.load_d + self.load_p
        self.remain_delivery -= selected_d

        safe_capacity_exp = self.safe_ratio[:, None].expand(-1, self.pomo_size)
        reset_capacity = torch.ones_like(self.remain_delivery) * safe_capacity_exp
        reload_d = torch.min(reset_capacity, self.remain_delivery)
        self.load_d[self.at_depot] = reload_d[self.at_depot]
        self.load_p[self.at_depot] = 0.0
        self.load = self.load_d + self.load_p

        self.visited_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        self.visited_flag[:, :, 0][~self.at_depot] = 0
        self.state_mask = self.visited_flag.clone()

        round_error_epsilon = 0.00001
        demand_unfeasible = self.load_d[:, :, None] + round_error_epsilon < demand_list[..., 0]
        future_total_load = self.load[:, :, None] - demand_list[..., 0] + demand_list[..., 1]
        overload_unfeasible = future_total_load - round_error_epsilon > 1.0
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
        self.step_state.cur_dist = self._get_cur_dist()

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

    def _get_cur_dist(self):
        if self.current_node is None:
            return None
        dist = self.problems[..., 0]
        cur_dist = dist[self.BATCH_IDX, self.current_node, :]
        return cur_dist