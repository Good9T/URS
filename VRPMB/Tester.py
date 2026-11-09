import torch
import os
from logging import getLogger

from Env import Env
from Model import Model
from vrpmb_route_plotter import draw_dvrs_route
from utils import get_result_folder, AverageMeter, TimeEstimator


class Tester:
    def __init__(self, env_params, model_params, tester_params):
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params
        self.logger = getLogger(name='tester')
        self.result_folder = get_result_folder()

        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            device = torch.device('cuda', cuda_device_num)
        else:
            device = torch.device("cpu")
        torch.set_default_device(device)

        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        model_load = self.tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.time_estimator = TimeEstimator()

    def run(self):
        score = AverageMeter()
        aug_score = AverageMeter()
        test_num_episode = self.tester_params['test_episodes']
        episode = 0
        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)
            batch_score, batch_aug_score = self._test_one_batch(batch_size)
            score.update(batch_score)
            aug_score.update(batch_aug_score)
            episode += batch_size

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score.avg, aug_score.avg))

            all_done = (episode == test_num_episode)
            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG MAIN VIEW SCORE: {:.4f} ".format(score.avg))
                self.logger.info(" AUG DUAL VIEW BEST SCORE: {:.4f} ".format(aug_score.avg))

    def _test_one_batch(self, batch_size):
        aug_enable = self.tester_params['aug_enable']
        aug_factor = self.tester_params['aug_factor'] if aug_enable else 1

        self.model.eval()
        with torch.no_grad():
            # ========== 主线视图 Main View ==========
            self.env.load_problems(batch_size, aug_enable=aug_enable, aug_factor=aug_factor)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

            state, reward, done = self.env.pre_step()
            while not done:
                selected, _ = self.model(state)
                state, reward, done = self.env.step(selected)

            selected_nodes = self.env.selected_node_list.clone()
            node_coords = self.env.node_coords.clone()
            node_demand = self.env.node_demand.clone()

            aug_reward_main = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
            max_pomo_reward_main, _ = aug_reward_main.max(dim=2)
            batch_reward, _ = max_pomo_reward_main.max(dim=0)
            no_aug_main = max_pomo_reward_main[0, :]  # aug=0 主视图单样本奖励

            # ========== 对称对偶视图 Symmetry Dual View ==========
            self.env.load_symmetry_problems()
            sym_reset_state, _, _ = self.env.reset()
            self.model.pre_forward(sym_reset_state)

            sym_state, sym_reward, sym_done = self.env.pre_step()
            while not sym_done:
                sym_selected, _ = self.model(sym_state)
                sym_state, sym_reward, sym_done = self.env.step(sym_selected)

            aug_reward_sym = sym_reward.reshape(aug_factor, batch_size, self.env.pomo_size)
            max_pomo_reward_sym, _ = aug_reward_sym.max(dim=2)
            sym_batch_reward, _ = max_pomo_reward_sym.max(dim=0)
            no_aug_sym = max_pomo_reward_sym[0, :]  # aug=0 对称视图单样本奖励

            # 双视图无增强基线：逐样本取主/对称aug0最优
            no_aug_dual_reward = torch.max(no_aug_main, no_aug_sym)
            no_aug_dual_score = -no_aug_dual_reward.float().mean()

            # 带增强双视图全局最优分数
            combined_max_reward = torch.max(batch_reward, sym_batch_reward)
            aug_dual_score = -combined_max_reward.float().mean()

            # ========== 单样本打印+绘图（仅test_episodes=1执行） ==========
            if self.tester_params['test_episodes'] == 1:
                # 1. 主线所有aug全局最优
                max_pomo_main, _ = aug_reward_main.max(dim=2)
                batch_best_main_val, aug_idx_main = max_pomo_main.max(dim=0)
                best_pomo_global_main = aug_reward_main[aug_idx_main[0], 0, :].argmax().item()
                global_main_cost = -aug_reward_main[aug_idx_main[0], 0, best_pomo_global_main].item()
                global_main_path = selected_nodes[0, best_pomo_global_main].cpu().tolist()

                # 2. 对称所有aug全局最优
                max_pomo_sym, _ = aug_reward_sym.max(dim=2)
                batch_best_sym_val, aug_idx_sym = max_pomo_sym.max(dim=0)
                best_pomo_global_sym = aug_reward_sym[aug_idx_sym[0], 0, :].argmax().item()
                global_sym_cost = -aug_reward_sym[aug_idx_sym[0], 0, best_pomo_global_sym].item()
                global_sym_path = self.env.selected_node_list[0, best_pomo_global_sym].cpu().tolist()

                demand_list = node_demand[0].cpu().tolist()

                print("\n==== DVRS Dual View GLOBAL ALL AUG Optimal Sample Info ====")
                print(f"Main View Global Best POMO {best_pomo_global_main} | Cost = {global_main_cost:.4f}")
                print(f"Main Global Opt Route: {global_main_path}")
                print(f"Symmetry View Global Best POMO {best_pomo_global_sym} | Cost = {global_sym_cost:.4f}")
                print(f"Sym Global Opt Route: {global_sym_path}")
                print(f"All Node Demand: {demand_list}")

                # 打印载重&剩余干线库存轨迹
                print("\n==== Load & Remain Linehaul Trace (Max Capacity=1.0) ====")
                total_line_d = sum(d for d in demand_list if d > 1e-6)
                cur_load = min(1.0, total_line_d)
                remain_line = total_line_d
                print(f"0 | Depot Start | RemainLine={remain_line:.4f} | Load={cur_load:.4f}")
                for node in global_main_path:
                    if node == 0:
                        cur_load = min(1.0, remain_line)
                        print(f"{node} | Back Depot Reset | RemainLine={remain_line:.4f} | Load={cur_load:.4f}")
                    else:
                        d = demand_list[node]
                        cur_load -= d
                        if d > 1e-6:
                            remain_line -= d
                            tag = "Linehaul L"
                        else:
                            tag = "Backhaul B"
                        print(f"{node} | {tag}{node} | RemainLine={remain_line:.4f} | Load={cur_load:.4f}")
                print("===============================================================\n")

                # 恢复主线原始数据绘图
                self.env.selected_node_list = selected_nodes
                self.env.node_coords = node_coords
                self.env.node_demand = node_demand
                save_img = os.path.join(self.result_folder, "dvrs_dual_compare.png")
                draw_dvrs_route(
                    self.env,
                    batch_i=0,
                    main_path=global_main_path,
                    sym_path=global_sym_path,
                    save_path=save_img
                )

            return no_aug_dual_score.item(), aug_dual_score.item()