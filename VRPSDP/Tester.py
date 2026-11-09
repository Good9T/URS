import torch
import os
from logging import getLogger

from Env import Env
from Model import Model
from vrpsdp_route_plotter import draw_dvrs_vrpsdp_route
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

        total_sym_wins = 0
        total_instances = 0

        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)
            batch_score, batch_aug_score, sym_win, batch_total = self._test_one_batch(batch_size)
            score.update(batch_score)
            aug_score.update(batch_aug_score)
            episode += batch_size

            total_sym_wins += sym_win
            total_instances += batch_total

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score.avg, aug_score.avg))

            all_done = (episode == test_num_episode)
            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG MAIN VIEW SCORE: {:.4f} ".format(score.avg))
                self.logger.info(" AUG DUAL VIEW BEST SCORE: {:.4f} ".format(aug_score.avg))

                sym_win_rate = total_sym_wins / total_instances * 100
                self.logger.info(" SYMMETRIC VIEW WIN RATE: {:.2f}% ({}/{})".format(
                    sym_win_rate, total_sym_wins, total_instances))

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
            # 主视图原始无增强基线 (aug=0)
            no_aug_main_score = -max_pomo_reward_main[0, :].float().mean()

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
            # 新增：对称视图原始无增强基线 (aug=0，和主视图对齐逻辑)
            no_aug_sym_score = -max_pomo_reward_sym[0, :].float().mean()

            # 【修复1】双视图融合最优：逐样本取主/对称所有增强里最优
            combined_max_reward = torch.max(batch_reward, sym_batch_reward)
            aug_dual_score = -combined_max_reward.float().mean()

            # 【修复2】无增强双视图基准（如需可返回，当前return按需调整）
            no_aug_dual_score = -torch.max(max_pomo_reward_main[0, :], max_pomo_reward_sym[0, :]).float().mean()

            sym_win_mask = sym_batch_reward > batch_reward
            sym_win_count = sym_win_mask.sum().item()
            total_count = batch_size

            # ========== 单样本打印载重轨迹 + 双视图对比绘图（仅test_episodes=1时执行） ==========
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
                print(f"All Node [d,p] Demand: {demand_list}")
                # 新增打印原始无增强双视图基线分数，方便校验
                print(
                    f"NoAug Main Score: {no_aug_main_score.item():.4f} | NoAug Sym Score: {no_aug_sym_score.item():.4f}")

                # 打印【全局最优主线路径】载重&剩余送货库存轨迹
                print("\n==== Load & Remain Delivery Trace (Safe Max Capacity=1.0) ====")
                total_line_d = sum(d for d, p in demand_list if d > 1e-6)
                cur_load = min(self.env.safe_capacity[0].item(), total_line_d)
                remain_line = total_line_d
                print(f"0 | Depot Start | RemainDelivery={remain_line:.4f} | Load_d={cur_load:.4f}")
                for node in global_main_path:
                    if node == 0:
                        cur_load = min(self.env.safe_capacity[0].item(), remain_line)
                        print(f"{node} | Back Depot Reset | RemainDelivery={remain_line:.4f} | Load_d={cur_load:.4f}")
                    else:
                        d, p = demand_list[node]
                        cur_load -= d
                        if d > 1e-6:
                            remain_line -= d
                            tag = f"Delivery L{node}"
                        else:
                            tag = f"Pickup B{node}"
                        print(f"{node} | {tag} | RemainDelivery={remain_line:.4f} | Load_d={cur_load:.4f}")
                print("===============================================================\n")

                # 恢复主线原始坐标/路径，绘图传入两条全局aug最优路线
                self.env.selected_node_list = selected_nodes
                self.env.node_coords = node_coords
                self.env.node_demand = node_demand
                save_img = os.path.join(self.result_folder, "dvrs_dual_compare.png")
                draw_dvrs_vrpsdp_route(
                    self.env,
                    batch_i=0,
                    main_path=global_main_path,
                    sym_path=global_sym_path,
                    save_path=save_img
                )

            return no_aug_dual_score.item(), aug_dual_score.item(), sym_win_count, total_count