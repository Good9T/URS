import torch
import os
from logging import getLogger

from Env import Env
from Model import Model
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
            device = torch.device('cpu')
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
        dual_score = AverageMeter()
        full_score = AverageMeter()
        test_num_episode = self.tester_params['test_episodes']
        episode = 0
        while episode < test_num_episode:
            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)
            batch_score, batch_dual_score, batch_full_score = self._test_one_batch(batch_size)
            score.update(batch_score)
            dual_score.update(batch_dual_score)
            full_score.update(batch_full_score)
            episode += batch_size

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info(
                f"episode {episode:3d}/{test_num_episode:3d}, "
                f"Elapsed[{elapsed_time_str}], "
                f"Remain[{remain_time_str}], "                
                f"score:{score.avg:.3f}, "
                f"dual_score:{dual_score.avg:.3f}, "
                f"full_score:{full_score.avg:.3f}, "
            )

            all_done = (episode == test_num_episode)
            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" SCORE: {:.4f} ".format(score.avg))
                self.logger.info(" DUAL SCORE: {:.4f} ".format(dual_score.avg))
                self.logger.info(" FULL SCORE: {:.4f} ".format(full_score.avg))

    def _test_one_batch(self, batch_size):
        aug_enable = self.tester_params['aug_enable']
        aug_factor = self.tester_params['aug_factor'] if aug_enable else 1

        self.model.eval()
        with torch.no_grad():

            self.env.load_problems(batch_size, aug_enable=aug_enable, aug_factor=aug_factor)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)
            state, reward, done = self.env.pre_step()
            while not done:
                selected, _ = self.model(state)
                state, reward, done = self.env.step(selected)

            aug_reward_main = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
            max_pomo_reward_main, _ = aug_reward_main.max(dim=2)
            aug_score_main, _ = max_pomo_reward_main.max(dim=0)
            score_main = max_pomo_reward_main[0, :]

            self.env.load_sym()
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)
            state_sym, reward_sym, done_sym = self.env.pre_step()
            while not done_sym:
                selected_sym, _ = self.model(state_sym)
                state_sym, reward_sym, done_sym = self.env.step(selected_sym)

            aug_reward_sym = reward_sym.reshape(aug_factor, batch_size, self.env.pomo_size)
            max_pomo_reward_sym, _ = aug_reward_sym.max(dim=2)
            aug_score_sym, _ = max_pomo_reward_sym.max(dim=0)
            score_sym = max_pomo_reward_sym[0, :]

            score = -score_main.float().mean()

            dual_reward = torch.max(score_main, score_sym)
            dual_score = -dual_reward.float().mean()

            full_reward = torch.max(aug_score_main, aug_score_sym)
            full_score = -full_reward.float().mean()

            return score.item(), dual_score.item(), full_score.item()