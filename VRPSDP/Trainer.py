import torch
from logging import getLogger

from Env import Env
from Model import Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils import *


class Trainer:
    def __init__(self, env_params, model_params, optimizer_params, trainer_params):
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            device = torch.device('cuda', cuda_device_num)
        else:
            device = torch.device('cpu')
        torch.set_default_device(device)

        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if not model_load['load_model_only']:
                self.start_epoch = 1 + model_load['epoch']
                self.result_log.set_raw_data(checkpoint['result_log'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.scheduler.last_epoch = model_load['epoch'] - 1
            self.logger.info('Saved Model Loaded !!')

        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        epochs = self.trainer_params['epochs']
        for epoch in range(self.start_epoch, epochs + 1):
            self.logger.info('=================================================================')
            train_score, train_loss_total, train_loss_main, train_loss_sym = self._train_one_epoch(epoch)
            self.scheduler.step()

            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss_total', epoch, train_loss_total)
            self.result_log.append('train_loss_main', epoch, train_loss_main)
            self.result_log.append('train_loss_sym', epoch, train_loss_sym)

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, epochs)
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, epochs, elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            if epoch > 1:
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                               self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                               self.result_log,
                                               labels=['train_loss_total', 'train_loss_main', 'train_loss_sym'])

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                               self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                               self.result_log,
                                               labels=['train_loss_total', 'train_loss_main', 'train_loss_sym'])

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):
        score_AM = AverageMeter()
        loss_total_AM = AverageMeter()
        loss_main_AM = AverageMeter()
        loss_sym_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, avg_loss_total, avg_loss_main, avg_loss_sym = self._train_one_batch(batch_size)

            score_AM.update(avg_score, batch_size)
            loss_total_AM.update(avg_loss_total, batch_size)
            loss_main_AM.update(avg_loss_main, batch_size)
            loss_sym_AM.update(avg_loss_sym, batch_size)

            episode += batch_size

            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 5:
                    self.logger.info(
                        'Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f}  TotalLoss: {:.4f}  MainLoss: {:.4f}  SymLoss: {:.4f}'
                        .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                score_AM.avg, loss_total_AM.avg, loss_main_AM.avg, loss_sym_AM.avg))

        self.logger.info(
            'Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f}  TotalLoss: {:.4f}  MainLoss: {:.4f}  SymLoss: {:.4f}'
            .format(epoch, 100. * episode / train_num_episode,
                    score_AM.avg, loss_total_AM.avg, loss_main_AM.avg, loss_sym_AM.avg))

        return score_AM.avg, loss_total_AM.avg, loss_main_AM.avg, loss_sym_AM.avg

    def _train_one_batch(self, batch_size):
        self.model.train()
        use_symmetry = True  # 切换开关 False=单视角 True=双视角

        # ========== Main View 主视角必跑 ==========
        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(batch_size, self.env.pomo_size, 0)
        state, reward, done = self.env.pre_step()

        while not done:
            selected, prob = self.model(state)
            prob = prob[:, :self.env.pomo_size]
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        if use_symmetry:
            # ========== 双视角分支 ==========
            self.env.load_symmetry_problems()
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

            sym_prob_list = torch.zeros(batch_size, self.env.pomo_size, 0)
            state, sym_reward, done = self.env.pre_step()

            while not done:
                selected, prob = self.model(state)
                prob = prob[:, :self.env.pomo_size]
                state, sym_reward, done = self.env.step(selected)
                sym_prob_list = torch.cat((sym_prob_list, prob[:, :, None]), dim=2)

            # 分开计算各自优势，不要拼接再切片
            main_adv = reward.float() - reward.float().mean(dim=1, keepdim=True)
            sym_adv = sym_reward.float() - sym_reward.float().mean(dim=1, keepdim=True)

            log_prob = prob_list.log().sum(dim=2)
            sym_log_prob = sym_prob_list.log().sum(dim=2)

            # 各自维度匹配相乘，不再共用adv导致5/15冲突
            loss_main = -(main_adv * log_prob).mean()
            loss_sym = -(sym_adv * sym_log_prob).mean()
            loss_total = (loss_main + loss_sym) / 2

            # best reward 依旧取两者拼接最大值
            combined_reward = torch.cat([reward, sym_reward], dim=1)
            best_reward, _ = combined_reward.max(dim=1)
            score = -best_reward.float().mean()

            self.model.zero_grad()
            loss_total.backward()
            self.optimizer.step()
            # 双视角直接返回
            return score.item(), loss_total.item(), loss_main.item(), loss_sym.item()
        else:
            # ========== 单视角分支 ==========
            adv = reward.float() - reward.float().mean(dim=1, keepdim=True)
            log_prob = prob_list.log().sum(dim=2)
            loss_main = -(adv * log_prob).mean()
            loss_total = loss_main

            best_reward, _ = reward.max(dim=1)
            score = -best_reward.float().mean()

            self.model.zero_grad()
            loss_total.backward()
            self.optimizer.step()
            # 单视角内部直接return，loss_sym给0.0浮点数
            return score.item(), loss_total.item(), loss_main.item(), 0.0
