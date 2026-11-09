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
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch'] - 1
            self.logger.info('Saved Model Loaded!')

        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        epochs = self.trainer_params['epochs']
        for epoch in range(self.start_epoch, epochs + 1):
            self.logger.info('=================================================================')
            train_score, train_loss, loss1, loss2 = self._train_one_epoch(epoch)
            self.scheduler.step()

            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)
            self.result_log.append('loss1',  epoch, loss1)
            self.result_log.append('loss2', epoch, loss2)

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
                                               labels=['train_loss'])

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
                                               labels=['train_loss'])

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):
        score_AM = AverageMeter()
        loss_AM = AverageMeter()
        loss_1_AM = AverageMeter()
        loss_2_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, avg_loss_total, avg_loss1, avg_loss2 = self._train_one_batch(batch_size)

            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss_total, batch_size)
            loss_1_AM.update(avg_loss1, batch_size)
            loss_2_AM.update(avg_loss2, batch_size)

            episode += batch_size

            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 5:
                    self.logger.info(
                        'Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f}  Loss: {:.4f}'
                        .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                score_AM.avg, loss_AM.avg))

        self.logger.info(
            'Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f}  Loss: {:.4f}'
            .format(epoch, 100. * episode / train_num_episode,
                    score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg, loss_1_AM.avg, loss_2_AM.avg

    def _train_one_batch(self, batch_size):
        self.model.train()

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

        pomo_main = self.env.pomo_size

        self.env.load_sym()
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)
        prob_list_sym = torch.zeros(batch_size, self.env.pomo_size, 0)
        state_sym, reward_sym, done_sym = self.env.pre_step()
        while not done_sym:
            selected_sym, prob_sym = self.model(state_sym)
            prob_sym = prob_sym[:, :self.env.pomo_size]
            state_sym, reward_sym, done_sym = self.env.step(selected_sym)
            prob_list_sym = torch.cat((prob_list_sym, prob_sym[:, :, None]), dim=2)

        pomo_sym = self.env.pomo_size

        combined_reward = torch.cat([reward, reward_sym], dim=1)
        global_mean = combined_reward.float().mean(dim=1, keepdim=True)
        adv = reward.float() - global_mean
        adv_sym = reward_sym.float() - global_mean

        log_prob = prob_list.log().sum(dim=2)
        log_prob_sym = prob_list_sym.log().sum(dim=2)

        loss = -(adv * log_prob).mean()
        loss_sym = -(adv_sym * log_prob_sym).mean()
        loss_total = (loss * pomo_main + loss_sym * pomo_sym)/(pomo_main + pomo_sym)

        best_reward, _ = combined_reward.max(dim=1)
        score = -best_reward.float().mean()

        self.model.zero_grad()
        loss_total.backward()
        self.optimizer.step()
        return score.item(), loss_total.item(), loss.item(), loss_sym.item()

