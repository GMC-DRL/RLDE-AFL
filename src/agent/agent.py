'''
agent.py 名字待定
'''

import torch
import numpy as np
from torch import nn
from agent.networks import MLP, Actor, Critic
from agent.utils import Memory, Memory_gleet
from torch.nn import functional as F
from torch.distributions import Normal
from Feature_Extractor.feature_extractor import Feature_Extractor, Gleet_FE
from agent.utils import save_class
from agent.utils import clip_grad_norms
from utils import log_to_tb_train, log_to_tb_operator, log_gen_operator
class agent:
    def __init__(self, config):
        '''
        一开始创建agent对象， 得到训练数据，train_set
        然后对每一个问题进行训练， 每次每个问题每个episode初始化env
        env为种群，应该包括env.step env.reset,

        agent 中应该包括动作维度， 种群编码器FE，
        状态动作网络Actor，    得到选择哪个动作 以及动作的log_prob
        状态参数网络configurationNetwork 得到参数配置，和动作结合起来可以更新种群

        主要的函数是train_episode()
        表示agent与环境交互一轮 然后保存轨迹 使用PPO更新网络参数
        '''
        self.__config = config
        self.__device = self.__config.device
        # FE

        if self.__config.fe_gleet:
            self.Fe = Gleet_FE(hidden_dim = self.__config.fe_hidden_dim, device = self.__config.device)
        else:
            self.Fe = Feature_Extractor(hidden_dim = self.__config.fe_hidden_dim,
                                        n_layers = self.__config.fe_n_layers,
                                        is_mlp = self.__config.mlp,
                                        device = self.__config.device)

        # Actor
        self.Actor = Actor(input_dim = self.__config.fe_hidden_dim,
                           mu_operator = self.__config.mutation_selector.n_operator,
                           cr_operator = self.__config.crossover_selector.n_operator,
                           n_mutation = self.__config.mutation_selector.n_mutation,
                           n_crossover = self.__config.crossover_selector.n_crossover,
                           device = self.__device
                           )

        # Critic
        self.Critic = Critic(input_dim = self.__config.fe_hidden_dim, device=self.__device)
        self.lr = {
            'fe': self.__config.fe_lr,
            'actor': self.__config.actor_lr,
            'critic': self.__config.critic_lr
        }

        self.is_test = False
        self.optimizer = torch.optim.Adam(
            [{'params': self.Fe.parameters(), 'lr': self.lr['fe']}] +
            [{'params': self.Actor.parameters(), 'lr': self.lr['actor']}] +
            [{'params': self.Critic.parameters(), 'lr': self.lr['critic']}]
        )
        self.__lr_decay = self.__config.lr_decay
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, self.__lr_decay, last_epoch = -1,)

        # init learning time
        self.__learning_time = 0

        self.__cur_checkpoint = 0
        # save init agent
        # sample /agent_save_dir/checkpoint5

        if self.__cur_checkpoint == 0 and not self.__config.no_saving:
            save_class(self.__config.save_dir, 'checkpoint' + str(self.__cur_checkpoint), self)
            self.__cur_checkpoint += 1

        self.epoch = 0 # 外面的轮数
        self.update_step = 0

        # self.Fe.to(self.__device)
        self.Actor.to(self.__device)
        self.Critic.to(self.__device)


    def get_selector(self):
        return self.__config.de_mutation_op, self.__config.mutation_selector, \
               self.__config.crossover_op,self.__config.crossover_selector

    def update_setting(self):
        pass

    def get_parameter_number(self):
        print(self.Fe.get_parameter_number())
        print(self.Actor.get_parameter_number())
        print(self.Critic.get_parameter_number())

    def train(self):
        torch.set_grad_enabled(True)
        # self.Fe.set_on_train()
        self.Fe.set_off_train()
        if self.__config.fe_train:
            self.Fe.set_on_train()
        self.Actor.train()
        self.Critic.train()

    def train_episode(self, env, tb_logger = None):
        if self.__config.fe_gleet:
            memory = Memory_gleet()
        else:
            memory = Memory()
        state = env.reset()  # state is x and y
        # x must be 3D: 1 * n * dim , 1 is tick
        gamma = self.__config.ppo_gamma
        n_step = self.__config.ppo_n_step
        K_epochs = self.__config.ppo_k_epochs
        eps_clip = self.__config.ppo_eps

        t = 0
        is_done = False

        episode_return = 0
        # sample trajectory
        while not is_done:
            t_s = t
            total_cost = 0
            entropy = []
            bl_val_detached = []
            bl_val = []
            # accumulate trainsition
            while t - t_s < n_step:
                memory.states_append(state)
                feature = self.Fe(state).squeeze(0).to(self.__device)  # 2D : [pop_size, hidden_dim]

                action, log_lh, entro_p = self.Actor(feature, require_entropy = True)
                # try:
                #     action, log_lh, entro_p = self.Actor(feature, require_entropy = True)
                # except ValueError as e:
                #     print(feature)
                #     print(env.problem.__str__())
                #     print(state['y']) 探索不够 落到一起了
                #     print(self.__learning_time)

                memory.actions.append(action.clone())
                memory.logprobs.append(log_lh)
                action = action.cpu().numpy()

                entropy.append(entro_p.detach().cpu())

                baseline_val_detached, baseline_val = self.Critic(feature[None, :])
                # 这里主要是bs得是多个环境，所以一开始取了[0] 在这里又补回来
                bl_val_detached.append(baseline_val_detached)
                bl_val.append(baseline_val)

                # env step
                # next_state, rewards, is_end, info = env.step(action)
                next_state, rewards, is_end = env.step(action)
                # tensorboard
                self.update_step += 1
                if not self.__config.no_tb:
                    log_to_tb_operator(tb_logger, self.__config.de_mutation_op, self.__config.crossover_op, action[:, 0],
                                       action[:, 1], self.update_step)

                # todo 这里 is_end 只是一个环境的，如果要多个环境， is_end : train_batch_size
                # self.log_g += 1
                #
                # if self.log_g % 3 == 0:
                #     self.log_num += 1
                #     file_path = self.__config.save_dir + '/Epoch' + str(self.epoch) + '/'
                #     save_class(file_path, 'gen' + str(self.log_g) + '_' + str(self.log_num), self)
                memory.rewards.append(torch.tensor(rewards, dtype = torch.float).to(self.__device))

                t = t + 1
                state = next_state
                episode_return += rewards
                if is_end:
                    is_done = True
                    break

            # store info
            t_time = t - t_s
            total_cost = total_cost / t_time

            # begin ppo update

            old_actions = torch.stack(memory.actions)  # 3D： t_time  * ps * (action_dim)
            old_states = memory.return_states()  # todo 3D : t_time * ps * dim_f
            old_logprobs = torch.stack(memory.logprobs).detach().view(-1)

            # Optimize PPO policy for K mini-epochs:
            old_value = None

            # last time state to feature

            for _k in range(K_epochs):
                if _k == 0:
                    logprobs = memory.logprobs
                else:
                    # Evaluating old actions and values:
                    logprobs = []
                    entropy = []
                    bl_val_detached = []
                    bl_val = []
                    old_features = self.Fe(old_states).to(self.__device)  # todo 3D: t_time * ps * hidden_dim turn bs
                    for tt in range(t_time):
                        # get new action_prob
                        _, log_p, entro_p = self.Actor(old_features[tt],
                                                       fixed_action = old_actions[tt],
                                                       require_entropy = True
                                                       )

                        logprobs.append(log_p)
                        entropy.append(entro_p.detach().cpu())

                        baseline_val_detached, baseline_val = self.Critic(old_features[tt][None, :])

                        bl_val_detached.append(baseline_val_detached)
                        bl_val.append(baseline_val)

                logprobs = torch.stack(logprobs).view(-1)
                entropy = torch.stack(entropy).view(-1)
                bl_val_detached = torch.stack(bl_val_detached).view(-1)
                bl_val = torch.stack(bl_val).view(-1)

                # get target value for critic
                Reward = []
                reward_reversed = memory.rewards[::-1]
                # get next value
                feature = self.Fe(state)  # todo 3D: 1 * ps * hidden_dim
                R = self.Critic(feature)[0] # [bs, 1]

                critic_output = R.clone()
                for r in range(len(reward_reversed)):
                    R = R * gamma + reward_reversed[r]
                    Reward.append(R)

                # clip the target
                Reward = torch.stack(Reward[::-1], 0)
                Reward = Reward.view(-1)

                # Finding the ratio (pi_theta / pi_theta_old):
                ratios = torch.exp(logprobs - old_logprobs.detach())

                # Finding Surrogate loss:
                advantages = Reward - bl_val_detached

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages

                # figure out first loss
                reinforce_loss = -torch.min(surr1, surr2).mean()

                # define baseline loss
                if old_value is None:
                    baseline_loss = ((bl_val - Reward) ** 2).mean()
                    old_value = bl_val.detach()
                else:
                    vpredclipped = old_value + torch.clamp(bl_val - old_value, -eps_clip, eps_clip)
                    v_max = torch.max(((bl_val - Reward) ** 2), ((vpredclipped - Reward) ** 2))
                    baseline_loss = v_max.mean()

                # todo 这里rlepso没有用，问下这是什么
                # check K-L divergence (for logging only)
                approx_kl_diergence = (.5 * (old_logprobs.detach() - logprobs) ** 2).mean().detach()
                approx_kl_diergence[torch.isinf(approx_kl_diergence)] = 0

                # calculate loss
                loss = baseline_loss + reinforce_loss

                # update gradient step

                self.optimizer.zero_grad()
                loss.backward()

                self.__learning_time += 1
                grad_norms = clip_grad_norms(self.optimizer.param_groups, 1) # 改成1
                # save training log

                # perform gradient descent
                self.optimizer.step()

                if self.__learning_time >= (self.__config.save_interval * self.__cur_checkpoint) and\
                    not self.__config.no_saving:
                    save_class(self.__config.save_dir, 'checkpoint' + str(self.__cur_checkpoint), self)
                    self.__cur_checkpoint += 1

                if self.__learning_time % self.__config.log_step == 0:
                    log_to_tb_train(tb_logger, self, episode_return, Reward, critic_output, ratios,
                                    bl_val_detached, total_cost, grad_norms,
                                    memory.rewards, entropy, approx_kl_diergence,
                                    reinforce_loss, baseline_loss, logprobs, self.__config.show_figs, self.__learning_time)


                if self.__learning_time >= self.__config.max_learning_step:
                    return True,    {'normalizer': env.log_cost[0],
                                    'gbest': env.log_cost[-1],
                                    'return': episode_return,
                                    'learn_steps': self.__learning_time}

            memory.clear_memory()
        pbar_info = {
            'normalizer': env.log_cost[0],
            'gbest': env.log_cost[-1],
            'return': episode_return,
            'learn_steps': self.__learning_time
        }
        return self.__learning_time >= self.__config.max_learning_step, pbar_info


    def rollout_episode(self, env):
        #
        is_done = False
        state = env.reset()
        R = 0

        mutation_op = {}
        crossover_op = {}

        for m_op in self.__config.de_mutation_op:
            mutation_op[m_op] = []
        for c_op in self.__config.crossover_op:
            crossover_op[c_op] = []

        m_mu_gen = [] # mutation参数的每一代 种群的参数 一个元素为[pop_size, n_mutation_op]
        m_sigma_gen = []

        c_mu_gen = []
        c_sigma_gen = []
        while not is_done:
            feature = self.Fe(state).squeeze(0).to(self.__device)
            Actor_output = self.Actor.get_action_sample(feature)
            action = Actor_output[0].detach().cpu().numpy()
            log_gen_operator(mutation_op, crossover_op, self.__config.de_mutation_op, self.__config.crossover_op, action[:, 0],
                                       action[:, 1])
            m_mu_gen.append(Actor_output[1].detach().cpu().numpy())
            m_sigma_gen.append(Actor_output[2].detach().cpu().numpy())
            c_mu_gen.append(Actor_output[3].detach().cpu().numpy())
            c_sigma_gen.append(Actor_output[4].detach().cpu().numpy())
            # action = self.Actor(feature)[0].cpu().numpy()
            state, reward, is_done = env.step(action)
            R += reward
        # return cost and fes R
        # 此时 dict 每个算子 每代的次数
        return {'cost': env.log_cost,
                'fes': env.fes,
                'return': R,
                'Logging': (mutation_op, crossover_op, m_mu_gen, m_sigma_gen, c_mu_gen, c_sigma_gen)}



## Test
if __name__ == '__main__':
    import numpy as np
    Fe = Feature_Extractor(is_train = True)
    # x = np.random.rand(3, 4)
    # y = np.random.rand(3)
    # state = Fe(x, y).squeeze(0)
    policy_net = Actor(16, 14, 5)
    # print(state)
    # action, log_prob = policy_net.forwards(state)
    # print(log_prob)
    # _, log_probs = policy_net.forwards(state, fixed_action = action)
    # print(log_probs)

    x1 = np.random.rand(3, 4)
    y1 = np.random.rand(3)

    x2 = np.random.rand(3, 4)
    y2 = np.random.rand(3)









