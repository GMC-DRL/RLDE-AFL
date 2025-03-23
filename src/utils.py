from problem import bbob
import numpy as np
from numpy import random
import os
from Population import Population
import torch
from plot import plot_grad_flow

from tensorboardX import SummaryWriter
# def construct_problem(problem, dim, upperbound, train_batch_size, test_batch_size, difficulty, instance_seed = 3849):
#     if problem in ['bbob', 'bbob-noisy']:
#         return bbob.BBOB_Dataset.get_datasets(suit = problem,
#                                               dim = dim,
#                                               upperbound = upperbound,
#                                               train_batch_size = train_batch_size,
#                                               test_batch_size = test_batch_size,
#                                               difficulty = difficulty,
#                                               instance_seed = instance_seed)

def construct_problem(config, seed):
    if config.problem in ['bbob', 'bbob-noisy']:
        return bbob.BBOB_Dataset.get_datasets(suit = config.problem,
                                              dim = config.dim,
                                              upperbound = config.upperbound,
                                              train_batch_size = config.train_batch_size,
                                              test_batch_size = config.test_batch_size,
                                              difficulty = config.difficulty,
                                              instance_seed = seed,
                                              mix_dim = config.mix_dim,
                                              test_all = config.test_all)
    else:
        raise ValueError(config.problem + ' is not defined!')

# a test for the initialization of the population

def save_log(config, train_set, epochs, steps, cost, returns, normalizer):
    if config.no_saving:
        return
    log_dir = config.log_dir + f'/train/{config.run_time}/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return_save = np.stack((steps, returns), 0)

    np.save(log_dir + 'return', return_save)
    for problem in train_set:
        name = f"{problem.__str__()}_{problem.dim}" if config.mix_dim else problem.__str__()
        if len(cost[name]) == 0:
            continue
        while len(cost[name]) < len(epochs):
            cost[name].append(cost[name][-1]) # 一直填充
            normalizer[name].append(normalizer[name][-1])
        cost_save = np.stack((epochs, cost[name], normalizer[name]), 0)
        np.save(log_dir + name + '_cost', cost_save)

# tb_logger
def log_to_tb_train(tb_logger, agent, Reward, R, Critic_out, ratios, bl_val_detached, total_cost, grad_norms, reward,
                    entropy, approx_kl_divergence, reinforce_loss, baseline_loss, logprobs, show_figs, mini_step):

    tb_logger.add_scalar('learnrate/fe', agent.optimizer.param_groups[0]['lr'], mini_step)
    tb_logger.add_scalar('learnrate/actor', agent.optimizer.param_groups[1]['lr'], mini_step)
    tb_logger.add_scalar('learnrate/critic', agent.optimizer.param_groups[2]['lr'], mini_step)

    tb_logger.add_scalar('train/episode_Return', Reward.item(), mini_step) # episode 的回报
    tb_logger.add_scalar('train/Target_Return_changed', R.mean().item(), mini_step) # 学习率的

    tb_logger.add_scalar('train/Critic_output', Critic_out.mean().item(), mini_step)
    tb_logger.add_scalar('train/ratios', ratios.mean().item(), mini_step)

    avg_reward = torch.stack(reward).mean().item()
    max_reward = torch.stack(reward).max().item()
    grad_norms, grad_norms_clipped = grad_norms

    tb_logger.add_scalar('train/avg_reward', avg_reward, mini_step)
    tb_logger.add_scalar('train/max_reward', max_reward, mini_step)
    tb_logger.add_scalar('loss/actor_loss', reinforce_loss.item(), mini_step)
    tb_logger.add_scalar('loss/-logprobs', -logprobs.mean().item(), mini_step)
    tb_logger.add_scalar('train/entropy', entropy.mean().item(), mini_step)
    tb_logger.add_scalar('train/approx_kl_divergence', approx_kl_divergence.item(), mini_step)
    tb_logger.add_histogram('train/bl_val', bl_val_detached.cpu(), mini_step)
    tb_logger.add_scalar('train/total_cost', total_cost, mini_step)

    tb_logger.add_scalar('grad/fe', grad_norms[0], mini_step)
    tb_logger.add_scalar('grad_clipped/fe', grad_norms_clipped[0], mini_step)
    tb_logger.add_scalar('grad/actor', grad_norms[1], mini_step)
    tb_logger.add_scalar('grad_clipped/actor', grad_norms_clipped[1], mini_step)
    tb_logger.add_scalar('grad/critic', grad_norms[2], mini_step)
    tb_logger.add_scalar('grad_clipped/critic', grad_norms_clipped[2], mini_step)

    tb_logger.add_scalar('loss/critic_loss', baseline_loss.item(), mini_step)
    tb_logger.add_scalar('loss/total_loss', (reinforce_loss + baseline_loss).item(), mini_step)

    if mini_step % 1000 == 0 and show_figs:
        tb_logger.add_images('grad/fe', [plot_grad_flow(agent.Fe)], mini_step)
        tb_logger.add_images('grad/actor', [plot_grad_flow(agent.Actor)], mini_step)
        tb_logger.add_images('grad/critic', [plot_grad_flow(agent.Critic)], mini_step)

def log_to_tb_rollout(tb_logger, problem, gbest, fes, R, T0, T1, T2, mini_step):

    tb_logger.add_scalar('rollout_T0/' + problem, T0, mini_step)
    tb_logger.add_scalar('rollout_T1/' + problem, T1, mini_step)
    tb_logger.add_scalar('rollout_T2/' + problem, T2, mini_step)

    tb_logger.add_scalar('rollout_cost/' + problem, gbest, mini_step)
    tb_logger.add_scalar('rollout_fes/' + problem, fes, mini_step)
    tb_logger.add_scalar('rollout_return/' + problem, R, mini_step)
    # problems = rollout_result['cost'].keys()
    #
    # tb_logger.add_scalar('rollout/T0', rollout_result['T0'], epoch)
    # tb_logger.add_scalar('rollout/T1', rollout_result['T1'], epoch)
    # tb_logger.add_scalar('rollout/T2', rollout_result['T2'], epoch)
    #
    # for problem in problems:
    #     tb_logger.add_scalar('rollout_cost/' + problem, rollout_result['cost'][problem], epoch)
    #     tb_logger.add_scalar('rollout_fes/' + problem, rollout_result['fes'][problem], epoch)
    #     tb_logger.add_scalar('rollout_return/' + problem, rollout_result['return'][problem], epoch)

def log_to_tb_operator(tb_logger, mutation_op, crossover_op, mutation_action, crossover_action, mini_step):
    # 存储

    for i, m_op in enumerate(mutation_op):
        m_cnt = len(np.where(mutation_action == i)[0])
        tb_logger.add_scalar('mutation/' + m_op, m_cnt, mini_step)

    for i, c_op in enumerate(crossover_op):
        c_cnt = len(np.where(crossover_action == i)[0])
        tb_logger.add_scalar('crossover/' + c_op, c_cnt, mini_step)

def log_to_tb_epoch(tb_logger, epoch_record, mini_step):
    for name in epoch_record.keys():
        tb_logger.add_scalar('Find_best/' + name, epoch_record[name], mini_step)

def log_to_tb_rollout_experiment(tb_logger, results, mini_step):
    cost = results['cost']
    problem_name = cost.keys()
    for problem in problem_name:
        problem_cost = np.stack(cost[problem])[:, -1]
        tb_logger.add_scalar('rollout_avgcost/' + problem, np.mean(problem_cost), mini_step)
        tb_logger.add_scalar('rollout_gbest/' + problem, np.min(problem_cost), mini_step)
        tb_logger.add_scalar('rollout_avgfes/' + problem, np.mean(results['fes'][problem]), mini_step)
        tb_logger.add_scalar('rollout_minfes/' + problem, np.min(results['fes'][problem]), mini_step)
        tb_logger.add_scalar('rollout_avgreturn/' + problem, np.mean(results['return'][problem]), mini_step)
        tb_logger.add_scalar('rollout_maxreturn/' + problem, np.max(results['return'][problem]), mini_step)
    tb_logger.add_scalar('rollout_time/T0', results['T0'], mini_step)
    tb_logger.add_scalar('rollout_time/T1', results['T1'], mini_step)
    tb_logger.add_scalar('rollout_time/T2', results['T2'], mini_step)

def log_gen_operator(mutation_dict, crossover_dict, mutation_op, crossover_op, mutation_action, crossover_action):
    for i, m_op in enumerate(mutation_op):
        m_cnt = len(np.where(mutation_action == i)[0])
        mutation_dict[m_op].append(m_cnt)

    for i, c_op in enumerate(crossover_op):
        c_cnt = len(np.where(crossover_action == i)[0])
        crossover_dict[c_op].append(c_cnt)

if __name__ == "__main__":
    problem = 'bbob'
    dim = 10
    upperbound = 5
    train_batch_size = 1
    test_batch_size = 1
    difficulty = 'easy'
    train_set, test_set = construct_problem(problem, dim, upperbound, train_batch_size, test_batch_size, difficulty, instance_seed = 0)
    f0 = train_set.data[0]
    print("hello world")
    # print(f0) 
    f0.reset()
    population = Population(pop_size = 50,dim = 10, max_G = 1000, max_fes = 300000, problem = f0)
    # print('population_x', population.current_vector)
    # print('population_y', population.current_fitness)
    

    