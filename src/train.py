import warnings
from problem import bbob
import numpy as np
from numpy import random
import matplotlib.pyplot as plt

from Population import Population

from utils import *
from Feature_Extractor import feature_extractor,attention_block

from Environment.env import Env

from agent.agent import agent
from agent.utils import save_class
from Mutation_Selection import *
from Mutation_Selection.Select import select_mutation as Select_Mutation
from Mutation_Selection import basic_mutation

from CrossOver_Selection import *
from CrossOver_Selection.Select import select_crossover as Select_Crossover
from CrossOver_Selection.simple_crossover import basic_crossover

from Selection_part.selection_part import greedy_select
from tqdm import tqdm
from test import rollout_experiment
import torch
import multiprocessing
def train(config, tb_logger = None):
    print("begin training")
    warnings.filterwarnings("ignore")
    # multiprocessing.set_start_method('spawn')
    # torch.set_num_threads(1)
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # figure out Mutation and Crossover selector
    config.mutation_selector = Select_Mutation(config.de_mutation_op)
    config.crossover_selector = Select_Crossover(config.crossover_op)
    train_set, _ = construct_problem(config, config.trainset_seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    Agent = agent(config)

    # move optimizer's data onto device
    for state in Agent.optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(config.device)

    Agent.get_parameter_number()
    # begin
    exceed_max_ls = False
    epoch = 0
    cost_record = {} # 记录各个问题的最好cost
    normalizer_record = {} # 记录各个问题的最好归一化
    epoch_record = {}
    return_record = []
    learning_steps = []
    epoch_steps = []
    for problem in train_set:
        name = f"{problem.__str__()}_{problem.dim}" if config.mix_dim else problem.__str__()
        cost_record[name] = []
        normalizer_record[name] = []
        epoch_record[name] = []
    while not exceed_max_ls:
        if epoch > config.max_epoch:
            break
        learn_step = 0
        train_set.shuffle() # 打乱训练集合下标 随机选一个问题
        Agent.train()
        # Agent.lr_scheduler.step(epoch) # 调整学习率
        with tqdm(range(train_set.N), desc = f"Training Agent! Epoch {epoch}") as pbar:
            # One epoch
            for problem_id, problem in enumerate(train_set):
                # One episode
                # todo 这里的env 可以是list 但是batch_size = 1 所以只是一个env
                problem.reset()
                env = Env(problem, config) # todo 修改一下Env的init
                exceed_max_ls, pbar_info_train = Agent.train_episode(env, tb_logger)
                pbar.set_postfix(pbar_info_train)
                pbar.update(1)
                name = f"{problem.__str__()}_{problem.dim}" if config.mix_dim else problem.__str__()
                learn_step_episode = pbar_info_train['learn_steps']
                cost_record[name].append(pbar_info_train['gbest'])
                normalizer_record[name].append(pbar_info_train['normalizer'])
                return_record.append(pbar_info_train['return'])
                learning_steps.append(learn_step_episode)

                learn_step += learn_step_episode

                # 记录一下problem训练的时候找到最小
                epoch_record[name] = pbar_info_train['gbest']
                if exceed_max_ls:
                    break
            epoch_steps.append(learn_step)

            if not config.no_saving:
                save_log(config, train_set, epoch_steps, learning_steps, cost_record, return_record, normalizer_record)

            epoch += 1
            Agent.epoch += 1

            # 存agent
            file_path = config.save_dir + 'Epoch/'
            if not config.no_save_epoch:
                save_class(file_path, 'epoch' + str(epoch), Agent)

            if not config.no_tb:
                log_to_tb_epoch(tb_logger, epoch_record, epoch)
            # rollout
            if not config.no_rollout:
                random_state = np.random.get_state()
                if epoch % config.rollout_interval == 0:
                    rollout_experiment(config, file_path + 'epoch' + str(epoch), int(epoch / config.rollout_interval), tb_logger)
                np.random.set_state(random_state)

            # 存agent
            # file_path = config.save_dir + 'Epoch/'
            # save_class(file_path, 'epoch' + str(epoch), Agent)
            # if epoch % 3 == 0:
            #     rollout(config, file_path + 'epoch' + str(epoch), epoch)
            # todo 画图
            # if epoch % config.draw_interval == 0:
            #     # 画图
            #     draw(env)
    
    
def draw(env):
    """
    Draw a line plot of average fitness and best fitness over generations.
    Parameters:
    env (object): The environment object containing fitness data.
    Returns:
    None
    """
    
    plt.figure(figsize=(10, 5))
    plt.plot(env.avg_fitness_list, label='Average Fitness')
    plt.plot(env.best_fitness_list, label='Best Fitness')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Fitness over Generations')
    plt.legend()
    plt.show()

#
if __name__ == '__main__':
    train()