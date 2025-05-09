import multiprocessing
import os.path
import pickle
import time
import types

from numpy import random
from tqdm import tqdm

from Environment.env import Env
from agent.agent import agent as Agent
from agent.networks import Actor
from utils import *


# from CrossOver_Selection import *
# from CrossOver_Selection.Select import select_crossover
def cal_T0(dim, fes):
    T0 = 0
    for i in range(10):
        start = time.perf_counter()
        for _ in range(fes):
            x = np.random.rand(dim)
            x / (x + 2)
            x * x
            np.sqrt(x)
            np.log(x)
            np.exp(x)
        end = time.perf_counter()
        T0 += (end - start) * 1000 # ms
    return T0 / 10

def test(config):
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_name = config.agent_model
    model_load_dir = config.agent_load_dir
    # load user`s agent
    if model_name is not None:
        file_path = model_load_dir + model_name + '.pkl'
        with open(file_path, 'rb') as f:
            agent = pickle.load(f)

    config.de_mutation_op, config.mutation_selector,\
    config.crossover_op, config.crossover_selector = agent.get_selector()
    _, test_set = construct_problem(config, config.testset_seed)
    log_dir = config.test_log_dir

    # agent.Actor.get_action = types.MethodType(Actor.get_action, agent.Actor)
    # agent.Actor.get_action_sample = types.MethodType(Actor.get_action_sample, agent.Actor)
    # agent.rollout_episode = types.MethodType(Agent.rollout_episode, agent)
    test_results = {'cost':{},
                    'fes':{},
                    'return':{},
                    'Logging':{},
                    'T0':0.,
                    'T1':0.,
                    'T2':0.}

    for problem in test_set:
        name = f"{problem.__str__()}_{problem.dim}" if config.mix_dim else problem.__str__()
        test_results['cost'][name] = []
        test_results['fes'][name] = []
        test_results['return'][name] = []
        test_results['Logging'][name] = []

    print("begin testing")
    T0 = cal_T0(config.dim, config.max_fes)
    test_results['T0'] = T0

    print(multiprocessing.cpu_count())

    n_process = 3
    if multiprocessing.get_start_method(allow_none = True) is None:
        multiprocessing.set_start_method("spawn")
    with multiprocessing.Pool(processes = int(n_process)) as pool:
        tasks = [(problem, config, agent, config.test_run) for problem in test_set]
        results = list(
            tqdm(pool.imap(single_test_wrapper, tasks), total = test_set.N, desc = "Testing!")
        )

    for problem_name, result in results:
        test_results['cost'][problem_name] = result['cost']
        test_results['fes'][problem_name] = result['fes']
        test_results['return'][problem_name] = result['return']
        test_results['Logging'][problem_name] = result['Logging']
        test_results['T1'] += result['T1']
        test_results['T2'] += result['T2']

    test_results['T1'] /= test_set.N
    test_results['T2'] /= test_set.N
    if not config.no_saving:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        with open(log_dir + 'test.pkl', 'wb') as f:
            pickle.dump(test_results, f, -1)

def rollout_worker(problem, config, agent, seed, i):
    T1 = 0  # low-optimizer层的时间
    T2 = 0  # meta-agent层的时间
    results = {'cost': [], 'fes': [], 'return': []}
    name = f"{problem.__str__()}_{problem.dim}" if config.mix_dim else problem.__str__()
    for run in range(51):
        start = time.perf_counter()
        np.random.seed(seed[run])

        problem.reset()
        env = Env(problem, config)
        info = agent.rollout_episode(env)
        cost = info['cost']
        while len(cost) < 51:
            cost.append(cost[-1])
        fes = info['fes']
        R = info['return']
        end = time.perf_counter()

        if i == 0:
            T1 += env.problem.T1    # todo 还得去看env
            T2 += (end - start) * 1000  # ms
        results['cost'].append(cost)
        results['fes'].append(fes)
        results['return'].append(R)
    
    if i == 0:
        T1 /= 51
        T2 /= 51
    else:
        T1 = None
        T2 = None
    
    return name, results, T1, T2

# def rollout(config, file_path, rollout_epoch, tb_logger = None):
#     torch.set_grad_enabled(False)
#     with open(file_path + '.pkl', 'rb') as f:
#         agent = pickle.load(f)
#     _, test_set = construct_problem(config, config.testset_seed)
#
#     test_results = {'cost': {},
#                     'fes': {},
#                     'return': {},
#                     'T0': 0.,
#                     'T1': 0.,
#                     'T2': 0.}
#
#     seed = range(11)
#     log_dir = config.save_dir + 'Rollout/'
#
#     for problem in test_set:
#         name = f"{problem.__str__()}_{problem.dim}" if config.mix_dim else problem.__str__()
#         test_results['cost'][name] = []
#         test_results['fes'][name] = []
#         test_results['return'][name] = []
#
#     T0 = cal_T0(config.dim, config.max_fes)
#     test_results['T0'] = T0
#     pbar_len = test_set.N * 11  # 跑51次
#
#     with tqdm(total = pbar_len, desc = f'Rollout Epoch {rollout_epoch}') as pbar:
#         for problem_id, problem in enumerate(test_set):
#             T1 = 0
#             T2 = 0
#             for run in range(11):
#                 start = time.perf_counter()
#                 np.random.seed(seed[run])
#                 problem.reset()
#                 env = Env(problem, config)
#                 info = agent.rollout_episode(env)
#                 cost = info['cost']
#                 while len(cost) < 51:
#                     cost.append(cost[-1])
#                 fes = info['fes']
#                 R = info['return']
#                 end = time.perf_counter()
#                 if problem_id == 0:
#                     T1 += env.problem.T1
#                     T2 += (end - start) * 1000
#
#                 name = f"{problem.__str__()}_{problem.dim}" if config.mix_dim else problem.__str__()
#                 test_results['cost'][name].append(cost)
#                 test_results['fes'][name].append(fes)
#                 test_results['return'][name].append(R)
#                 pbar_info = {'problem': name,
#                              'run': run,
#                              'gbest': cost[-1],
#                              'fes': fes,
#                              'Return': R}
#                 pbar.set_postfix(pbar_info)
#                 pbar.update(1)
#                 steps = (rollout_epoch - 1) * 11 + run + 1
#                 if not config.no_tb:
#                     log_to_tb_rollout(tb_logger, name, cost[-1], fes, R,
#                                       T0, env.problem.T1, (end - start) * 1000, steps)
#             if problem_id == 0:
#                 test_results['T1'] = T1 / 11
#                 test_results['T2'] = T2 / 11
#         if not os.path.exists(log_dir):
#             os.makedirs(log_dir)
#         log_path = log_dir + 'rollout' + str(rollout_epoch) + '.pkl'
#         if not config.no_saving:
#             with open(log_path, 'wb') as f:
#                 pickle.dump(test_results, f, -1)
#     torch.set_grad_enabled(True)
#     np.random.seed(config.seed)

def single_test(problem, config, agent, config_run):
    T1 = 0
    T2 = 0

    seed = range(config_run + 1)
    name = f"{problem.__str__()}_{problem.dim}" if config.mix_dim else problem.__str__()
    results = {'cost': [], 'fes': [], 'return': [], 'Logging': [],'T1': 0.0, 'T2': 0.0}
    for run in range(config_run):
        start = time.perf_counter()
        np.random.seed(seed[run + 1])
        torch.manual_seed(seed[run + 1])
        torch.cuda.manual_seed(seed[run + 1])
        problem.reset()
        env = Env(problem, config)
        info = agent.rollout_episode(env)
        cost = info['cost']
        while len(cost) < 51:
            cost.append(cost[-1])
        fes = info['fes']
        R = info['return']

        end = time.perf_counter()

        T1 += env.problem.T1
        T2 += (end - start) * 1000
        results['cost'].append(cost)
        results['fes'].append(fes)
        results['return'].append(R)
        results['Logging'].append(info['Logging'])
    T1 /= config_run
    T2 /= config_run
    results['T1'] = T1
    results['T2'] = T2
    return name, results

def single_test_wrapper(args):
    problem, config, agent, config_run = args
    return single_test(problem, config, agent, config_run)

def rollout_experiment(config, file_path, train_epoch, tb_logger = None):
    torch.set_grad_enabled(False)
    with open(file_path + '.pkl', 'rb') as f:
        agent = pickle.load(f)
    _, test_set = construct_problem(config, config.testset_seed)


    test_results = {'cost': {},
                    'fes': {},
                    'return': {},
                    'T0': 0.,
                    'T1': 0.,
                    'T2': 0.}

    log_dir = config.save_dir + 'Rollout/'

    for problem in test_set:
        name = f"{problem.__str__()}_{problem.dim}" if config.mix_dim else problem.__str__()
        test_results['cost'][name] = []
        test_results['fes'][name] = []
        test_results['return'][name] = []

    T0 = cal_T0(config.dim, config.max_fes)
    test_results['T0'] = T0

    n_process = 3
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method("spawn")
    with multiprocessing.Pool(processes = int(n_process)) as pool:
        tasks = [(problem, config, agent, config.rollout_run) for problem in test_set]
        results = list(
            tqdm(pool.imap(single_test_wrapper, tasks), total = test_set.N, desc = "Rollout")
        )

    for problem_name, result in results:
        test_results['cost'][problem_name] = result['cost']
        test_results['fes'][problem_name] = result['fes']
        test_results['return'][problem_name] = result['return']
        test_results['T1'] += result['T1']
        test_results['T2'] += result['T2']

    test_results['T1'] /= test_set.N
    test_results['T2'] /= test_set.N

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = log_dir + 'rollout' + str(int(train_epoch / config.rollout_interval)) + '.pkl'
    if not config.no_saving:
        with open(log_path, 'wb') as f:
            pickle.dump(test_results, f, -1)

    if not config.no_tb:
        log_to_tb_rollout_experiment(tb_logger, test_results, train_epoch)
    torch.set_grad_enabled(True)
