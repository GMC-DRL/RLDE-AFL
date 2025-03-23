# example : gleet
import argparse
import json
import time
import os


crossover_operators = ["binomial", "exponential", "prefential", "MDE_pBX"]
mutation_operators =["best_1", "best_2", "rand_1", "rand_2","current_to_best_1",
                            "rand_to_best_1","current_to_rand_1","MDE_pBX", "pro_rand_1",
                            "TopoMut_DE","JADE", "HARDDE","current_to_rand_1_archive","weighted_rand_to_qbest_1"]

def get_config(args = None, config_file = None):
    parser = argparse.ArgumentParser()

    # --------------------- Common Config ---------------------
    parser.add_argument('--problem', default = 'bbob', choices = ['bbob', 'bbob-noisy'], help = 'specify the problem suite')
    parser.add_argument('--difficulty', default = 'easy', choices = ['easy', 'difficult'], help = 'difficulty level')
    parser.add_argument('--dim', type = int, default = 10, help = 'dimension of search space')
    parser.add_argument('--upperbound', type = float, default = 5., help = 'upperbound of search space')
    parser.add_argument('--device', default = 'cpu', help = 'device to use')
    parser.add_argument('--run_experiment', default = None, action = 'store_true')  # todo 只测 epoch0 和 epoch 101
    parser.add_argument('--train', default = None, action = 'store_true', help = 'switch to train mode')
    parser.add_argument('--test', default = None, action = 'store_true', help = 'switch to test mode')
    parser.add_argument('--run_name', default = 'test', help = 'name to identify the run')
    parser.add_argument('--load_path', default = None, help = 'path to load model agent')
    parser.add_argument('--resume', default = None, help = 'resume from previous checkpoint file')

    parser.add_argument('--no_saving', action = 'store_true', help = 'disable saving checkpoints')
    parser.add_argument('--no_tb', action = 'store_true', help = 'disable Tensorboard logging')
    parser.add_argument('--no_rollout', action = 'store_true', help = 'disable rollout in training')
    parser.add_argument('--no_save_epoch', action = 'store_true', help = 'save agent per epoch')
    parser.add_argument('--show_figs', action = 'store_true', help = 'enable figure logging')
    # todo rollout
    # --------------------- Training Parameters ---------------------
    parser.add_argument('--max_learning_step', type = int, default = 1500000,
                        help = 'the maximum learning step for training')
    parser.add_argument('--train_batch_size', type = int, default = 1,
                        help = 'batch size of training set')
    parser.add_argument('--max_epoch', type = int, default = 100)
    parser.add_argument('--mix_dim', action = 'store_true', help = 'random 10 20 dim')

    # --------------------- Testing Parameters ---------------------
    parser.add_argument('--agent_model', type = str, default = None, help = 'model you want test')
    parser.add_argument('--agent_load_dir', type = str, help = 'load your agent model')
    parser.add_argument('--test_batch_size', type = int, default = 1,
                        help = 'batch size of testing set')



    # --------------------- Environment Parameters ---------------------
    parser.add_argument('--pop_size', type = int, default = 100, help = 'population size')
    parser.add_argument('--max_G', type = int, default = 1000, help = 'maximum number of generations')
    parser.add_argument('--max_fes', type = int, default = 10000, help = 'maximum number of function evaluations')
    # max_fes>20000    ?50000
    parser.add_argument('--de_mutation_op', type = str, nargs = '+', default = mutation_operators,
                        help = 'Select De_mutation_op you want add in agent')
    # --de_mutation_op best_1 best_2 rand_1 rand_2
    parser.add_argument('--crossover_op', type = str, nargs = '+', default = crossover_operators,
                        help = 'Select crossover_op you want add in agent')
    parser.add_argument('--reward_ratio', type = float, default = 0.0)

    # --------------------- Agent Parameters ---------------------
    parser.add_argument('--fe_hidden_dim', type = int, default = 64,
                        help = 'the output dim of feature_extractor', choices = [64, 128, 256])
    parser.add_argument('--fe_n_layers', type = int, default = 1,
                        help = 'the layers of feature_extractor', choices = [1, 3, 5])
    parser.add_argument('--fe_lr', type = float, default = 0.00001)
    parser.add_argument('--actor_lr', type = float, default = 0.00001)
    parser.add_argument('--critic_lr', type = float, default = 0.00001)
    parser.add_argument('--lr_decay', type = float, default = 0.9862327, help = 'learning rate decay per epoch',
                        choices = [0.998614661, 0.9862327])
    parser.add_argument('--ppo_gamma', type = float, default = 0.99)
    parser.add_argument('--ppo_lamda', type = float, default = 0.9)
    parser.add_argument('--ppo_eps', type = float, default = 0.2)
    parser.add_argument('--ppo_n_step', type = int, default = 10)
    parser.add_argument('--ppo_k_epochs', type = int, default = 3)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--trainset_seed', type = int, default = 0)
    parser.add_argument('--testset_seed', type = int, default = 0)
    # test todo 后面再搞
    parser.add_argument('--fe_train', action = 'store_true', help = 'training fe')
    parser.add_argument('--mlp', action = 'store_true', help = 'fe is mlp?')
    parser.add_argument('--fe_gleet', action = 'store_true') # 特征是否是手工，消融用
    parser.add_argument('--test_all', action = 'store_true', help = '测试的时候把训练的也加进来')
    


    parser.add_argument('--save_dir', type = str, default = 'outputs/model/train', help = 'directory to write output models to')
    parser.add_argument('--log_dir', type = str, default = 'outputs/logs', help = 'directory to write TensorBoard information to')
    parser.add_argument('--n_checkpoint', type = int, default = 20, help = 'number of training checkpoints')
    parser.add_argument('--n_logpoint', type = int, default = 50, help = 'number of logpoints')
    parser.add_argument('--rollout_interval', type = int, default = 3, help = 'rollout after training per interval')
    parser.add_argument('--rollout_run', type = int, default = 10, help = 'run times problem in rollout')

    parser.add_argument('--test_run', type = int, default = 50)
    parser.add_argument('--log_step', type = int, default = 50, help = 'log info every log_step gradient steps')
    config = parser.parse_args(args)

    # if a config file is provided, update the config from the file
    if config_file and os.path.isfile(config_file):
        with open(config_file, 'r') as f:
            file_config = json.load(f)
            for key, value in file_config.items():
                if hasattr(config, key):
                    setattr(config, key, value)

    # config.maxFEs = 2000 * config.dim
    #
    # config.n_logpoint = 50 # 记录点的log个数
    config.save_interval = config.max_learning_step // config.n_checkpoint # 保存model 间隔
    config.log_interval = config.max_fes // config.n_logpoint # 记录的间隔

    # config.run_time = time.strftime("%Y%m%dT%H%M%S")
    # config.run_name = "{}_{}".format(config.run_name, config.run_time) \
    #     if not config.resume else config.resume.split('/')[-2]
    config.run_time = f'{config.run_name}-{time.strftime("%Y%m%dT%H%M%S")}_{config.problem}_{config.difficulty}_{config.dim}D'
    config.test_log_dir = config.log_dir + '/test/' + config.run_time + '/'
    config.rollout_log_dir = config.log_dir + '/rollout/' + config.run_time + '/'
    # config.save_dir = os.path.join(
    #     config.output_dir,
    #     "{}_{}".format(config.problem, config.dim),
    #     config.run_name
    # ) if not config.no_saving else None
    if config.train:
        config.save_dir = config.save_dir + '/' + config.run_time + '/'

    if config.no_saving:
        config.save_dir = None
        config.test_log_dir = None
        config.rollout_log_dir = None

    if config.run_experiment:
        config.save_dir = config.save_dir + '/' + config.run_time + '/'
        config.agent_model = "epoch" + str(config.max_epoch + 1) # todo
        config.agent_load_dir = config.save_dir + 'Epoch/'
    return config

def get_config_table(config):
    from prettytable import PrettyTable

    table = PrettyTable()
    table.field_names = ["Parameters", "Value"]

    config_dict = vars(config)
    for key, value in config_dict.items():
        table.add_row([key, value])
    print(table)

if __name__ == '__main__':
    config = get_config(config_file = "config.json")
    assert ((config.train is not None) +
            (config.test is not None)) == 1, \
            'Among train, test, only one mode can be given at one time'
    get_config_table(config)

