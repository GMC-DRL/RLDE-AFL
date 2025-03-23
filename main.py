from problem import bbob
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import multiprocessing
from Population import Population

from utils import construct_problem
from Feature_Extractor import feature_extractor,attention_block
from train import train
from test import test
from Mutation_Selection import *
from Mutation_Selection.Select import select_mutation
from Mutation_Selection import basic_mutation

from CrossOver_Selection import *
from CrossOver_Selection.Select import select_crossover
from CrossOver_Selection.simple_crossover import basic_crossover

from Selection_part.selection_part import greedy_select

from config import *
import torch
from tensorboardX import SummaryWriter
from logger import estimate_test_statics

def main():
    if multiprocessing.get_start_method(allow_none=True) is None:
        multiprocessing.set_start_method("spawn")

    config = get_config()
    assert ((config.train is not None) +
            (config.test is not None) +
            (config.run_experiment is not None)) == 1, \
        'Among train, test, run_experiment only one mode can be given at one time'
    # get_config_table(config) # todo pip install prettytable

    tb_logger = None
    if not config.no_tb:
        tb_logger = SummaryWriter(os.path.join("tensorboard", "{}_{}".format(config.problem, config.dim),
                                           config.run_name))
    # --------------------- Start train ---------------------
    if config.train:
        torch.set_grad_enabled(True)
        train(config, tb_logger)

    # --------------------- Start inference ---------------------
    if config.test:
        torch.set_grad_enabled(False)
        test(config)
        estimate_test_statics(config.test_log_dir)

    if config.run_experiment:
        torch.set_grad_enabled(True)
        train(config, tb_logger)

        torch.set_grad_enabled(False)
        test(config)
        estimate_test_statics(config.test_log_dir)
    pass

if __name__=="__main__":
    main()
