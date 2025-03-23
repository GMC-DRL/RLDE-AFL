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
def last_main():
    # Initialize the main code here
    # Your code goes here
    problem = 'bbob'
    dim = 10
    upperbound = 5
    train_batch_size = 1
    test_batch_size = 1
    difficulty = 'easy'
    train_set, test_set = construct_problem(problem, dim, upperbound, train_batch_size, test_batch_size, difficulty, instance_seed = 0)
    f0 = train_set.data[0]
    f0.reset()
    population = Population(pop_size = 50,dim = 10, max_G = 1000, max_fes = 300000, problem = f0)
    # 100 would be great
    # print('population:',population.current_vector)
    loop=0
    print(f0.optimum)
    avg_fitness_list = []
    best_fitness_list = []
    
    # the train begin:
    while population.fes < population.max_fes:
        # feature part
        loop+=1
        # print('loop:',loop)
        # print('generation:',population.generation)
        # check the optimization
        if loop%100==0:
            print('x',population.current_vector)
            # print('y',population.current_fitness)
            avg_fitness,best_fitness=population.calculate_result()
            avg_fitness_list.append(avg_fitness)
            best_fitness_list.append(best_fitness)
            print('avg_fitness:',avg_fitness)
            print('best_fitness:',best_fitness)
        fe=feature_extractor.Feature_Extractor(attention_order = "individual") # is it correct to use individual?
        feature=population.get_feature(fe)
        # print('feature:',feature)
        
        # todo: the selection policy for the mutation operators
        # about the normal distribution 
        Len = population.pop_size
        # new_population = population
        # new_population.current_vector = np.empty_like(population.current_vector)
        # new_population.current_fitness = np.empty_like(population.current_fitness)
        
        # mutation part
        mutation_operators = "MadDE"
        # a rough implementation of the random selection
        # mutation_operators=select_mutation.random_select()
        # a temporary solution for the mutation operators
        # it will be replaced after implementing the selection policy
        selected_operator_class = select_mutation.select_mutation_operator(mutation_operators)
        
        for i in range(Len):
            
            # mutation part        
            if selected_operator_class:
                mutation_operator = selected_operator_class
                # print('mutation_operator', mutation_operator)
                # print('hello:',mutation_operator.hello)
                mutation_parameters = [0.5]  # Example parameter for mutation, you can adjust it as needed
                mutated_individual = mutation_operator.mutation(mutation_parameters, population,i)
                # print('mutated_individual',mutated_individual)
            else:
                print(f"Mutation operator '{mutation_operators}' not found.")


            # crossover part
            crossover_parameters = [0.5]  # Example parameter for crossover, you can adjust it as needed
            crossover_operators="binomial_crossover"
            # where should we dicide the crossover_operators
            crossover_operator_class= select_crossover.select_crossover_operator(crossover_operators)
            crossover_individual = crossover_operator_class.crossover(crossover_parameters,population,i,mutated_individual)
            # crossover_individual = rand_1_crossover().crossover(crossover_parameters,population,i,mutated_individual)
            # print('crossover_individual',crossover_individual)
            # crossover_population = np.empty_like(mutated_population)
            # for i in range(len(population.current_vector)):
            #     for j in range(len(population.current_vector[0])):
            #         if random.random() < 0.5:
            #            crossover_population[i][j] = mutated_population[i][j]
            #         else:
            #             crossover_population[i][j] = population.current_vector[i][j]


            # selection part
            selected_individual=greedy_select().select(population,i,crossover_individual)
            # selected_population = np.empty_like(crossover_population)
            # for i in range(len(population.current_vector)):
            #     score_previous=population.current_fitness[i]
            #     score_target= population.get_individual_costs(f0,crossover_population[i])

            #     if score_previous < score_target:
            #         selected_population[i] = population.current_vector[i]

            #     else:
            #         selected_population[i]= crossover_population[i]

            # update population
            population.current_vector[i]=selected_individual
            population.current_fitness[i]=population.get_individual_costs(selected_individual)
            
        population.update_population()
    # print('population:',population.current_vector)
    # print('population:',population.current_fitness)
    print('result:',population.current_vector,population.calculate_result())

    # the result is the cost of the population, not individual
    # todo: we can record the best individual in the history
    plt.figure(figsize=(10, 5))
    plt.plot(avg_fitness_list, label='Average Fitness')
    plt.plot(best_fitness_list, label='Best Fitness')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Fitness over Generations')
    plt.legend()
    plt.show()
  
  

# todo: rollouts every 3 epoches?



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
