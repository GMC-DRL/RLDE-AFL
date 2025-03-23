from Environment.env import Env
from Mutation_Selection.Select import select_mutation
from CrossOver_Selection.Select import select_crossover
from Selection_part.selection_part import greedy_select
from config import *
import numpy as np
from utils import construct_problem
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(3185)
def differential_evolution(config):
    # Construct the problem
    # print('config:',config)
    _, test_set = construct_problem(config,233)
    
    # Initialize the environment with the first problem in the test set
    problem = test_set[5]
    problem.reset()
    problem_name= problem.__str__()
    mutation_selector = select_mutation(config.de_mutation_op)
    crossover_selector = select_crossover(config.crossover_op)
    config.mutation_selector = mutation_selector
    config.crossover_selector = crossover_selector
    env = Env(problem, config)
    env.reset()
    pop_size=env.population.pop_size
    
    # Define the mutation and crossover parameters (example parameters, adjust as needed)
    mutation_parameters = {
        'param1': 0.5,
        'param2': 0.3,
        'param3': 0.4
    }
    crossover_parameters = {
        'param1': 0.5,
    }
    
    mutationparams=np.zeros((pop_size,3))
    mutationparams[:,0]=mutation_parameters['param1']
    mutationparams[:,1]=mutation_parameters['param2']
    mutationparams[:,2]=mutation_parameters['param3']
    # print('mutationparams:',mutationparams)
    crossoverparams = np.full((pop_size,), 0.5)
    # print('crossoverparams:', crossoverparams)
    fitness_over_generations = []
    gbest_over_generations = []
    # Differential Evolution loop
    for generation in range(config.max_G):
        # print(f"Generation {generation}")
        
        mutation_amount= len(config.de_mutation_op)
        crossover_amount = len(config.crossover_op)
        # Randomly select mutation and crossover operators by index
        mutation_op_index = np.random.randint(0, mutation_amount)
        crossover_op_index = np.random.randint(0, crossover_amount)
        
        mutation_op = config.de_mutation_op[mutation_op_index]
        crossover_op = config.crossover_op[crossover_op_index]
        
        # print(f"Selected mutation operator: {mutation_op}")
        # print(f"Selected crossover operator: {crossover_op}")
        
        mutation_operator = mutation_selector.select_mutation_operator(mutation_op_index)
        crossover_operator = crossover_selector.select_crossover_operator(crossover_op_index)
        # print(f"Before update: current_vector.shape = {env.population.current_vector.shape}")
        # Apply mutation and crossover to the population
        pop_index=[i for i in range(env.population.pop_size)]
        mutated_population = mutation_operator.mutation(env, pop_index, mutationparams)
        # print(f"mutated_population.shape = {mutated_population.shape}")
        crossover_population = crossover_operator.crossover(env, pop_index, mutated_population, crossoverparams)
        # print(f"crossover_population.shape = {crossover_population.shape}")
        updated_pop=greedy_select().select(env, crossover_population)
        
        # Update the environment with the new population
        
        # env.population.current_vector = updated_pop
        # env.population.current_fitness = env.get_costs(env.problem, updated_pop)
        # print(f"After update: current_vector.shape = {env.population.current_vector.shape}")
        env.population.update_population()
        
        mean_fitness = np.mean(env.population.current_fitness)
        fitness_over_generations.append(mean_fitness)
        gbest_over_generations.append(env.population.gbest_val)
        # Print the updated population
        if generation % 10 == 0:
            print('generation:',generation)
            # print("Updated Population:", env.population.current_vector)
            print("Updated Fitness Scores:", np.mean(env.population.current_fitness))
            print("Updated Gbest Value:", env.population.gbest_val)
        
        # Optionally, you can add code to track and plot the fitness over generations
    return fitness_over_generations, gbest_over_generations,problem_name

if __name__ == '__main__':
    # Define a sample config object with necessary attributes
    config = get_config()
    config.max_G=15
    config.seed=233
    get_config_table(config)
    de_mutation_op=["best_1", "best_2", "rand_1", "rand_2","current_to_best_1",
                            "rand_to_best_1","current_to_rand_1","MDE_pBX", "pro_rand_1",
                            "TopoMut_DE","JADE", "HARDDE","current_to_rand_1_archive","weighted_rand_to_qbest_1"]
    
    crossover_op= ["binomial", "exponential", "prefential", "MDE_pBX"]
    
    results1={}
    results2={}
    result1_gbest={}
    result2_gbest={}
    for op in de_mutation_op:
        config.crossover_op=["binomial"]
        config.de_mutation_op=[]
        config.de_mutation_op.append(op)
        fitness_over_generations,gbest_over_generations,name1=differential_evolution(config)
        results1[op] = fitness_over_generations
        result1_gbest[op]=gbest_over_generations
    print("first finish")
    
    for op in crossover_op:
        # print(op)
        config.de_mutation_op=["rand_1"]
        config.crossover_op=[]
        config.crossover_op.append(op)
        fitness_over_generations,gbest_over_generations,name2=differential_evolution(config)
        results2[op] = fitness_over_generations
        result2_gbest[op]=gbest_over_generations
    print("second finish")

    # save_to_excel
    # path1 = 'src/outputs/mutation_comparision.xlsx'
    # data1 = []
    # for op in de_mutation_op:
    #     mean_value = results1[op][-1]
    #     gbest_value = result1_gbest[op][-1]
    #     data1.append([mean_value, gbest_value])
    # df = pd.DataFrame(data1, columns=['Mean Fitness', 'Gbest Value'], index=de_mutation_op)

    # with pd.ExcelWriter(path1, mode='a',if_sheet_exists='overlay') as writer:
    #     df.to_excel(writer, sheet_name=name1)
    # path2 = 'src/outputs/crossover_comparision.xlsx'
    # data2 = []
    # for op in crossover_op: 
    #     mean_value = results2[op][-1]
    #     gbest_value = result2_gbest[op][-1]
    #     data2.append([mean_value, gbest_value])
    # df = pd.DataFrame(data2, columns=['Mean Fitness', 'Gbest Value'], index=crossover_op)
    # with pd.ExcelWriter(path2, mode='a',if_sheet_exists='overlay') as writer:
    #     df.to_excel(writer, sheet_name=name2)

    # Plot the results
    # 绘制变异算子的结果
    plt.figure(figsize=(12, 8))
    for op, fitness_over_generations in results1.items():
        plt.plot(fitness_over_generations, label=op)
    
    plt.xlabel('Generations')
    plt.ylabel('Mean Fitness')
    plt.title('Comparison of DE Mutation Operators')
    plt.legend()
    plt.show()
    
    # 绘制交叉算子的结果
    plt.figure(figsize=(12, 8))
    for op, fitness_over_generations in results2.items():
        plt.plot(fitness_over_generations, label=op)
    
    plt.xlabel('Generations')
    plt.ylabel('Mean Fitness')
    plt.title('Comparison of DE Crossover Operators')
    plt.legend()
    plt.show()
        
            