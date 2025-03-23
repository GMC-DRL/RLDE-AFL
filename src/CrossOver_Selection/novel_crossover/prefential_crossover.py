from ..basic_crossover import basic_crossover
import math
import numpy as np

'''
----步骤----
需要在初始化时生产current_vector和prefential_vector, current_vector[i] better than prefential_vector[i]
若调用Prefential crossover, 其不需要mutation, 直接使用S1和prefential_vector进行cross再尝试替换S1
若替换失败, 进行第二次尝试, 随机选择一个mutation, 用其他crossover生成新个体尝试替换S1
若成功则结束, 若再次失败, 尝试替换prefential_vector对应的个体

--改进--
把attempt1和attempt2放在一起, 已经选好mutation
会在这里计算cost, 可能会造成浪费
'''
class prefential(basic_crossover):
    def get_parameters_numbers(self):
        """
        Returns the number of parameters.
        """
        return 1
    
    def crossover(self,env,crossover_indice,mutated_individual):
        """
        Perform crossover operation between a mutated individual and an original individual from the population.
        Args:
            env (Environment): The environment object containing the population.
            crossover_indice (int): The index of the original individual in the population.
            mutated_individual (numpy.ndarray): The mutated individual.
        Returns:
            numpy.ndarray: The crossover individual resulting from the crossover operation.
        """
        
        population_object = env.population
        parameters = env.action['crossover_parameters']
        
        crossover_parameters = parameters[0]
        origin_individual = population_object.current_vector[crossover_indice]
        
        crossover_individual = np.empty_like(mutated_individual)

        # attempt 1, prefential crossover: use the prefential_vector to do
        idx = np.random.randint(0, len(crossover_individual)-1)
        pre_idx = np.random.randint(0, population_object.pop_size-1)
        prefential_individual = env.prefential_vector[pre_idx] # the random choose individual from prefential_vector
        for j in range(len(crossover_individual)):
            if np.random.random() < crossover_parameters or j == idx:
                crossover_individual[j] = prefential_individual[j]
            else:
                crossover_individual[j] = origin_individual[j]
        
        # compare the cost of the crossover_individual and the mutated_individual
        # is waste a comparison, maybe it can be used in the selection part?
        score_previous=population_object.current_fitness[crossover_indice]
        if env.get_individual_costs(crossover_individual) < score_previous:
            return crossover_individual
        
        # attempt 2 
        # one dimension that guarantees the crossover
        for j in range(len(crossover_individual)):
            if np.random.random() < crossover_parameters or j == idx:
                crossover_individual[j] = mutated_individual[j]
            else:
                crossover_individual[j] = origin_individual[j]
        
        return crossover_individual
        # try to replace prefential_vector in the selection part

    # population
    def crossover(self,env,crossover_indice,mutated_pop,parameters):
        """
        Perform crossover operation between a mutated individual and an original individual from the population.
        Args:
            env (Environment): The environment object containing the population.
            crossover_indice (int): The index of the original individual in the population.
            mutated_individual (numpy.ndarray): The mutated individual.
        Returns:
            numpy.ndarray: The crossover individual resulting from the crossover operation.
        """
        crossover_parameter = parameters
        crossover_parameter = crossover_parameter[:, np.newaxis]
        sub_origin_pop=self.construct_sub_origin_vector(env, crossover_indice)        
        sub_mutated_pop=self.construct_sub_mutated_vector(crossover_indice,mutated_pop)
        individual_size=env.problem.dim
        sub_pop_size=len(crossover_indice)
        
        # attempt 1, prefential crossover: use the prefential_vector to do
        prefential_pop = self.construct_prefential_individual(env, sub_pop_size)
        updated_pop = self.prefential_cross(env, crossover_indice, sub_mutated_pop, crossover_parameter, prefential_pop)
        
        # firstly do attempt1, record the better individuals, where do attempt2 for remaining individuals
        score_previous = env.population.current_fitness[crossover_indice]
        score_new = env.get_sub_pop_cost(updated_pop)

        successful_indices = []
        for i in range(sub_pop_size):
            if score_new[i] < score_previous[i]:
                sub_origin_pop[i] = updated_pop[i]
                successful_indices.append(i)
        # if successful_indices != []:
            # print("attempt 1 success")

        # attempt 2
        mask = np.random.rand(sub_pop_size, individual_size) < crossover_parameter
        for i in successful_indices:
            mask[i] = False  # 排除成功的个体

        updated_pop = np.where(mask, sub_mutated_pop, sub_origin_pop)
        return updated_pop
        
    def prefential_cross(self,env, crossover_indice, mutated_pop, crossover_parameter, prefential_pop):
        sub_pop_size = len(crossover_indice)
        individual_size = env.problem.dim

        mask = np.random.rand(sub_pop_size, individual_size) < crossover_parameter
        # print('mask:',mask)

        updated_pop = np.where(mask, mutated_pop, prefential_pop)
        return updated_pop
        
        
    def construct_prefential_individual(self,env,sub_pop_len):
        
        random_indices=self.construct_random_indices(env,sub_pop_len,1)
        random_indices=random_indices[0]
        prefential_individual = env.population.prefential_vector[random_indices.T]
        
        return prefential_individual
    
    