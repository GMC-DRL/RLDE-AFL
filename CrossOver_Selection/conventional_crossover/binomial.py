from ..basic_crossover import basic_crossover
import numpy as np


class binomial(basic_crossover):
    def get_parameters_numbers(self):
        """
        Returns the number of parameters.
        """
        return 1
    
        #todo: numpy/pytorch multiplied process 
    # def crossover(self,env,crossover_indice,mutated_individual):
    #     """
    #     Perform crossover operation between a mutated individual and an original individual from the population.
    #     Args:
    #         env (Environment): The environment object containing the population.
    #         crossover_indice (int): The index of the original individual in the population.
    #         mutated_individual (numpy.ndarray): The mutated individual.
    #     Returns:
    #         numpy.ndarray: The crossover individual resulting from the crossover operation.
    #     """
        
    #     population_object = env.population
    #     parameters = env.action['crossover_parameters']
        
    #     crossover_parameters = parameters[0]
    #     origin_individual = population_object.current_vector[crossover_indice]
        
    #     crossover_individual = np.empty_like(mutated_individual)
        
    #     # one dimension that guarantees the crossover
        
    #     # todo:numpy.random 
    #     # reason: for random seed
    #     idx = np.random.randint(0, len(crossover_individual)-1)
    #     for j in range(len(crossover_individual)):
    #         if np.random.random() < crossover_parameters or j == idx:
    #             crossover_individual[j] = mutated_individual[j]
    #         else:
    #             crossover_individual[j] = origin_individual[j]
    #     return crossover_individual
    
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
        # print('parameters:',parameters)
        # print('parameters.shape:',parameters.shape)
        # crossover_parameter = parameters[:,0]
        crossover_parameter = parameters
        crossover_parameter = crossover_parameter[:, np.newaxis]
        sub_origin_pop=self.construct_sub_origin_vector(env, crossover_indice)        
        sub_mutated_pop=self.construct_sub_mutated_vector(crossover_indice,mutated_pop)
        individual_size=env.problem.dim
        sub_pop_size=len(crossover_indice)
        # Generate a random mask based on crossover probability
        mask = np.random.rand(sub_pop_size, individual_size) < crossover_parameter

        # Ensure that each individual has at least one crossover point
        random_indices = np.random.randint(0, individual_size, size=sub_pop_size)
        mask[np.arange(sub_pop_size), random_indices] = True

        # Apply the mask to select elements from mutated_pop or original population
        updated_pop = np.where(mask, sub_mutated_pop, sub_origin_pop)

        return updated_pop
        
                 
        
    