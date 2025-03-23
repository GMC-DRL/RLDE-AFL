# removed?
from ..basic_mutation import basic_mutation
import numpy as np
import random

class rand_2_bin(basic_mutation):
    def get_parameters_numbers(self):
    #F --scaling factor 0<=F<=1 
        return 1
    
    # def mutation(self, env,individual_indice):
    #     """
    #     Perform mutation on an individual in the population.
    #     Args:
    #         env (Environment): The environment object containing the population.
    #         individual_indice (int): The index of the individual to mutate.
    #     Returns:
    #         The mutated individual.
    #     Raises:
    #         None.
    #     """
        
    #     population_object=env.population
    #     parameters=env.action['mutation_parameters']
        
    #     F = parameters[0]
        
    #     # for i in range(len(population)):
    #     # Select three random individuals from the population
    #     population=population_object.current_vector
    #     if random.random()<env.action['crossover_parameters'][0] or individual_indice==random.randint(0,population_object.pop_size):
    #         Len=population_object.pop_size
    #         indices = random.sample(range(Len), 5)
    #         x1, x2, x3,x4,x5 = population[indices[0]], population[indices[1]], population[indices[2]], population[indices[3]],population[indices[4]]

    #         # Perform mutation using rand2 strategy
    #         mutated_vector=  x1+F*(x2-x3+F*(x4-x5)) # is it correct?
    #         # Add the mutated vector to the new population
    #         new_individual=mutated_vector        
    #     else:
    #         new_individual=population[individual_indice]
        
    #     return new_individual
    
    def mutation(self,env,indices,parameters):
        population_object=env.population
        population=population_object.current_vector
        random_indices=self.construct_random_indices(env,len(indices),5)
        x1,x2,x3,x4,x5=population[random_indices.T]
        F=parameters[:,0]
        F = F[:, np.newaxis]
        sub_pop=self.construct_sub_vector(env,indices)
        if np.random.rand() < env.action['crossover_parameters'][0] or indices == random.randint(0, population_object.pop_size):
            mutated_vector = x1 + F * (x2 - x3 + F * (x4 - x5))
            mutated_vector=self.re_boudary(env,mutated_vector)
        else:
            mutated_vector=sub_pop
        return mutated_vector