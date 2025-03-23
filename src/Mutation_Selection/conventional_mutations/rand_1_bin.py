# removed?
from ..basic_mutation import basic_mutation
import random
import numpy as np

class rand_1_bin(basic_mutation):
    def get_parameters_numbers(self):
        #F --scaling factor 0<=F<=1
        return 1
    
    # def mutation(self, env,individual_indice):
    #     population_object=env.population
    #     parameters=env.action['mutation_parameters']
        
    #     population=population_object.current_vector
    #     F = parameters[0]
    #     Len=population_object.pop_size
    #     new_individual=np.empty_like(population[individual_indice])
    #     indices = random.sample(range(Len), 3)
        
        
    #     # Select three random individuals from the population
    #     x1, x2, x3 = population[indices[0]], population[indices[1]], population[indices[2]]
    #     # print('x1',x1)
    #     # print('x2',x2)
    #     # print('x3',x3)
    #     if random.random()<env.action['crossover_parameters'][0] or individual_indice==random.randint(0,population_object.pop_size):
    #         # Perform mutation using rand1 strategy
    #         mutated_vector = x1 + F * (x2 - x3)
    #     else:
    #         mutated_vector = population[individual_indice]
    #     # print('mutated_vector',mutated_vector)
    #     # Add the mutated vector to the new population
    #     new_individual = mutated_vector

    #     return new_individual
    
    def mutation(self,env,indices,parameters):
        population_object=env.population
        population=population_object.current_vector

        random_indices=self.construct_random_indices(env,len(indices),3)
        x1,x2,x3=population[random_indices.T]
        F=parameters[:,0]
        F = F[:, np.newaxis]
        sub_pop=self.construct_sub_vector(env,indices)
        # if random.random() < env.action['crossover_parameters'][0] or individual_index == random.randint(0, Len - 1):
        # not complete
        if random.random() < env.action['crossover_parameters'][0]:
            mutated_vector = x1 + F * (x2 - x3)
            mutated_vector=self.re_boudary(env,mutated_vector)
        else:
            mutated_vector=sub_pop
            
        
        return mutated_vector
            