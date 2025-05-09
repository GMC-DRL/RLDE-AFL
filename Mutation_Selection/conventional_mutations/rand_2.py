from ..basic_mutation import basic_mutation
import numpy as np

class rand_2(basic_mutation):
    def get_parameters_numbers(self):
        # F --scaling factor 0<=F<=1
        return 1
    
    # def mutation(self, env,individual_indice):
    #     """
    #     Perform mutation using the rand2 strategy.
    #     Args:
    #         env (Environment): The environment object.
    #         individual_indice (int): The index of the individual.
    #     Returns:
    #         new_individual: The mutated individual.
    #     """
        
    #     population_object=env.population
    #     parameters=env.action['mutation_parameters']
        
    #     F = parameters[0]
        
    #     # for i in range(len(population)):
    #     # Select three random individuals from the population
    #     population=population_object.current_vector
    #     Len=population_object.pop_size
    #     indices = random.sample(range(Len), 5)
    #     x1, x2, x3,x4,x5 = population[indices[0]], population[indices[1]], population[indices[2]], population[indices[3]],population[indices[4]]
        
    #     # Perform mutation using rand2 strategy
    #     mutated_vector=  x1+F*(x2-x3+F*(x4-x5)) # is it correct?
    #     # Add the mutated vector to the new population
    #     new_individual=mutated_vector        
        
        
    #     return new_individual
    
   # population version 
    def mutation(self,env,pop_indexs,parameters):
        """
        Perform mutation using the rand/1 strategy.
        This method generates a mutated vector for a given population using the DE/rand/1 mutation strategy.
        It constructs two random vectors from the population and combines them with a scaling factor to produce the mutated vector.
        Args:
            env (object): The environment object containing the population and mutation parameters.
            pop_indexs (list): List of indices of the population members to be mutated.
        Returns:
            np.ndarray: The mutated vector generated by the rand/1 strategy.
        """
        population_object=env.population
        # parameters=env.action['mutation_parameters']
        population = population_object.current_vector
        
        random_indices= self.construct_random_indices(env,len(pop_indexs),5)
        x1,x2,x3,x4,x5=population[random_indices.T]
        F=parameters[:,0]
        F = F[:, np.newaxis]
        
        mutated_vector=  x1+F*(x2-x3)+F*(x4-x5)
        mutated_vector=self.re_boudary(env,mutated_vector)
        return mutated_vector       
    