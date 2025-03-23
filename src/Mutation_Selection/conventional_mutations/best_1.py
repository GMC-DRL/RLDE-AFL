from ..basic_mutation import basic_mutation
import numpy as np  
class best_1(basic_mutation):
    def get_parameters_numbers(self):
        # F --scaling factor 0<=F<=1
        return 1
    
    # def mutation(self, env,individual_indice):
    #     """
    #     Perform mutation on an individual in the population.
    #     Parameters:
    #     - env: The environment object containing the population and mutation parameters.
    #     - individual_indice: The index of the individual to mutate.
    #     Returns:
    #     - new_individual: The mutated individual.
    #     """
        
    #     population_object=env.population
    #     parameters=env.action['mutation_parameters']
        
    #     F = parameters[0]
    #     population=population_object.current_vector
    #     # to be done 
    #     best_individual=np.min(population_object.current_fitness)
    #     # print('best_individual',best_individual)
    #     best_individual_indice=np.argmin(population_object.current_fitness)
    #     # print('best_individual_indice',best_individual_indice)        
    #     Len=population_object.pop_size
    #     indices = random.sample(range(Len), 2)
    #     x1, x2= population[indices[0]], population[indices[1]]
    #     mutated_vector =population[best_individual_indice]+F*(x1-x2)
        
    #     # is it correct?
    #     new_individual=mutated_vector            
    #     # print('new_individual',new_individual)
    #     return new_individual
    
       # population version 
    def mutation(self,env,pop_indexs,parameters):
        """
        Perform mutation operation on the population using the DE/best/1 strategy.
        Args:
            env (object): The environment object containing the population.
            pop_indexs (list): List of population indices to be mutated.
            parameters (numpy.ndarray): Array of parameters where the first column is the mutation factor F.
        Returns:
            numpy.ndarray: The mutated vector for the given population indices.
        """
        
        
        population_object=env.population
        
        
        F = parameters[:,0]
        F = F[:, np.newaxis]
        population = population_object.current_vector
        
        best_individual_indice=np.argmin(population_object.current_fitness)
        best_individual=population[best_individual_indice]
        
        random_indices=self.construct_random_indices(env,len(pop_indexs),2)
        x1,x2=population[random_indices.T] 
        
        mutated_vector =best_individual+F*(x1-x2)
        # mutated_vector= mutated_vector[pop_indexs]
        mutated_vector=self.re_boudary(env,mutated_vector)
        return mutated_vector