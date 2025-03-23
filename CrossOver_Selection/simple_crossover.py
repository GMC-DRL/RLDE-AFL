# a temporary crossover for rand 1
# only used in prelimary test
# todo: remove it 
import numpy as np
from CrossOver_Selection.basic_crossover import basic_crossover
class simple_crossover(basic_crossover):
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
        
        # one dimension that guarantees the crossover
        idx = np.random.randint(0, len(crossover_individual))
        for j in range(len(crossover_individual)):
            if np.random.random() < crossover_parameters or j == idx:
                crossover_individual[j] = mutated_individual[j]
            else:
                crossover_individual[j] = origin_individual[j]
        return crossover_individual
    
    def crossover(self,env,mutated_vector,indices):
        
        population_object = env.population
        parameters = env.action['crossover_parameters']
        crossover_parameters = parameters[0]
        sub_pop = self.construct_sub_vector(env,indices)
        crossover_vector = np.empty_like(mutated_vector)    
        mask= np.random.rand(len(mutated_vector))<crossover_parameters
        crossover_vector=np.where(mask,mutated_vector,sub_pop)
        return crossover_vector
        
        
        
        
    def construct_sub_vector(self,env,individual_indice):
        """
        Constructs a origin vector by selecting individuals from the population.
        Args:
            env (object): The environment object containing the population.
            individual_indice (list of int): Indices of individuals to be selected from the population.
        Returns:
            numpy.ndarray: An array of selected individuals from the population.
        """
        
        population_object=env.population
        population=population_object.current_vector
        selected_pop = np.array([population[i] for i in individual_indice])
        return selected_pop
    
    
    def construct_random_indices(self,env,sub_pop_size,x_num):
        indices= np.random.choice(env.population.pop_size,(sub_pop_size,x_num))
        return indices    
    