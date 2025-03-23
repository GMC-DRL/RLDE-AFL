import numpy as np
from ..basic_crossover import basic_crossover

class exponential(basic_crossover):
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
        
        start = np.random.randint(0, len(crossover_individual)-1)
        length = 1
        while np.random.random() < crossover_parameters:
            length += 1
        for j in range(len(crossover_individual)):
            if start <= j < start + length:
                crossover_individual[j] = mutated_individual[j]
            else:
                crossover_individual[j] = origin_individual[j]
        return crossover_individual
            
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
        # Generate a random mask based on crossover probability
        mask=self.construct_mask(env,sub_pop_size,individual_size,crossover_parameter)
        
        # Apply the mask to select elements from mutated_pop or original population
        updated_pop = np.where(mask, sub_mutated_pop, sub_origin_pop)

        return updated_pop
    
    def construct_mask(self,env,sub_pop_size,individual_size,crossover_parameter):
        start_positions = np.random.randint(0, individual_size-1, size=sub_pop_size)
        random_values = np.random.random(size=(sub_pop_size, individual_size))
        lengths = np.ones(sub_pop_size, dtype=int)
        lengths += np.sum(random_values < crossover_parameter, axis=1)
        
        # Create the mask
        mask = np.zeros((sub_pop_size, individual_size), dtype=bool)
        for i in range(sub_pop_size):
            start = start_positions[i]
            length = lengths[i]
            end = start + length
            if end <= individual_size:
                mask[i, start:end] = True
            else:
                mask[i, start:] = True
                mask[i, :end - individual_size] = True
        
        return mask