import numpy as np

class basic_crossover:
    def get_parameters_numbers(self):
        """
        Returns the number of parameters.
        """
        pass
    
    def crossover(self,parameters,population_class,crossover_indice,mutated_individual):
        """
        Performs crossover operation on the population.
        Args:
            parameters (dict): A dictionary containing the parameters for the crossover operation.
            population_class (class): The class representing the population.
            crossover_indice (int): The index of the individual to perform crossover with.
            mutated_individual (object): The individual that has undergone mutation.
        Returns:
            None
        """
        
        pass
    
    def crossover(self,env,sub_pop,mutated_sub_pop,parameters):
        
        
        pass
    
    
    def construct_sub_origin_vector(self,env,individual_indice):
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
    
    def construct_sub_mutated_vector(self,individual_indice,mutated_vector):
        """
        Constructs a mutated vector by selecting individuals from the population.
        Args:
            individual_indice (list of int): Indices of individuals to be selected from the population.
            mutated_vector (numpy.ndarray): The mutated vector.
        Returns:
            numpy.ndarray: An array of selected individuals from the population.
        """
        
        selected_mutated_pop = np.array([mutated_vector[i] for i in individual_indice])
        return selected_mutated_pop
    
    def construct_random_indices(self,env,sub_pop_size,x_num):
        indices= np.random.choice(env.population.pop_size,(sub_pop_size,x_num))
        return indices