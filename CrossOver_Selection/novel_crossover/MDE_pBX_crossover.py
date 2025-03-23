from ..basic_crossover import basic_crossover
import math
import numpy as np

class MDE_pBX(basic_crossover):
    # seems like there is a problem with it 
    # the avg_gitness is similar to best_fitness
    # and the SARMrand1 will compare the history best1 and history best2
    # and it will be so similar ,which could cause divide 0 problem
    """
    This class represents the MDE_pBX crossover.
    Methods:
    - get_parameters_numbers: Returns the number of parameters.
    - crossover: Performs the crossover.
    """
    
    def get_parameters_numbers(self):
        """
        Returns the number of parameters.
        1. CR (float): The crossover probability.
        2.p :randomly choose p top individuals from the population
        """
        return 2
    
    
    def crossover(self,parameters,population_class,crossover_indice,mutated_individual):
        """
        Perform crossover operation between a selected individual and a mutated individual.
        Args:
            parameters (list): List of parameters.
            population_class (Population): Instance of the Population class.
            crossover_indice (int): Index of the crossover operation.
            mutated_individual (ndarray): Mutated individual.
        Returns:
            ndarray: Crossover individual resulting from the operation.
        """
        crossover_parameters = parameters[0]
        
        Np=population_class.pop_size
        G=population_class.generation
        Gmax=population_class.max_generation
        population=population_class.current_vector
        fitness = population_class.current_fitness
        # An origin implementation of MDE_pBX crossover
        # which would be replaced 
        # p = parameters[1]
        
        p= math.ceil((Np/2)*(1-((G-1)/Gmax)))
        # print('G:',G)
        # print('Gmax:',Gmax)
        # print('Np:',Np)
        # print('p:',p)
        # Sort the population based on fitness and select the top p individuals
        top_indices = np.argsort(fitness)[:p]
        top_individuals = population[top_indices]
        selected_top_individual = top_individuals[np.random.randint(0, p)]
        crossover_individual = np.empty_like(mutated_individual)
        # crossover
        crossover_individual = np.where(np.random.rand(population_class.dim) < population_class.Cr,
                                selected_top_individual,
                                mutated_individual)
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
        p=0.5
        sub_origin_pop=self.construct_sub_origin_vector(env, crossover_indice)        
        sub_mutated_pop=self.construct_sub_mutated_vector(crossover_indice,mutated_pop)
        individual_size=env.problem.dim
        sub_pop_size=len(crossover_indice)
        
        top_individuals=self.construct_top_individuals(env,sub_pop_size,p)
        
        mask = np.random.rand(sub_pop_size, individual_size) < crossover_parameter
        
        updated_pop = np.where(mask, top_individuals, sub_mutated_pop)      
        return updated_pop
        
    def construct_top_individuals(self,env,sub_pop_len,p):
        p= 15
        top_indices = np.argsort(env.population.current_fitness)[:p]
        top_individuals = env.population.current_vector[top_indices]
        chosen_individuals = np.empty((sub_pop_len, env.problem.dim))
        random_indices = np.random.randint(0, p, size=sub_pop_len)
        chosen_individuals = top_individuals[random_indices]
        return chosen_individuals