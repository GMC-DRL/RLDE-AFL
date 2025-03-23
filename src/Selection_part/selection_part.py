# a temporary greedy selection for rand 1
# only used in prelimary test

# idea: what if we use other selection strategies ?
import numpy as np



class greedy_select:
    def __init__(self):
        pass
    
    # individual version
    def select(self,env,judge_indice,crossover_individual):
        """
        Selects an individual based on its fitness score and the crossover individual.
        Args:
            env (Environment): The environment object.
            judge_indice (int): The index of the individual to be judged.
            crossover_individual (Individual): The individual resulting from crossover.
        Returns:
            Individual: The selected individual.
        """
        
        population_object = env.population
        
        origin_individual = population_object.current_vector[judge_indice]
        score_previous=population_object.current_fitness[judge_indice]
        score_target= env.get_individual_costs(crossover_individual)
        
        if score_previous < score_target:
            # have compared in prefential_cross, so the update is for attemp 2
            if score_target < population_object.prefential_fitness[judge_indice]:
                population_object.prefential_vector[judge_indice] = crossover_individual
            return origin_individual
            
        else:
            population_object.JADE_A.append(origin_individual)
            population_object.HARD_B.append(origin_individual)
            
            return crossover_individual
    
    # population version
    def select(self, env, crossover_pop):
        """
        Selects individuals from the crossover population based on their fitness scores
        compared to the current population. Updates the current population and fitness
        scores accordingly.
        Args:
            env (Environment): The environment object containing the population and methods
                       to evaluate individual fitness.
            crossover_pop (ndarray): The population generated from crossover operations.
        Returns:
            None
        Updates:
            - population_object.current_vector: The selected population based on fitness scores.
            - population_object.current_fitness: The updated fitness scores of the selected population.
            - population_object.JADE_A: List updated with individuals from the current population
                        that were not replaced.
            - population_object.HARD_B: List updated with individuals from the current population
                        that were not replaced.
        """
    
        population_object = env.population
        scores_previous = population_object.current_fitness
        # print('scores_previous:',scores_previous.shape)
        scores_target = env.get_costs(env.problem,crossover_pop)
        # print('scores_target:',scores_target.shape)

        improve_filters = scores_target < scores_previous
        env.per_no_imporve += 1
        tmp = np.where(improve_filters, env.per_no_imporve, np.zeros_like(env.per_no_imporve))
        env.per_no_imporve -= tmp

        scores_target = scores_target.flatten()
        
        # Select the population based on fitness scores
        selected_population = np.where(scores_previous[:, np.newaxis] < scores_target[:, np.newaxis], 
                                       population_object.current_vector, 
                                       crossover_pop)


        # Update JADE_A and HARD_B lists
        for i in range(population_object.pop_size):
            if scores_previous[i] >= scores_target[i]:
                
                # population_object.JADE_A.append(population_object.current_vector[i])
                # population_object.HARD_B.append(population_object.current_vector[i])
                population_object.archive.append(population_object.current_vector[i])


        population_object.current_vector = selected_population
        population_object.current_fitness = np.where(scores_previous < scores_target, 
                                                     scores_previous, 
                                                     scores_target)
        return selected_population