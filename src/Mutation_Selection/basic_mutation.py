import numpy as np


class basic_mutation:
    """
    This class represents a basic mutation.
    Methods:
    - get_parameters_numbers: Returns the number of parameters.
    - mutation: Performs the mutation.
    """
    # individual version
    # def mutation(self,env,individual_indice):
    #     """
    #     Perform mutation on the given individual.
    #     Parameters:
    #     - env: The environment object.
    #     - individual_indice: The index of the individual to mutate.
    #     Returns:
    #     - None
    #     """
        
    #     pass
    
    # population version
    def mutation(self,env,pop_index,parameters):
        """
        Perform a mutation operation on a population.
        Args:
            env: The environment in which the mutation is performed.
            pop_index: The index of the population to mutate.
            parameters: Additional parameters required for the mutation process.
        Returns:
            None
        """
        pass
    
    
    def get_parameters_numbers(self):
        """
        Returns the number of parameters.
        """
        # todo: considering the type of parameters(discrete or continuous) or the number of parameters(maximum?)
        #       and the range of parameters 
        
        pass
    
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
    
    def construct_sub_current_fittest(self,env,individual_indice):
        population=env.population.current_vector
        fitness=env.population.current_fitness
        selected_fitness = np.array([fitness[i] for i in individual_indice])
        return selected_fitness
    
    def construct_random_indices(self, env, sub_pop_size, x_num):
        indices = np.zeros((sub_pop_size, x_num), dtype=int)
        for i in range(sub_pop_size):
            indices[i] = np.random.choice(env.population.pop_size, x_num, replace=False)
        return indices

    def construct_archive_indices(self, env, sub_pop_size, x_num):
        Len = env.archive.pop_size + env.population.pop_size
        indices = np.zeros((sub_pop_size, x_num), dtype=int)
        for i in range(sub_pop_size):
            indices[i] = np.random.choice(Len, x_num, replace=False)
        return indices
    
    
    def re_boudary(self,env,sub_pop):
        # print('!!!!re_boudary!!!!')
        lb=env.problem.lb
        ub=env.problem.ub
        range=ub-lb
        below_idx=sub_pop<lb
        above_idx=sub_pop>ub
        sub_pop[below_idx]=lb+np.random.rand()*range
        sub_pop[above_idx]=ub-np.random.rand()*range
        return sub_pop
    
    def construct_qbest(self,env,parameters):
        """
        Constructs the qbest vector.
        Args:
            env (object): The environment object containing the population.
        Returns:
            numpy.ndarray: The qbest vector.
        """
        population_object=env.population
        population=population_object.current_vector
        Len=population_object.pop_size
        q=parameters[:,1]
        if np.all(q == 0):
            # print('q.empty')
            q = np.random.uniform(low=0.01, high=1.0, size=q.shape)
        else:
            q_mean = np.mean(q)  
            q[q == 0] = q_mean  
                   
        group_sizes=np.ceil(Len*q).astype(int)
        group_sizes=np.minimum(group_sizes,Len)
        # print('q:',q)
        # print('group_sizes:',group_sizes)
        
        group_best_indice=[]
        all_group_indices = [np.random.choice(Len, size, replace=False) for size in group_sizes]
        # 但是这样就会每个个体都要进行一次群体选择
        group_best_indice = [indices[np.argmin(population_object.current_fitness[indices])] for indices in all_group_indices]
        # print('group_best_indice:',group_best_indice)
        return population[group_best_indice]
    