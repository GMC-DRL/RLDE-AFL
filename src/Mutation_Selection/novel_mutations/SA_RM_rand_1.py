from ..basic_mutation import basic_mutation
import numpy as np
# there seems to waste some evaluate source......
# in SA_RM
# that we need to evaluate the new_cost and RM_cost which may be used in the selection



class SA_RM_rand_1(basic_mutation):
    def get_parameters_numbers(self):
        # F--scaling factor 0<=F<=1
        return 1
    def cal_FQ(self,parameters,population_object,individual_indice):
        """
        Calculate the FQ (Fitness Quality) value for a given individual.
        Parameters:
        - parameters: The parameters used for the calculation.
        - population_object: The population class containing the history and fitness values.
        - individual_indice: The index of the individual for which to calculate the FQ.
        Returns:
        - FQ: The calculated FQ value.
        Note:
        - The FQ value is calculated based on the fitness history of the population.
        - The calculation involves finding the minimum fitness values and performing mathematical operations.
        - A constant value C is used in the calculation.
        - The FQ value is a measure of the quality of the individual's fitness.
        """
        C=10
        # to ensure 0<fq<1
        Q1_vec=population_object.history_fitness[0]
        Q1_best=Q1_vec.min()
        Q_last_vec=population_object.history_fitness[-1]
        Q_last_best_1 = Q_last_vec.min()
        
        sorted_vec = np.sort(Q_last_vec)
        Q_last_best_2 = sorted_vec[1] if len(sorted_vec) > 1 else None
        i=1
        while Q_last_best_2==Q_last_best_1 and i<population_object.pop_size-1:
            i+=1
            Q_last_best_2=sorted_vec[i]
        # print('qlastbest1',Q_last_best_1)
        # print('qlastbest2',Q_last_best_2)
        # didn't we consider the divide 0 problem,as the q_last_best_1 may equal q_last_best_2?
        # a possible bugs
        # a simple solution is to add a constant 
        
        FQ=1/(np.log10((C*Q1_best)/abs(Q_last_best_1-Q_last_best_2)))
        return FQ
    
    def cal_F(self,parameters,population_object,individual_indice):
        F=0.5*(np.cos((population_object.generation/population_object.max_G)*np.pi)+1)
        
        return F
    
    def self_adaptive_factors(self,parameters,population_object,individual_indice):
        """
        Calculate the self-adaptive factors for mutation selection.
        this is SA
        Parameters:
        - parameters: The parameters used for calculating the factors.
        - population_object: The population class object.
        - individual_indice: The index of the individual.
        Returns:
        - new_F: The calculated self-adaptive factor.
        """
        FQ=self.cal_FQ(parameters,population_object,individual_indice) 
        F=self.cal_F(parameters,population_object,individual_indice)
        new_F=FQ*F*np.random.rand()
        return new_F

    def RM(self, env, new_vector):
        """
        Applies the RM mutation operator to generate a new individual.
        Parameters:
        - population_object: An instance of the population class.
        - new_individual: The individual to be mutated.
        Returns:
        - The mutated individual.
        Description:
        The RM mutation operator randomly generates a new individual by sampling values from a uniform distribution
        within the lower and upper bounds of the problem. The cost of the new individual is compared to the cost of
        the randomly generated individual (RM_x). If the cost of the new individual is lower, the RM_x is returned.
        Otherwise, the new_individual is returned unchanged.
        """
        new_cost = env.get_sub_pop_cost(new_vector)
        new_average_cost = new_cost.mean()
        population_object=env.population
        RM_x = np.random.uniform(low=env.problem.lb, high=env.problem.ub, size=new_vector.shape)
        
        RM_cost= env.get_sub_pop_cost(RM_x)
        RM_average_cost=RM_cost.mean()
        if new_average_cost<RM_average_cost:                
            return new_vector  
        else:
            return RM_x
        
    # def mutation(self, env,individual_indice):
    
    #     """
    #     Perform mutation on an individual in the population.
    #     Args:
    #         env (Environment): The environment object containing the population.
    #         individual_indice (int): The index of the individual to mutate.
    #     Returns:
    #         Individual: The mutated individual.
    #     Raises:
    #         None
    #     """
        
    #     population_object=env.population
    #     parameters=env.action['mutation_parameters']
    #     new_F=self.self_adaptive_factors(parameters,population_object,individual_indice)
    #     population=population_object.current_vector
    #     Len=population_object.pop_size
    #     indices = random.sample(range(Len), 2)
    #     x1, x2 = population[indices[0]], population[indices[1]]
    #     mutated_vector=population[individual_indice]+new_F*(x1-x2)
    #     new_individual=self.RM(env,mutated_vector)        

    #     return new_individual
    
    # population version
    def mutation(self,env,indices,parameters):
        population_object=env.population
        population=population_object.current_vector
        # parameters=env.action['mutation_parameters']
        # F=self.self_adaptive_factors(parameters,population_object,indices)
        F=parameters[:,0]
        F = F[:, np.newaxis]
        
        sub_pop=self.construct_sub_vector(env,indices)
        Len=len(indices)
        random_indices= self.construct_random_indices(env,len(indices),2)
        x1,x2=population[random_indices.T]
        mutated_vector=sub_pop+F*(x1-x2)
        re_mutated_vector=self.RM(env,mutated_vector)
        re_mutated_vector=self.re_boudary(env,re_mutated_vector) 
        return re_mutated_vector