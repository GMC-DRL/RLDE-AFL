from ..basic_mutation import basic_mutation
import numpy as np

# there seems to waste some evaluate source......
# in SA_RM
# that we need to evaluate the new_cost and RM_cost which may be used in the selection

class SA_RM_best_1(basic_mutation):
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
        # the possible reason why rand1 didn't have this problem is 
        # that best1 mutated based on the best individual 
        # so the new individual will differ from the origin individual very little
        # 
        # didn't we consider the divide 0 problem,as the q_last_best_1 may equal q_last_best_2?
        # a possible bugs
        # it really happened in the test
        # the reason is that as the generation increases, the difference between the best and the second best is decreasing
        #
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

    def RM(self, env, new_individual):
        """
        Random Mutation (RM) method for mutation selection.
        Args:
            env (Environment): The environment object.
            new_individual (ndarray): The new individual to be evaluated.
        Returns:
            ndarray: The selected individual after mutation.
        """
        
        population_object=env.population
        new_cost = env.get_individual_costs(new_individual)
        
        RM_x = np.random.uniform(low=env.problem.lb, high=env.problem.ub, size=population_object.dim)
        
        RM_cost= env.get_individual_costs(RM_x)
        
        if new_cost<RM_cost:                
            return new_individual  
        else:
            return RM_x
        
    def mutation(self, env,individual_indice,parameters):
        
        """
        Perform mutation on an individual in the population.
        Parameters:
        - env: The environment object containing the population and mutation parameters.
        - individual_indice: The index of the individual to mutate.
        Returns:
        - new_individual: The mutated individual.
        """
        
        population_object=env.population
        # parameters=env.action['mutation_parameters']
        
        # new_F=self.self_adaptive_factors(parameters,population_object,individual_indice)
        new_F=parameters[:,0]
        new_F = new_F[:, np.newaxis]
        population=population_object.current_vector
        best_individual=np.min(population_object.current_fitness)
        # print('best_individual',best_individual)
        best_individual_indice=np.argmin(population_object.current_fitness)
        # print('best_individual_indice',best_individual_indice)        
        Len=population_object.pop_size
        indices = np.random.choice(range(Len), 2,replace=False)
        x1, x2= population[indices[0]], population[indices[1]]
        mutated_vector =population[best_individual_indice]+new_F*(x1-x2)
        new_vector=self.RM(env,mutated_vector)  
        new_vector=self.re_boudary(env,new_vector)
        return new_vector