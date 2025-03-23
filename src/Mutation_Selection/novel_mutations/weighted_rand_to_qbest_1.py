from ..basic_mutation import basic_mutation
import numpy as np
from scipy.stats import cauchy
from scipy.special import softmax

class weighted_rand_to_qbest_1(basic_mutation):
    def get_parameters_numbers(self):
        return 3
    # F q fas
    
    def mutation(self,env,pop_index,parameters):
        NP=len(pop_index)
        best=self.construct_qbest(env,parameters)
        sub_pop=self.construct_sub_vector(env,pop_index)
        NB=best.shape[0]
        rb = np.random.randint(NB, size=NP)
        duplicate = np.where(rb == np.arange(NP))[0]
        count=0
        while duplicate.shape[0] > 0 and count < 25:
            rb[duplicate] = np.random.randint(NB, size=duplicate.shape[0])
            duplicate = np.where(rb == np.arange(NP))[0]
            count += 1

        count = 0
        r1 = np.random.randint(NP, size=NP)
        duplicate = np.where((r1 == rb) + (r1 == np.arange(NP)))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r1[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r1 == rb) + (r1 == np.arange(NP)))[0]
            count += 1

        count = 0
        r2 = np.random.randint(NP, size=NP)
        duplicate = np.where((r2 == rb) + (r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r2 == rb) + (r2 == np.arange(NP)) + (r2 == r1))[0]
            count += 1
        Fas=parameters[:,2]
        xb = best[rb]
        x1 = sub_pop[r1]
        x2 = sub_pop[r2]
        F=parameters[:,0]
        F = F[:, np.newaxis]
        Fas=Fas[:, np.newaxis]
        # print('F.shape:',F.shape)
        # print('Fas.shape',Fas.shape)
        
        v = F * x1 + F * Fas * (xb - x2)
        v = self.re_boudary(env,v)
        return v
        
        
    # def construct_qbest(self,env,parameters):
    #     """
    #     Constructs the qbest vector.
    #     Args:
    #         env (object): The environment object containing the population.
    #     Returns:
    #         numpy.ndarray: The qbest vector.
    #     """
    #     population_object=env.population
    #     population=population_object.current_vector
    #     Len=population_object.pop_size
    #     q=parameters[:,1]
    #     group_sizes=np.ceil(Len*q).astype(int)
    #     group_sizes=np.minimum(group_sizes,Len)
    #     # 然后怎么处理？
    #     # group_size=int(Len*q)
    #     group_best_indice=[]
    #     all_group_indices = [np.random.choice(Len, size, replace=False) for size in group_sizes]
    #     # 但是这样就会每个个体都要进行一次群体选择
    #     group_best_indice = [indices[np.argmin(population_object.current_fitness[indices])] for indices in all_group_indices]
    #     return population[group_best_indice]
    