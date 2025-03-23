from ..basic_mutation import basic_mutation
import numpy as np
import math
'''
A new mutation method based on TopoMut
'''
class TopoMut_DE(basic_mutation):
    def __init__(self) -> None:
        self.NP = 100
        self.F = 0.5 # // not need
        self.CR = 0.9
        self.k = 10 # number of nearest neighbors // not need
        self.TMP = 0.25 # the probability to do topo-mut // not need

    def get_parameters_numbers(self):
        """
        Returns the number of parameters.
        1.k: number of nearest neighbors
        2.TMP: the probability to do topo-mut

        ATTENTION: 
        the paper gives the k and TMP,where k = NP/10,TMP is adaptive
        """
        return 2
    
    # temporarily not used
    def Topograph(self,population) -> None:
        """
        generate a kNN matrix indicating the nearest neighbors of each individual
        """
        current_vector = population.current_vector
        pop_size, dim = current_vector.shape
        # generate N*N distance matrix
        distance_matrix = np.zeros((pop_size,pop_size))
        for i in range(pop_size):
            for j in range(i+1,pop_size):
                distance_matrix[i,j] = np.linalg.norm(current_vector[i]-current_vector[j])
                distance_matrix[j,i] = distance_matrix[i,j]
            distance_matrix[i,i] = np.inf
        # generate kNN matrix
        k = pop_size//10 # number of nearest neighbors
        kNN_matrix = np.zeros((pop_size,k))
        for i in range(pop_size):
            kNN_matrix[i] = np.argsort(distance_matrix[i])[:k]
        for i in range(pop_size):
            for j in range(k):
                if population.current_fitness[i] < population.current_fitness[int(kNN_matrix[i,j])]:
                    kNN_matrix[i,j] = kNN_matrix[i,j]
                else:
                    kNN_matrix[i,j] = -kNN_matrix[i,j]
        self.kNN_matrix = kNN_matrix

    # to do 
    # when to run Topograph()?
    # def mutation(self,env,individual_indice):
    
    #     """
    #     Perform mutation on an individual in the population.
    #     Args:
    #         env (Environment): The environment object containing the population.
    #         individual_indice (int): The index of the individual to mutate.
    #     Returns:
    #         numpy.ndarray: The mutated individual.
    #     Raises:
    #         None
    #     """
        
    #     population = env.population
    #     parameters = env.action['mutation_parameters']
    #     # linear 
    #     self.TMP = env.fes/env.max_fes
    #     # exponenial 
    #     # self.TMP = 0.1* math.exp(population.fes/population.max_fes)
    #     F = parameters[0]
    #     Len = population.pop_size
    #     indices = np.random.choice(range(Len), 3, replace = False)
    #     x1, x2, x3 = population.current_vector[[indices[0],indices[1],indices[2]]]
    #     # do topo-mut with probability TMP
    #     if np.random.rand() < self.TMP:
    #         # choose the nearest best neighbor of indice
    #         self.Topograph(population)
    #         info = self.kNN_matrix[individual_indice]
    #         flag = False
    #         for idx in range(len(info)): #len = k
    #             if info[idx] < 0:
    #                 if flag == False:
    #                     flag = True
    #                     purpose = idx
    #                 elif population.current_fitness[int(info[idx])] < population.current_fitness[int(info[purpose])]:
    #                     purpose = idx
    #         if flag == False: # no better neighbors
    #             purpose = individual_indice
    #         x1 = population.current_vector[purpose]
    #     new_individual = x1 + F * (x2 - x3)
    #     return new_individual
    
    
    def mutation(self,env,indices,parameters):
        population_object=env.population
        Len=population_object.pop_size
        population=population_object.current_vector
        sub_pop_size=len(indices)
        sub_pop=self.construct_sub_vector(env,indices)
        # parameters=env.action['mutation_parameters']
        F=parameters[:,0]
        F = F[:, np.newaxis]
        random_indices=self.construct_random_indices(env,sub_pop_size,2)
        x2,x3=population[random_indices.T]
        self.Topograph(population_object)
        info=self.kNN_matrix
                
        flag=np.zeros(info.shape[0],dtype=bool)
        
        negative_indices= info<0
        
        purpose=np.arange(info.shape[0])
        
        for i in range(Len):
            if np.any(negative_indices[i]):
                flag[i] = True
                valid_indices = np.where(negative_indices[i])[0]
                fitness_values = population_object.current_fitness[info[i, valid_indices].astype(int)]
                purpose[i] = valid_indices[np.argmin(fitness_values)]
        
        # 如果没有更好的邻居，保持当前个体的索引
        purpose[~flag]=np.arange(info.shape[0])[~flag]
        
        topu=population[purpose]
        x1=topu[indices]
        
        mutated_vector=x1+F*(x2-x3)
        mutated_vector=self.re_boudary(env,mutated_vector)
        return mutated_vector

    
if __name__ == "__main__":
    A = TopoMut_DE()
    print("hello")