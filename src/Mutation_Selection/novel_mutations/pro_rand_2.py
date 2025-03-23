'''
mutation_5
if you want to know about code comments, refer to pro_rand_1.py
'''
from ..basic_mutation import basic_mutation
import numpy as np

class pro_rand_2(basic_mutation):


    def EuclideanDistance(self, x, y):
        return np.sqrt(np.sum(np.square(x - y)))

    def cal_R_d(self, population_object):
        current_vector = population_object.current_vector
        pop_size = population_object.pop_size
        R_d = np.zeros((pop_size, pop_size))
        for i in range(pop_size):
            for j in range(i + 1, pop_size):
                R_d[i, j] = self.EuclideanDistance(current_vector[i], current_vector[j])
                R_d[j, i] = R_d[i, j]
        return R_d

    def cal_R_i_d(self, population_object, i):
        current_vector = population_object.current_vector
        pop_size = population_object.pop_size
        R_i_d = np.zeros(pop_size)
        for j in range(pop_size):
            if i != j:
                R_i_d[j] = self.EuclideanDistance(current_vector[i], current_vector[j])
        return R_i_d

    def cal_R_p(self, R_d):
        '''
        Calculates the probability matrix for the population based on the Euclidean distance matrix.

        :param R_d: Euclidean distance matrix of the population
        :return: R_p (numpy array) The probability matrix, where each element represents the probability of being selected
        '''
        R_p = np.zeros_like(R_d)
        Sum = np.sum(R_d, axis=1,keepdims=True)
        R_p = 1 - (R_d / Sum)
        return R_p
    
    def cal_R_i_p(self, R_i_d):
        R_i_p = np.zeros_like(R_i_d)
        Sum = np.sum(R_i_d)
        for j, value in enumerate(R_i_d):
            R_i_p[j] = 1 - (value / Sum)
        return R_i_p

    def roulette_wheel_selection(self, R_i_p, i):
        R_i_p = np.delete(R_i_p, i) # Remove the probability of the individual at index i
        R_i_p /= R_i_p.sum() # Normalize the remaining probabilities
        rand_nums = np.random.rand(5) # Generate three random numbers for selection
        cumulative_probs = np.cumsum(R_i_p) # Compute cumulative probabilities
        selected_indices = np.searchsorted(cumulative_probs, rand_nums)  # Find indices corresponding to random numbers
        selected_indices = [idx if idx < i else idx + 1 for idx in selected_indices] # Adjust indices to account for deletion
        return selected_indices
    
    def roulette_wheel_selection(self, R_p,selection_size):
        R_p= np.fill_diagonal(R_p,0)
        R_p /= R_p.sum(axis=1,keepdims=True)
        cumulative_probs=np.cumsum(R_p,axis=1)
        rand_nums=np.random.rand(selection_size,R_p.shape[0])
        selected_indices=np.searchsorted(cumulative_probs,rand_nums)
        for i in range(len(selected_indices)):
            selected_indices=[idx if idx < i else idx + 1 for idx in selected_indices]
        return selected_indices
        
    def construct_r(self, env,indices):
        population_object=env.population    
        R_d=self.cal_R_d(population_object)
        R_p=self.cal_R_p(R_d)
        selected_indices=self.roulette_wheel_selection(R_p,5)
        return selected_indices
        
    # def mutation(self, env, i):
    #     population_object = env.population
    #     parameters = env.action['mutation_parameters']
    #     R_i_d = self.cal_R_i_d(population_object, i)
    #     R_i_p = self.cal_R_i_p(R_i_d)
    #     r1, r2, r3, r4, r5 = self.roulette_wheel_selection(R_i_p, i)
    #     F = parameters[0]
    #     current_vector = population_object.current_vector
    #     mutated_vector = current_vector[r1] + F * (current_vector[r2] - current_vector[r3]) + F * (current_vector[r4] - current_vector[r5])
    #     return mutated_vector

    def mutation(self,env,indices,parameters):
        population_object=env.population
        population=population_object.current_vector
        R_indices=self.construct_r(env,indices)
        r1,r2,r3,r4,r5=population[R_indices]
        F=parameters[:,0]
        F = F[:, np.newaxis]
        mutated_vector=r1+F*(r2-r3)+F*(r4-r5)
        mutated_vector=mutated_vector[indices]
        mutated_vector=self.re_boudary(env,mutated_vector)
        return mutated_vector
    
    def crossover(self, population_object, i, mutated_vector, CR = 0.9):
        current_i_vector = population_object.current_vector[i]
        dim = len(mutated_vector)
        cross_points = np.random.rand(dim) < CR
        cross_points[np.random.randint(0, dim)] = True
        crossover_vector = np.where(cross_points, mutated_vector, current_i_vector)
        return crossover_vector

    def select(self, population_object, i, crossover_vector):
        cost_c = population_object.get_individual_costs(crossover_vector)
        current_fitness = population_object.current_fitness
        if cost_c < current_fitness[i]:
            #population_object.JADE_A.append(population_object.current_vector[i])  # Add the old individual to archive for JADE
            population_object.current_vector[i] = crossover_vector
            population_object.current_fitness[i] = cost_c



