'''
mutation_5
implement rand_1
'''
from ..basic_mutation import basic_mutation
import numpy as np

class pro_rand_1(basic_mutation):

    def get_parameters_numbers(self):
        return 1

    def EuclideanDistance(self, x, y):
        '''
        Calculates the Euclidean distance between two vectors.

        :param x: Vector x (numpy array)
        :param y: Vector y (numpy array)
        :return: The Euclidean distance between x and y
        '''
        return np.sqrt(np.sum(np.square(x - y)))

    def cal_R_d(self, population_object):
        '''
        Calculates the Euclidean distance matrix for the current population.

        :param population_object: Population class object containing current population information
        :return: R_d (numpy array) The Euclidean distance matrix of size (pop_size, pop_size)
        '''
        current_vector = population_object.current_vector
        pop_size = population_object.pop_size
        R_d = np.zeros((pop_size, pop_size))
        for i in range(pop_size):
            for j in range(i + 1, pop_size):
                R_d[i, j] = self.EuclideanDistance(current_vector[i], current_vector[j])
                R_d[j, i] = R_d[i, j]
        return R_d

    def cal_R_i_d(self, population_object, i):
        '''
        Calculates the Euclidean distance between a specific individual and all others in the population.

        :param population_object: Population class object containing current population information
        :param i: Index of the individual in the population
        :return: R_i_d (numpy array) The distance vector of the individual to all others in the population
        '''
        current_vector = population_object.current_vector
        pop_size = population_object.pop_size
        R_i_d = np.zeros(pop_size)
        for j in range(pop_size):
            if i != j:
                R_i_d[j] = self.EuclideanDistance(current_vector[i], current_vector[j])

        return R_i_d

    def cal_R_i_p(self, R_i_d):
        '''
        Calculates the probability vector for an individual based on its distance to others.

        :param R_i_d: Distance vector of the individual to all others in the population
        :return: R_i_p (numpy array) The probability vector, where each element represents the probability of being selected
        '''
        R_i_p = np.zeros_like(R_i_d)
        Sum = np.sum(R_i_d)
        for j, value in enumerate(R_i_d):
            R_i_p[j] = 1 - (value / Sum)
        return R_i_p
    def cal_R_p(self, R_d):
        '''
        Calculates the probability matrix for the population based on the Euclidean distance matrix.

        :param R_d: Euclidean distance matrix of the population
        :return: R_p (numpy array) The probability matrix, where each element represents the probability of being selected
        '''
        R_p = np.zeros_like(R_d)
        Sum = np.sum(R_d, axis=1,keepdims=True)
        R_p = 1 - (R_d / Sum)
        # print('R_p shape:',R_p.shape)
        return R_p
    
    def roulette_wheel_selection(self, R_i_p, i):
        '''
        Performs roulette wheel selection to choose three individuals from the population, excluding the individual at index i.

        :param R_i_p: Probability vector for the individual based on its distance to others
        :param i: Index of the individual to exclude from selection
        :return: List of three selected indices
        '''
        R_i_p = np.delete(R_i_p, i) # Remove the probability of the individual at index i
        R_i_p /= R_i_p.sum() # Normalize the remaining probabilities
        rand_nums = np.random.rand(3) # Generate three random numbers for selection
        cumulative_probs = np.cumsum(R_i_p) # Compute cumulative probabilities
        selected_indices = np.searchsorted(cumulative_probs, rand_nums)  # Find indices corresponding to random numbers
        selected_indices = [idx if idx < i else idx + 1 for idx in selected_indices] # Adjust indices to account for deletion

        return selected_indices

    def roulette_wheel_selection(self, R_p,selection_size):
        np.fill_diagonal(R_p,0)
        # print('R_p.shape:',R_p.shape)
        R_p /= R_p.sum(axis=1,keepdims=True)
        cumulative_probs=np.cumsum(R_p,axis=1)
        rand_nums=np.random.rand(R_p.shape[0],selection_size)
        # print('cumulative.shape:',cumulative_probs.shape)
        # print('rand_nums.shape:',rand_nums.shape)
        # selected_indices=np.searchsorted(cumulative_probs,rand_nums)
        selected_indices=np.zeros((R_p.shape[0],selection_size),dtype=int)
        for i in range(len(selected_indices)):
            selected_indices[i] = np.searchsorted(cumulative_probs[i], rand_nums[i])
            selected_indices[i]=[idx if idx < i else idx + 1 for idx in selected_indices[i]]
            # 为什么会越界，我写错了什么
            selected_indices[i] = np.clip(selected_indices[i], 0, R_p.shape[0] - 1) 
        return selected_indices
        
    def construct_r(self, env,indices):
        population_object=env.population    
        R_d=self.cal_R_d(population_object)
        R_p=self.cal_R_p(R_d)
        selected_indices=self.roulette_wheel_selection(R_p,3)
        return selected_indices
        
        
    # def mutation(self, env, i):
    #     '''
    #     Performs mutation operation to create a new mutated vector for the individual at index i.

    #     :param parameters: Mutation parameters, typically including the scaling factor F
    #     :param population_object: Population class object containing current population information
    #     :param i: Index of the individual to be mutated
    #     :return: The mutated vector for the individual at index i
    #     '''
    #     population_object=env.population
    #     parameters=env.action['mutation_parameters']
    #     R_i_d = self.cal_R_i_d(population_object, i)
    #     R_i_p = self.cal_R_i_p(R_i_d)
    #     r1, r2, r3 = self.roulette_wheel_selection(R_i_p, i)
    #     F = parameters[0]
    #     current_vector = population_object.current_vector
    #     mutated_vector = current_vector[r1] + F * (current_vector[r2] - current_vector[r3])
    #     return mutated_vector

    def mutation(self,env,indices,parameters):
        population_object=env.population
        population=population_object.current_vector
        R_indices=self.construct_r(env,indices)
        # print('R_indices.shape:',R_indices.shape)
        r1, r2, r3 = [population[R_indices[indices, i]] for i in range(3)]
        F=parameters[:,0]
        F = F[:, np.newaxis]
        mutated_vector=r1+F*(r2-r3)
        # mutated_vector=mutated_vector[indices]
        mutated_vector=self.re_boudary(env,mutated_vector)
        return mutated_vector

    def crossover(self, population_object, i, mutated_vector, CR = 0.9):
        '''
        Performs crossover operation to combine the mutated vector with the current individual's vector.

        :param population_object: Population class object containing current population information
        :param i: Index of the individual undergoing crossover
        :param mutated_vector: The mutated vector for the individual
        :param CR: Crossover rate, determining the probability of crossover at each dimension (default is 0.9)
        :return: The resulting vector after crossover
        '''
        current_i_vector = population_object.current_vector[i]
        dim = len(mutated_vector)
        cross_points = np.random.rand(dim) < CR
        cross_points[np.random.randint(0, dim)] = True
        crossover_vector = np.where(cross_points, mutated_vector, current_i_vector)
        return crossover_vector

    def select(self, population_object, i, crossover_vector):
        '''
        Performs selection operation to determine if the new individual should replace the current one.

        :param population_object: Population class object containing current population information
        :param i: Index of the individual undergoing selection
        :param crossover_vector: The vector resulting from crossover
        '''
        cost_c = population_object.get_individual_costs(crossover_vector)
        current_fitness = population_object.current_fitness
        if cost_c < current_fitness[i]:
            #population_object.JADE_A.append(population_object.current_vector[i])  # Add the old individual to archive for JADE
            population_object.current_vector[i] = crossover_vector
            population_object.current_fitness[i] = cost_c



