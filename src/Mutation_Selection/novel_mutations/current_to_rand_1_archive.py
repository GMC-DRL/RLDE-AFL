from ..basic_mutation import basic_mutation
import numpy as np
from scipy.stats import cauchy
from scipy.special import softmax

class current_to_rand_1_archive(basic_mutation):
    def get_parameters_numbers(self):
        return 1
    
    def mutation(self, env, pop_index,parameters):
        # print('current_to_rand_1_archive')
        population_object = env.population 
        population = population_object.current_vector
        archive = population_object.archive
        
        F=parameters[:,0]
        F=F[:,np.newaxis]
        
        sub_pop = self.construct_sub_vector(env, pop_index)
        sub_current_fitness = self.construct_sub_current_fittest(env, pop_index)
        NP=len(pop_index)
        archive_size=len(archive)
        # random_indices=self.construct_random_indices(env,pop_index,len(pop_index),1)
        # x1=population[random_indices.T[0]]
        # archive_indices=self.construct_archive_indices(env,random_indices.T[0],pop_index,len(pop_index),1)
        # # print('archive_indices:',archive_indices)
        # x2=self.construct_archive_pop(env,pop_index,archive_indices)[0]
        count = 0
        r1 = np.random.randint(NP, size=NP)
        duplicate = np.where((r1 == np.arange(NP)))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r1[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r1 == np.arange(NP)))[0]
            count += 1

        count = 0
        r2 = np.random.randint(NP + archive_size, size=NP)
        duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = np.random.randint(NP + archive_size, size=duplicate.shape[0])
            duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
            count += 1

        x1 = sub_pop[r1]
        if archive_size > 0:
            x2 = np.concatenate((sub_pop, archive), 0)[r2]
        else:
            x2 = sub_pop[r2]
        # Fs = Fs[:, np.newaxis]
        v = sub_pop + F * (x1 - x2)
        # print('sub_pop.shape:',sub_pop.shape)
        # print('v.shape:',v.shape)
        v=self.re_boudary(env,v)
        return v
        
    def construct_random_indices(self, env,indices, sub_pop_size, x_num):
        population_size = env.population.pop_size
        random_indices = np.zeros((sub_pop_size, x_num), dtype=int)
        
        for i in range(sub_pop_size):
            available_indices = list(set(range(population_size)) - {indices[i]})
            # print('available_indices',available_indices)
            random_indices[i] = np.random.choice(available_indices, x_num, replace=False)
        
        return random_indices
    def construct_archive_pop(self,env,indices,chosen):
        archive = np.array(env.population.archive)
        if archive.size == 0:
            PUA = env.population.current_vector
        else:
            PUA = np.vstack((env.population.current_vector, env.population.archive))
        archive_pop_size = len(indices)
        
        # random_indice=np.random.choice(len(PUA), archive_pop_size, replace=False)
        return PUA[chosen]
    def construct_archive_indices(self, env,random_indices,indices, sub_pop_size, x_num):
        Len = len(env.population.archive)+ env.population.pop_size
        population_size = env.population.pop_size
        archive_indices = np.zeros((sub_pop_size, x_num), dtype=int)
        random_indices=random_indices
        # print('random_indices',random_indices)
        for i in range(sub_pop_size):
            available_indices = list(set(range(Len)) - {indices[i]} - {random_indices[i]} )
            archive_indices[i] = np.random.choice(available_indices, x_num, replace=False)
        
        return archive_indices
# # DE/currentto-rand/1  archive
# def ctr_w_arc(self, group, archive, Fs):
#     NP, dim = group.shape
#     NA = len(archive)
#     count = 0
#     r1 = np.random.randint(NP, size=NP)
#     duplicate = np.where((r1 == np.arange(NP)))[0]
#     while duplicate.shape[0] > 0 and count < 25:
#         r1[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
#         duplicate = np.where((r1 == np.arange(NP)))[0]
#         count += 1
#     count = 0
#     r2 = np.random.randint(NP + NA, size=NP)
#     duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
#     while duplicate.shape[0] > 0 and count < 25:
#         r2[duplicate] = np.random.randint(NP + NA, size=duplicate.shape[0])
#         duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
#         count += 1
#     x1 = group[r1]
#     if NA > 0:
#         x2 = np.concatenate((group, archive), 0)[r2]
#     else:
#         x2 = group[r2]
#     Fs = Fs[:, np.newaxis]
#     v = group + Fs * (x1 - x2)
#     return v