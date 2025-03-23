import torch
import numpy as np
import copy
# from random import sample
# import random
class Population():
    # deal with the population and cost (cost was received from the environment)
    # deal with the archive
    # deal with the history
    # deal with the generation
    # adjusted by the environment
    
    def __init__(self, pop_size, dim, max_G):
        '''
        :param pop_size:
        :param dim:
        :param max_G:
        :param problem:
        '''
        self.pop_size = pop_size
        self.dim = dim
        self.max_G = max_G
        self.generation = 0
        # self.history_vector = []
        # self.history_fitness = []

        # todo JADE part
        self.JADE_A = [] # store select fail vector, size = pop_size
        self.MadDE_A = [] # store select fail vector, size = pop_size * 2.3
        self.HARD_B = [] #store select fail vector, size = pop_size*rhar, rhar=3
        # self.init_population()      
        self.archive = []


    
    def init_population(self, rand_vector, c_cost_1):
        self.generation = 1

        self.current_vector = rand_vector
        # self.prefential_vector = np.empty_like(prefential_vector)

        # mask = c_cost_1 < c_cost_2
        # self.current_vector[mask] = rand_vector[mask]
        # self.prefential_vector[mask] = prefential_vector[mask]
        # self.current_vector[~mask] = prefential_vector[~mask]
        # self.prefential_vector[~mask] = rand_vector[~mask]

        self.current_fitness = c_cost_1
        # self.prefential_fitness = c_cost_2
        # self.history_vector.append(self.current_vector)
        # self.history_fitness.append(self.current_fitness)

        self.gbest_val = np.min(self.current_fitness)
        self.gbest_index = np.argmin(self.current_fitness)
        self.gbest_vector = self.current_vector[self.gbest_index]



    # def get_feature(self, fe):
    #     return fe(self.current_vector, self.current_fitness)

    def update_population(self):
        self.generation+=1
        # self.history_vector.append(self.current_vector)
        # self.history_fitness.append(self.current_fitness)
        # update gbest
        # because of greedy the pop have last generation gbest information

        self.gbest_val = np.min(self.current_fitness)
        self.gbest_index = np.argmin(self.current_fitness)
        self.gbest_vector = self.current_vector[self.gbest_index]
        # self.update_JADE_A()
        self.update_MadDE_A()
        # self.update_HARD_B()
        self.update_archive()
        
    def calculate_result(self):
        average_fitness = np.mean(self.current_fitness)
        best_fitness = np.min(self.current_fitness)
        return average_fitness,best_fitness

    def copy(self):
        return copy.deepcopy(self)

    # def update_JADE_A(self):
        # while(len(self.JADE_A) > self.pop_size):
            # index = random.randint(0, len(self.JADE_A) - 1)
            # self.JADE_A.pop(index)
            # pass

    def update_MadDE_A(self):
        while (len(self.MadDE_A) > 2.3 * self.pop_size):
            index = np.random.randint(0, len(self.MadDE_A) - 1)
            self.MadDE_A.pop(index)
            pass

    # def update_HARD_B(self):
        # while(len(self.HARD_B) > self.pop_size*3):
            # index = random.randint(0, len(self.HARD_B) - 1)
            # self.HARD_B.pop(index)
            # pass
# todo:combine

    def update_archive(self):
        while(len(self.archive)>self.pop_size*3):
            index = np.random.randint(0, len(self.archive) - 1)
            self.archive.pop(index)
            pass