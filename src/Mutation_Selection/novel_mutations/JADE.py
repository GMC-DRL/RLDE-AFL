from ..basic_mutation import basic_mutation
import numpy as np
from scipy.stats import cauchy
class JADE(basic_mutation):
    def __init__(self):
        self.S_f = []
        self.S_cr = []
        self.p = 0.05
        self.c = 0.1
        self.u_cr = 0.5
        self.u_f = 0.5

    def reset(self):
        self.S_f = []
        self.S_cr = []

    def get_parameters_numbers(self):
        return 2
    
    def choose_p_best(self, current_vector, current_fitness):
        pop_size, _ = current_vector.shape
        choose_num = int(pop_size * self.p)
        sorted_indices = np.argsort(current_fitness)
        top_p_indicies = sorted_indices[:choose_num]
        random_best_index = np.random.choice(top_p_indicies)
        return random_best_index

    def origin_mutation(self, i, population_object):
        pop_size = population_object.pop_size
        current_vector = population_object.current_vector
        current_i_vector = current_vector[i]
        idx_x_best_p = self.choose_p_best(current_vector, population_object.current_fitness)
        x_best_p = current_vector[idx_x_best_p]
        idxs = [idx for idx in range(pop_size) if idx != i]
        x_r1, x_r2 = current_vector[np.random.choice(idxs, 2, replace = False)] # 抽取两个下标 replace 不放回
        idx_x_r2 = np.random.randint(0, pop_size + len(population_object.JADE_A) - 3)
        if idx_x_r2 >= (pop_size - 2):
            x_r2 = population_object.JADE_A[idx_x_r2 - pop_size + 2]
        F_i = cauchy.rvs(loc = self.u_f, scale = 0.1) # 柯西分布随机数
        while F_i < 0 or F_i > 1:
            if F_i < 0:
                F_i = cauchy.rvs(loc = self.u_f, scale = 0.1)
            else:
                F_i = 1
        self.F_i = F_i
        mutated_vector = current_i_vector + F_i * (x_best_p - current_i_vector) + F_i * (x_r1 - x_r2)
        # 越界处理
        lb = population_object.problem.lb
        ub = population_object.problem.ub
        for mutated_vector_i in range(len(mutated_vector)):
            if mutated_vector[mutated_vector_i] < lb:
                mutated_vector[mutated_vector_i] = (current_i_vector[mutated_vector_i] + lb) / 2
                pass
            elif mutated_vector[mutated_vector_i] > ub:
                mutated_vector[mutated_vector_i] = (current_i_vector[mutated_vector_i] + ub) / 2
                pass
            pass
        return mutated_vector

    def origin_crossover(self, i, population_object, mutated_vector):
        # CR_i = np.clip(random.gauss(self.u_cr, 0.1), 0, 1) # 限制0-1
        CR_i = np.clip(np.random.normal(self.u_cr, 0.1), 0, 1)
        self.CR_i = CR_i
        current_i_vector = population_object.current_vector[i]
        dim = len(mutated_vector)
        cross_points = np.random.rand(dim) < CR_i
        cross_points[np.random.randint(0, dim)] = True
        crossover_vector = np.where(cross_points, mutated_vector, current_i_vector)
        return crossover_vector

    def origin_select(self, i, population_object, crossover_vector):
        cost_c = population_object.get_individual_costs(crossover_vector)
        current_fitness = population_object.current_fitness
        if cost_c < current_fitness[i]:
            population_object.JADE_A.append(population_object.current_vector[i])
            population_object.current_vector[i] = crossover_vector
            population_object.current_fitness[i] = cost_c
            self.S_cr.append(self.CR_i)
            self.S_f.append(self.F_i)

    def origin_update(self):
        c = self.c
        if self.S_cr:
            self.u_cr = (1 - c) * self.u_cr + c * np.mean(self.S_cr)
            self.u_f = (1 - c) * self.u_f + c * (sum(ff ** 2 for ff in self.S_f) / sum(self.S_f))

    # def mutation(self, env, i):
    #     """
    #     Perform mutation operation on an individual in the population.
    #     Args:
    #         env (Environment): The environment object containing the population and problem information.
    #         i (int): The index of the individual to mutate.
    #     Returns:
    #         numpy.ndarray: The mutated vector representing the individual.
    #     """
        
    #     population_object = env.population
    #     parameters = env.action['mutation_parameters']
        
    #     # 这里的 parameters 分配的是 F_i 和 CR_i
    #     # 如果只是mutation的话, CR还得在crossover分发
    #     self.F_i = parameters[0] # todo 这里 parameters 参数还未确定，以后得转换下标
    #     # self.CR_i = parameters[1]

    #     pop_size = population_object.pop_size
    #     current_vector = population_object.current_vector
    #     current_i_vector = current_vector[i]
    #     idx_x_best_p = self.choose_p_best(current_vector, population_object.current_fitness)
    #     x_best_p = current_vector[idx_x_best_p]
    #     idxs = [idx for idx in range(pop_size) if idx != i]
    #     x_r1, x_r2 = current_vector[np.random.choice(idxs, 2, replace = False)]
    #     idx_x_r2 = random.randint(0, pop_size + len(population_object.JADE_A) - 3)
    #     if idx_x_r2 >= (pop_size - 2):
    #         x_r2 = population_object.JADE_A[idx_x_r2 - pop_size + 2]
    #     F_i = self.F_i
    #     # while F_i < 0 or F_i > 1:
    #     #     if F_i < 0:
    #     #         F_i = cauchy.rvs(loc = self.u_f, scale = 0.1)
    #     #     else:
    #     #         F_i = 1
    #     mutated_vector = current_i_vector + F_i * (x_best_p - current_i_vector) + F_i * (x_r1 - x_r2)
    #     # 越界处理
    #     lb = env.problem.lb
    #     ub = env.problem.ub
    #     for mutated_vector_i in range(len(mutated_vector)):
    #         if mutated_vector[mutated_vector_i] < lb:
    #             mutated_vector[mutated_vector_i] = (mutated_vector[mutated_vector_i] + lb) / 2
    #             pass
    #         elif mutated_vector[mutated_vector_i] > ub:
    #             mutated_vector[mutated_vector_i] = (mutated_vector[mutated_vector_i] + ub) / 2
    #             pass
    #         pass
    #     return mutated_vector
    def construct_random_indices(self, env,indices, sub_pop_size, x_num):
        population_size = env.population.pop_size
        random_indices = np.zeros((sub_pop_size, x_num), dtype=int)
        
        for i in range(sub_pop_size):
            available_indices = list(set(range(population_size)) - {indices[i]})
            # print('available_indices',available_indices)
            random_indices[i] = np.random.choice(available_indices, x_num, replace=False)
        
        return random_indices
    
    def construct_archive_indices(self, env,random_indices,indices, sub_pop_size, x_num):
        Len = len(env.population.archive)+ env.population.pop_size
        population_size = env.population.pop_size
        archive_indices = np.zeros((sub_pop_size, x_num), dtype=int)
        random_indices=random_indices.T[0]
        # print('random_indices',random_indices)
        for i in range(sub_pop_size):
            available_indices = list(set(range(Len)) - {indices[i]} - {random_indices[i]} )
            archive_indices[i] = np.random.choice(available_indices, x_num, replace=False)
        
        return archive_indices
    
    # population version
    def mutation(self,env,indices,parameters):
        # todo: x1 和 x2 的选择有误
        population_object=env.population
        population=population_object.current_vector
        sub_pop_size=len(indices)
        sub_pop=self.construct_sub_vector(env,indices)
        F=parameters[:,0]
        F = F[:, np.newaxis]
        best_individual_indice=np.argmin(population_object.current_fitness)
        best_individual=population[best_individual_indice]
        x_best_p=self.construct_qbest(env,parameters)
        random_indices=self.construct_random_indices(env,indices,sub_pop_size,1)
        x1=population[random_indices.T][0]    
        archive_indices=self.construct_archive_indices(env,random_indices,indices,sub_pop_size,1)
        x2=self.construct_archive_pop(env,indices,archive_indices)[0]
        # print('subpop.shape',sub_pop.shape)
        # print('xbest:',x_best_p.shape)
       
        mutated_vector=sub_pop+F*(x_best_p-sub_pop)+F*(x1-x2) # is it correct?
        
        # print('mutated_vector.shape',mutated_vector.shape)
        mutated_vector=self.re_boudary(env,mutated_vector)
        return mutated_vector
        
    # def construct_XpbestG(self,env):
    #     """
    #     Constructs the XpbestG vector, which is the best individual from a randomly selected subset of the population.
    #     Args:
    #         env: An environment object that contains the current population and their fitness values.
    #     Returns:
    #         XpbestG: The best individual vector from the selected subset of the population.
    #     """
        
    #     p=0.15
    #     # todo:modified by network
    #     population = env.population.current_vector
    #     num_to_select = int(p * env.population.pop_size)
    #     selected_indices = np.random.choice(env.population.pop_size, num_to_select, replace=False)
    #     selected_individuals = env.population.current_vector[selected_indices]
    #     selected_fitness = env.population.current_fitness[selected_indices]
    #     best_individual_index = selected_indices[np.argmin(selected_fitness)]
    #     XpbestG = population[best_individual_index]
    #     return XpbestG 


    def construct_archive_pop(self,env,indices,chosen):
        archive = np.array(env.population.archive)
        if archive.size == 0:
            PUA = env.population.current_vector
        else:
            PUA = np.vstack((env.population.current_vector, env.population.archive))
        archive_pop_size = len(indices)
        
        # random_indice=np.random.choice(len(PUA), archive_pop_size, replace=False)
        return PUA[chosen]
        
        
        
    def crossover(self, population_object, i, mutated_vector):
        CR_i = self.CR_i
        current_i_vector = population_object.current_vector[i]
        dim = len(mutated_vector)
        cross_points = np.random.rand(dim) < CR_i
        cross_points[np.random.randint(0, dim)] = True
        crossover_vector = np.where(cross_points, mutated_vector, current_i_vector)
        return crossover_vector

    def select(self, population_object, i, crossover_vector):
        cost_c = population_object.get_individual_costs(crossover_vector)
        current_fitness = population_object.current_fitness
        if cost_c < current_fitness[i]:
            population_object.JADE_A.append(population_object.current_vector[i])
            population_object.current_vector[i] = crossover_vector
            population_object.current_fitness[i] = cost_c
            # 不需要存CR 和 F 因为u_cr 和 u_f 都由网络给出
            # self.S_cr.append(self.CR_i)
            # self.S_f.append(self.F_i)

