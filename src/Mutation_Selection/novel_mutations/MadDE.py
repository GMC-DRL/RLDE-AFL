from ..basic_mutation import basic_mutation
import numpy as np
from scipy.stats import cauchy
from scipy.special import softmax

# 拆了
class MadDE(basic_mutation):
    def __init__(self):
        self.npm = 2
        self.Nmin = 4

        self.m = 3
        self.pm = np.ones(self.m) / self.m
        self.p = 0.18
        self.PqBX = 0.01
        
    def get_parameters_numbers(self):
        return 2
    
    def __sort(self, current_fitness):
        ind = np.argsort(current_fitness) # 取前面最好Np个的下标
        return ind

    def cal_np(self, env):
        self.dim = env.dim
        self.Nmax = self.npm * self.dim * self.dim
        self.Np =int(np.round(self.Nmax + (self.Nmin - self.Nmax) * env.fes / env.max_fes))

    # DE/current-to-pbest/1 + archive
    def ctb_w_arc(self, group, best, archive, Fs):
        """
        Generate a new population of candidate solutions using the current population, 
        the best solutions, and an archive of past solutions.
        Parameters:
        group (numpy.ndarray): The current population of candidate solutions with shape (NP, dim).
        best (numpy.ndarray): The best solutions found so far with shape (NB, dim).
        archive (numpy.ndarray): An archive of past solutions with shape (NA, dim).
        Fs (float): Scaling factor for the mutation.
        Returns:
        numpy.ndarray: A new population of candidate solutions with the same shape as `group`.
        """
        
        NP, dim = group.shape
        NB = best.shape[0]
        NA = len(archive)

        # 选pbest 下标
        count = 0
        rb = np.random.randint(NB, size = NP)
        duplicate = np.where(rb == np.arange(NP))[0]
        while duplicate.shape[0] > 0 and count < 25:
            rb[duplicate] = np.random.randint(NB, size = duplicate.shape[0])
            duplicate = np.where(rb == np.arange(NP))[0]
            count += 1

        # 选随机下标r1
        count = 0
        r1 = np.random.randint(NP, size = NP)
        duplicate = np.where((r1 == rb) + (r1 == np.arange(NP)))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r1[duplicate] = np.random.randint(NP, size = duplicate.shape[0])
            duplicate = np.where((r1 == rb) + (r1 == np.arange(NP)))[0]
            count += 1

        # 选随机下标r2
        count = 0
        r2 = np.random.randint(NP + NA, size = NP)
        duplicate = np.where((r2 == rb) + (r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = np.random.randint(NP + NA, size = duplicate.shape[0])
            duplicate = np.where((r2 == rb) + (r2 == np.arange(NP)) + (r2 == r1))[0]
            count += 1

        xb = best[rb]
        x1 = group[r1]
        if NA > 0:
            x2 = np.concatenate((group, archive), 0)[r2]
        else:
            x2 = group[r2]
        Fs = Fs[:, np.newaxis]
        v = group + Fs * (xb - group) + Fs * (x1 - x2)

        return v

    # DE/currentto-rand/1  archive
    def ctr_w_arc(self, group, archive, Fs):
        NP, dim = group.shape
        NA = len(archive)

        count = 0
        r1 = np.random.randint(NP, size=NP)
        duplicate = np.where((r1 == np.arange(NP)))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r1[duplicate] = np.random.randint(NP, size=duplicate.shape[0])
            duplicate = np.where((r1 == np.arange(NP)))[0]
            count += 1

        count = 0
        r2 = np.random.randint(NP + NA, size=NP)
        duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
        while duplicate.shape[0] > 0 and count < 25:
            r2[duplicate] = np.random.randint(NP + NA, size=duplicate.shape[0])
            duplicate = np.where((r2 == np.arange(NP)) + (r2 == r1))[0]
            count += 1

        x1 = group[r1]
        if NA > 0:
            x2 = np.concatenate((group, archive), 0)[r2]
        else:
            x2 = group[r2]
        Fs = Fs[:, np.newaxis]
        v = group + Fs * (x1 - x2)

        return v

    # DE/weighted-rand-to-qbest/1
    def weighted_rtb(self, group, best, Fs, Fas):
        NP, dim = group.shape
        NB = best.shape[0]

        count = 0
        rb = np.random.randint(NB, size=NP)
        duplicate = np.where(rb == np.arange(NP))[0]
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

        xb = best[rb]
        x1 = group[r1]
        x2 = group[r2]
        Fs = Fs[:, np.newaxis]
        v = Fs * x1 + Fs * Fas * (xb - x2)

        return v

    #这里直接进来，不需要对每个i遍历
    def mutation(self,env,pop_index,parameters):
        #self.cal_np(population) # 先计算当前的MadDE种群大小
        population_object=env.population
        
        F = parameters[:,0]
        F = F[:, np.newaxis]
        pm = parameters[:,1]
        pm=softmax(pm)
        pm_mean=np.mean(pm)
        # pm=np.ones(3)*pm
        pm_dirichlet = np.random.dirichlet([pm_mean, pm_mean, pm_mean], size=1)
        # print('pm:',pm_dirichlet)
        self.pm = pm_dirichlet.flatten()
        # is it correct to use the method to prepare the pm? 
        population = population_object.current_vector
        
        sub_pop=self.construct_sub_vector(env,pop_index)
        sub_current_fitness=self.construct_sub_current_fittest(env,pop_index)
        
        parameters=env.action['mutation_parameters']
        dim = population_object.dim
        NP = len(pop_index)
        # print('index:',pop_index)
        # print('F:',F)
        ind = self.__sort(sub_current_fitness)
        # F = parameters[0] # todo 这里待定 应该是NP个F_i
        q = 2 * self.p - self.p * env.fes / env.max_fes
        Fa = 0.5 + 0.5 * env.fes / env.max_fes 
        # 得到需要进行操作的vector
        current_vector_np = sub_pop[ind]

        # 选择策略
        mu = np.random.choice(3, size = NP, p = self.pm)
        p1 = current_vector_np[mu == 0] # 第一种策略的变异向量
        p2 = current_vector_np[mu == 1] # 2
        p3 = current_vector_np[mu == 2] # 3

        pbest = current_vector_np[:max(int(self.p * NP), 2)]
        qbest = current_vector_np[:max(int(q * NP), 2)]
        
        Fs = F.T[0] # NP * dim
    
        # todo archive 待定
        # 三种策略变异
        # print('Fs:',Fs)
        # print('mu:',mu)
        v1 = self.ctb_w_arc(p1, pbest, population_object.archive, Fs[mu == 0])
        v2 = self.ctr_w_arc(p2, population_object.archive, Fs[mu == 1])
        v3 = self.weighted_rtb(p3, qbest, Fs[mu == 2], Fa)
        v = np.zeros((NP, dim))
        v[mu == 0] = v1
        v[mu == 1] = v2
        v[mu == 2] = v3

        # 越界处理
        mutated_vector=self.re_boudary(env,v)
        return mutated_vector # new_NP * dim

    # def mutation(self,parameters,population,individual_indice):
    #     dim = population.dim
    #     NP = population.pop_size

    #     current_vector = population.current_vector
    #     current_i_vector = current_vector[individual_indice]
    #     F = parameters[0] # todo 这里待定 一个F_i
    #     q = 2 * self.p - self.p * population.fes / population.max_fes
    #     Fa = 0.5 + 0.5 * population.fes / population.max_fes

    #     # 选择策略
    #     mu = np.random.choice(3, size = 1, p = self.pm)
    #     # mu == 0 DE/current-to-pbest/1 + archive
    #     # mu == 1 DE/currentto-rand/1  archive
    #     # mu == 2 DE/weighted-rand-to-qbest/1

    #     # 得到 pbest 和 qbest
    #     ind = self.__sort(population.current_fitness)
    #     current_best_vector = current_vector[ind]
    #     pbest = current_best_vector[:max(int(self.p * NP), 2)]
    #     qbest = current_best_vector[:max(int(q * NP), 2)]

    #     if mu == 0:
    #         # DE/current-to-pbest/1 + archive
    #         archive = np.array(population.MadDE_A)
    #         v = self.ctb_w_arc(current_i_vector[None, :], pbest, archive, F)
    #     elif mu == 1:
    #         # DE/currentto-rand/1  archive
    #         archive = np.array(population.MadDE_A)
    #         v = self.ctr_w_arc(current_i_vector[None, :], archive, F)
    #     else:
    #         # DE/weighted-rand-to-qbest/1
    #         v = self.weighted_rtb(current_i_vector[None, :], qbest, F, Fa)
    #     # 越界处理
    #     v = v.reshape(-1)
    #     lb = population.problem.lb
    #     ub = population.problem.ub
    #     for mutated_vector_i in range(len(v)):
    #         if v[mutated_vector_i] < lb:
    #             v[mutated_vector_i] = (v[mutated_vector_i] + lb) / 2
    #             pass
    #         elif v[mutated_vector_i] > ub:
    #             v[mutated_vector_i] = (v[mutated_vector_i] + ub) / 2
    #             pass
    #         pass
    #     return v


    # `def mutation(self,env,indices):
    #     population_object=env.population
    #     population=population_object.current_vector
    #     sub_pop=self.construct_sub_vector(env,indices)
    #     parameters=env.action['mutation_parameters']
    #     F=parameters[0]
    #     Fa=0.3
    #     xpbestG=self.construct_pbest(env)
    #     xqbest=self.construct_qbest(env)
        
    #     strategy_vector=self.construct_strategy_vector(len(indices))
    #     distributed_population=self.distribute(strategy_vector,sub_pop)        
    #     mutated_vec=np.zeros_like(sub_pop)
        
        
    #     if 1 in distributed_population:
    #         vector1 = np.array(distributed_population[1])
            
    #         v1 = vector1 + F * (xpbestG - vector1) + F * (x1 - x3)
    #         mutated_vec[strategy_vector == 1] = v1

    #     if 2 in distributed_population:
    #         vector2 = np.array(distributed_population[2])
    #         v2 = vector2 + F * (x1 - x3)
    #         mutated_vec[strategy_vector == 2] = v2

    #     if 3 in distributed_population:
    #         vector3 = np.array(distributed_population[3])
    #         v3 = F * x1 + F*Fa * (xpbestG - x2)
    #         mutated_vec[strategy_vector == 3] = v3
    
    #     return mutated_vec
    #     # 未完成
    #     # 这样的话我还要重新再将他们分类然后放进去
    
    def construct_strategy_vector(pop_size):
        strategies=np.random.choice([0,1,2],size=pop_size)
        return strategies
    
    def distribute(self, strategy_vector, population):
        """
        Distributes the population based on the strategy vector.
        Args:
            strategy_vector (numpy.ndarray): The strategy vector.
            population (numpy.ndarray): The population to distribute.
        Returns:
            dict: The distributed population as a dictionary with keys as strategy identifiers and values as lists of individuals.
        """
        distributed_population = {1: [], 2: [], 3: []}
        
        for i in range(len(strategy_vector)):
            strategy = strategy_vector[i]
            if strategy in distributed_population:
                distributed_population[strategy].append(population[i])
            else:
                raise ValueError(f"Invalid strategy {strategy} in strategy_vector.")
        
        return distributed_population
        
    def construct_pbest(self,env):
        """
        Constructs the pbest vector.
        Args:
            env (object): The environment object containing the population.
        Returns:
            numpy.ndarray: The pbest vector.
        """
        population_object=env.population
        population=population_object.current_vector
        Len=population_object.pop_size
        p=0.15
        num_to_select = int(p * len(population))
        selected_indices = np.random.choice(len(population), num_to_select, replace=False)
        selected_individuals = population[selected_indices]
        selected_fitness = population_object.current_fitness[selected_indices]
        best_individual_index = selected_indices[np.argmin(selected_fitness)]
        XpbestG = population[best_individual_index]
        return XpbestG
    
    def construct_qbest(self,env):
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
        q=  15*0.01
        group_size=int(Len*q)
        group_indices=np.random.choice(Len,group_size,replace=False)
        group_fitness=population_object.current_fitness[group_indices]
        group_best_indice=group_indices[np.argmin(group_fitness)]
        return population[group_best_indice]
    
    # todo: combine two functions into one
    
    # todo: move to base class
    def construct_archive_pop(self,env,indices):
        
        JADE_A = np.array(env.population.JADE_A)
        MAD_A = np.array(env.population.MadDE_A)
        if MAD_A.size == 0:
            PUB = env.population.current_vector
        else:
            PUB = np.vstack((env.population.current_vector, env.population.MadDE_A))
            
        archive_pop_size = len(indices)
        random_indice=np.random.choice(len(PUB), archive_pop_size, replace=False)
        
        return PUB[random_indice]
    
    