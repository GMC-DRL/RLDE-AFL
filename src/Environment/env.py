from Population import Population
from Feature_Extractor import feature_extractor
from Mutation_Selection.Select import select_mutation as Select_Mutation
from CrossOver_Selection.Select import select_crossover
from Selection_part.selection_part import greedy_select
import numpy as np
import gymnasium as gym
# todo: inherit gym
class Env(gym.Env):
    # env contains 3 parts :
    #  an action 
    #  a problem
    #  a population
    # 
    # it will adjust the population according to the action and problem
    
    
    def __init__(self, problem, config):
        # 从 config 中提取参数
        self.__config = config
        self.pop_size = config.pop_size
        self.dim = problem.dim
        self.max_G = config.max_G
        self.max_fes = config.max_fes
        self.difficulty = config.difficulty

        self.action = {}
        self.fes = 0
        self.problem = problem
        self.mutation_selector = config.mutation_selector
        self.crossover_selector = config.crossover_selector

        # mutation 和 crossover 的 参数数量
        self.n_mutation = self.mutation_selector.n_mutation
        self.n_crossover = self.crossover_selector.n_crossover
        # self.action_space = gym.spaces.Discrete(2)  
        # self.observation_space = gym.spaces.Box(low=np.array([self.problem.lb] * self.dim),
                                                # high=np.array([self.problem.ub] * self.dim),
                                                # dtype=np.float32)

        # gleet
        self.no_improve = 0

        # log
        self.log_index = None
        self.log_cost = None
        self.log_interval = config.log_interval
        # problem.reset() # 放在外面
        self.reset()



    def reset(self):
        """
        Resets the environment with the given parameters.
        Parameters:
        - problem: The problem to be solved.
        - pop_size: The size of the population.
        - dim: The dimensionality of the problem.
        - max_G: The maximum number of generations.
        - max_fes: The maximum number of function evaluations.
        - difficulty: The difficulty level of the problem.
        Returns:
        None
        """
        
        # initialize the environment
        # env_config = {
            # 'pop_size': 50,
            # 'dim': 10,
            # 'max_G': 1000,
            # 'max_fes': 300000,
            # 'problem': f0
        # }
        # self.max_fes=max_fes
        self.fes = 0
        self.g = 0
        # # problem construct
        # self.problem = problem
        # self.difficulty = difficulty
        
        # population construct
        pop_size = self.pop_size
        dim = self.dim
        max_G = self.max_G
        self.population = Population(pop_size, dim, max_G )
        rand_vector = np.random.uniform(low = self.problem.lb,
                                        high = self.problem.ub,
                                        size = (pop_size, dim))
        # prefential_vector = np.random.uniform(low = self.problem.lb,
        #                                 high = self.problem.ub,
        #                                 size = (pop_size, dim))
        c_cost = self.get_costs(self.problem, rand_vector)
        # c_cost_pre = self.get_costs(self.problem, prefential_vector)
        self.population.init_population(rand_vector, c_cost)
        # #net initialize && feature construct
        # fe=feature_extractor.Feature_Extractor(attention_order = "individual") # is it correct to use individual?
        # feature=self.population.get_feature(fe)
    
        # operator construct
        # should be decided by the net
        self.action={}
        # # action contain the operators and parameters
        
        # mutation_operators = "rand_1"
        # action['mutation_operators'] = select_mutation.select_mutation_operator(mutation_operators)

        # crossover_operators = "simple_crossover"
        # action['crossover_operators'] = select_crossover.select_crossover_operator(crossover_operators)

        # configuration
        # should be decided by the net and feature
        # action['Mutation_parameters']=[0.5,0.5]
        # action['Crossover_parameters']=[0.5]
    
        # drawing preparation   
        self.avg_fitness_list = []
        self.best_fitness_list = []

        # gleet
        self.no_improve = 0
        self.per_no_imporve = np.zeros_like(c_cost)
        self.max_dist = np.sqrt((self.problem.ub - self.problem.lb) ** 2 * self.dim)
        # log
        self.log_index = 1
        self.log_cost = [self.population.gbest_val]
        self.init_gbest = self.population.gbest_val

        if self.__config.fe_gleet:
            return self.get_gleet_state()
        else:
            return self.get_pop_state()
        
        
        
    # def step(self, action):
    #     """
    #     Executes a single step of the environment.
    #     Parameters:
    #     - action (dict): A dictionary containing the mutation_operators and crossover_operators.
    #     Returns:
    #     None
    #     Raises:
    #     None
    #     Description:
    #     This method performs a single step of the environment. It takes an action as input, which is a dictionary containing the mutation_operators and crossover_operators. It iterates over the population and applies the mutation operator to each individual. Then, it applies the crossover operator to the mutated individual. Finally, it selects the individual using a greedy selection strategy and updates the population. The average fitness and best fitness of the population are calculated and stored in the avg_fitness_list and best_fitness_list respectively.
    #     """
    #     # a rough implementation of the step function
    #     # random select the action
    #     self.action=action
    #     Len=self.population.pop_size
    #     for i in range(Len):
    #         if 'mutation_operators' in action:
    #             # print('select_operator:',int(action['mutation_operators'][0][0][i]))
    #             mutation_operator = select_mutation.select_mutation_operator(int(action['mutation_operators'][0][0][i]))
    #             mutated_individual = mutation_operator.mutation(self, i)
    #         else:
    #             print("Mutation operator not found in action.")
    #         if 'crossover_operators' in action:
    #             crossover_operator = action['crossover_operators']
    #             crossover_individual = crossover_operator.crossover(self, i, mutated_individual)
    #         else:
    #             print("Crossover operator not found in action.")
    #         selector=greedy_select()
    #         selected_individual = selector.select(self, i, crossover_individual)

    #         self.population.current_vector[i] = selected_individual
    #         self.population.current_fitness[i] = self.get_individual_costs(selected_individual)
            
    #     self.population.update_population()
        
    #     avg_fitness, best_fitness = self.population.calculate_result()
    #     self.avg_fitness_list.append(avg_fitness)
    #     self.best_fitness_list.append(best_fitness)

    def get_gleet_state(self):
        max_step = self.max_fes // self.pop_size
        max_cost = self.population.gbest_val
        fea0 = self.population.current_fitness / max_cost

        # 因为每代贪心选择，所以最小的一定是历史全局最优
        fea1 = (self.population.current_fitness - max_cost) / max_cost

        # fea2 贪心选择 其实每一代都是pbest
        fea2 = np.zeros_like(self.population.current_fitness)

        fea3 = np.full(shape = (self.pop_size), fill_value = (self.max_fes - self.fes) / self.max_fes)

        # fea4 也是需要pbest
        fea4 = self.per_no_imporve / max_step

        fea5 = np.full(shape = (self.pop_size), fill_value = self.no_improve / max_step)

        fea6 = np.sqrt(np.sum((self.population.current_vector - np.expand_dims(self.population.gbest_vector, axis = 0)) ** 2, axis = -1)) / self.max_dist

        # fea7 也是 pbest
        fea7 = np.zeros_like(self.population.current_fitness)
        # fea8 也是 不需要
        randpar_cur_vec = np.zeros_like(self.population.current_vector)
        gbest_cur_vec = np.expand_dims(self.population.current_vector[self.population.gbest_index], axis = 0) - self.population.current_vector
        fea8 = np.sum(randpar_cur_vec * gbest_cur_vec, axis = -1) / ((np.sqrt(np.sum(randpar_cur_vec ** 2, axis = -1)) * np.sqrt(np.sum(gbest_cur_vec ** 2, axis = -1))) + 1e-5)
        fea8 = np.where(np.isnan(fea8), np.zeros_like(fea8), fea8)


        return np.concatenate((fea0[:,None],fea1[:,None],fea2[:,None],fea3[:,None],fea4[:,None],fea5[:,None],fea6[:,None],fea7[:,None],fea8[:,None]),axis=-1)[None, :]

    def get_pop_state(self):
        xs = (self.population.current_vector - self.problem.lb) / (self.problem.ub - self.problem.lb)
        pop = {'x' : xs[None, :],
               'y' : self.population.current_fitness[None, :]}
        return pop

    def step(self, action):
        """
        Perform a single step in the environment using the given action.
        Args:
            action (torch.Tensor): A tensor containing the actions to be performed. 
                                   The first column represents the mutation operator 
                                   indices, and the remaining columns represent the 
                                   configuration parameters.
        Returns:
            tuple: A tuple containing:
                - pop_state (Any): The current state of the population.
                - reward (int): The reward obtained from this step. It is 1 if the 
                                best fitness has improved or stayed the same, and -1 
                                otherwise.
                - is_done (bool): A boolean indicating whether the maximum number of 
                                  generations has been reached.
        """
        _, n_action = action.shape
        mutation_operator = action[:, 0]  # 1D : pop_size
        crossover_operator = action[:, 1]  # 1D : pop_size
        mutation_parameters = action[:, 2: 2 + self.n_mutation]  # 2D : pop_size * n_mutation
        crossover_parameters = action[:, -self.n_crossover]  # 2D : pop_size * n_crossover
        self.action['mutation_parameters'] = mutation_parameters
        # print('configs:',configs.shape)

        # todo 把这里接上就行了
        self.action['crossover_parameters'] = [0.5]
        Len = self.population.pop_size
        is_done = False

        pre_gbest = self.population.gbest_val
        # 应为一个update函数
        # for i in range(Len):
        #     if self.g >= self.max_G:
        #         is_done = True
        #         break

        #     self.action['mutation_parameters'] = configs[i, :]
        #     mutation_operator = select_mutation.select_mutation_operator(int(mutation_de[i]))
        #     mutated_individual = mutation_operator.mutation(self, i)

        #     crossover_individual = select_crossover.select_crossover_operator("binomial_crossover").crossover(self, i, mutated_individual)




        #     selector = greedy_select()
        #     select_individual = selector.select(self, i, crossover_individual)

        #     self.population.current_vector[i] = select_individual
        #     self.population.current_fitness[i] = self.get_individual_costs(select_individual)

        # self.population.update_population()
        self.update( mutation_operator, mutation_parameters,crossover_operator, crossover_parameters)
        avg_fitness, best_fitness = self.population.calculate_result()
        # if np.random.rand() < 0.1:
            # print('avg_fitness:',avg_fitness)
            # print('best_fitness:',best_/fitness)
        # 以上将被替换
        
        
        
        
        
        # 简单的reward设计
        # todo: fitness difference value normalization-->reward
        # try: 
        # reward = (best_fitness - self.best_fitness) / best_fitness ^^(or init_fitness)^^ 

        if self.fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.log_cost.append(self.population.gbest_val)

        if self.problem.optimum is None:
            is_done = self.fes >= self.max_fes
        else:
            is_done = self.fes >= self.max_fes or self.population.gbest_val <= 1e-8

        # reward = (self.population.gbest_val - pre_gbest) / self.population.gbest_val
        # reward = (pre_gbest - self.population.gbest_val)/pre_gbest

        # gleet
        if self.population.gbest_val < pre_gbest:
            self.no_improve = 0
        else:
            self.no_improve += 1

        reward = (pre_gbest - self.population.gbest_val) / self.init_gbest
        if self.__config.reward_ratio > 0:
            reward = self.__config.reward_ratio * reward
        # if self.population.gbest_val < pre_gbest:
        #     reward = 1
        # else:
        #     reward = -1
        self.g += 1
        # 简单的reward设计
        # todo: fitness difference value normalization-->reward
        # try: 
        # reward = (best_fitness - self.best_fitness) / best_fitness ^^(or init_fitness)^^ 
        # 结束进化，再存一次
        if is_done:
            if len(self.log_cost) >= self.__config.n_logpoint + 1:
                self.log_cost[-1] = self.population.gbest_val
            else:
                self.log_cost.append(self.population.gbest_val)

        if self.__config.fe_gleet:
            return self.get_gleet_state(), reward, is_done
        else:
            return self.get_pop_state(), reward, is_done

    def classification_pop(self,act_operators):
        """
        Classifies the population according to the given operators.
        Args:
            act_operators (list): A list of operators used to classify the population.
        Returns:
            dict: A dictionary where keys are operators and values are lists of indices 
                  from the act_operators list that correspond to each operator.
        """
        
        # classify the population according to the operators
        operators_dict={}
        for idx,de in enumerate(act_operators):
            if de not in operators_dict:
                operators_dict[de]=[]
            operators_dict[de].append(idx)
        return operators_dict
    
    
    
    def apply_mutation(self, classified_indices, configs):
        # print('mutation')
        
        # classified_indices: dict: {de: [indices]}
        origin_pop=self.population.current_vector
        updated_pop=np.zeros_like(origin_pop)
        # print(self.action)
        for de in classified_indices:
            indices = classified_indices[de]
            parameters = configs[indices]
            operator = self.mutation_selector.select_mutation_operator(int(de))
            # print('operator:',operator)
            # print('indices_len:',len(indices))
            updated_sub_pop = operator.mutation(self, indices,parameters)
            
            updated_pop[indices] = updated_sub_pop
            # print('after:',updated_sub_pop.shape[0])            
        return updated_pop
    
    def apply_crossover(self, classified_indices, configs, mutated_pop):
        # print('crossover')
        origin_pop=self.population.current_vector
        updated_pop=np.zeros_like(origin_pop)
        for de in classified_indices:
            indices= classified_indices[de]
            parameters=configs[indices]
            operator=self.crossover_selector.select_crossover_operator(int(de))
            # print('operator:',operator)
            # print('indices_len:',len(indices))
            updated_sub_pop=operator.crossover(self, indices, mutated_pop,parameters)
            updated_pop[indices]=updated_sub_pop
            # print('after:',updated_sub_pop.shape[0])            
        # operator=self.crossover_selector.select_crossover_operator(0)
        # updated_pop=operator.crossover(self, classified_indices, mutated_pop)
        return updated_pop
    
    def update(self, mutation_list, mutation_parameters, crossover_list, crossover_parameters):
        # print('start update')
        
        mutation_classified_pop=self.classification_pop(mutation_list)
        mutated_pop=self.apply_mutation(mutation_classified_pop, mutation_parameters)
        # print('mutated_pop:',mutated_pop.shape)
        
        crossover_classified_pop=self.classification_pop(crossover_list)
        crossover_pop=self.apply_crossover(crossover_classified_pop,crossover_parameters,mutated_pop)
        # print('crossover_pop:',crossover_pop.shape)
        updated_pop=greedy_select().select(self, crossover_pop)
        
        # print('update finish')
        # print('update',updated_pop.shape)
        self.population.update_population()


    def get_history(self):
        return self.population.history_vector, self.population.history_fitness

    # todo 这三函数其实都差不多，可以写成一个的
    def get_sub_pop_cost(self,x):
        self.fes += len(x)
        problem = self.problem
        if problem.optimum is None:
            out = problem.eval(x)
        else:
            out = problem.eval(x) - problem.optimum
        return out

    def get_individual_costs(self,x):
        self.fes += 1
        problem = self.problem
        if problem.optimum is None:
            out = problem.eval(x)
        else:
            out = problem.eval(x) - problem.optimum
        return out
    
    def get_costs(self, problem, x):
        self.fes += self.population.pop_size
        if problem.optimum is None:
            out = problem.eval(x)
        else:
            out = problem.eval(x) - problem.optimum
        return out
    
    
        