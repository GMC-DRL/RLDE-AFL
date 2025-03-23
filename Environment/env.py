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
        self.fes = 0
        self.g = 0

        pop_size = self.pop_size
        dim = self.dim
        max_G = self.max_G
        self.population = Population(pop_size, dim, max_G)
        rand_vector = np.random.uniform(low = self.problem.lb,
                                        high = self.problem.ub,
                                        size = (pop_size, dim))

        c_cost = self.get_costs(self.problem, rand_vector)

        self.population.init_population(rand_vector, c_cost)

        self.action = {}


        # drawing preparation
        self.avg_fitness_list = []
        self.best_fitness_list = []

        self.no_improve = 0
        self.per_no_imporve = np.zeros_like(c_cost)
        self.max_dist = np.sqrt((self.problem.ub - self.problem.lb) ** 2 * self.dim)
        # log
        self.log_index = 1
        self.log_cost = [self.population.gbest_val]
        self.init_gbest = self.population.gbest_val

        return self.get_pop_state()


    def get_pop_state(self):
        xs = (self.population.current_vector - self.problem.lb) / (self.problem.ub - self.problem.lb)
        fes = self.fes / self.max_fes
        pop = {'x': xs[None, :],
               'y': self.population.current_fitness[None, :],
               'fes': np.array([[fes]])}
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


        pre_gbest = self.population.gbest_val

        self.update(mutation_operator, mutation_parameters, crossover_operator, crossover_parameters)
        avg_fitness, best_fitness = self.population.calculate_result()


        if self.fes >= self.log_index * self.log_interval:
            self.log_index += 1
            self.log_cost.append(self.population.gbest_val)

        if self.problem.optimum is None:
            is_done = self.fes >= self.max_fes
        else:
            is_done = self.fes >= self.max_fes or self.population.gbest_val <= 1e-8


        if self.population.gbest_val < pre_gbest:
            self.no_improve = 0
        else:
            self.no_improve += 1

        reward = (pre_gbest - self.population.gbest_val) / self.init_gbest
        if self.__config.reward_ratio > 0:
            reward = self.__config.reward_ratio * reward

        self.g += 1
        # todo: fitness difference value normalization-->reward

        if is_done:
            if len(self.log_cost) >= self.__config.n_logpoint + 1:
                self.log_cost[-1] = self.population.gbest_val
            else:
                self.log_cost.append(self.population.gbest_val)

        return self.get_pop_state(), reward, is_done

    def classification_pop(self, act_operators):
        """
        Classifies the population according to the given operators.
        Args:
            act_operators (list): A list of operators used to classify the population.
        Returns:
            dict: A dictionary where keys are operators and values are lists of indices
                  from the act_operators list that correspond to each operator.
        """

        # classify the population according to the operators
        operators_dict = {}
        for idx, de in enumerate(act_operators):
            if de not in operators_dict:
                operators_dict[de] = []
            operators_dict[de].append(idx)
        return operators_dict

    def apply_mutation(self, classified_indices, configs):
        # print('mutation')

        # classified_indices: dict: {de: [indices]}
        origin_pop = self.population.current_vector
        updated_pop = np.zeros_like(origin_pop)
        # print(self.action)
        for de in classified_indices:
            indices = classified_indices[de]
            parameters = configs[indices]
            operator = self.mutation_selector.select_mutation_operator(int(de))
            # print('operator:',operator)
            # print('indices_len:',len(indices))
            updated_sub_pop = operator.mutation(self, indices, parameters)

            updated_pop[indices] = updated_sub_pop
            # print('after:',updated_sub_pop.shape[0])
        return updated_pop

    def apply_crossover(self, classified_indices, configs, mutated_pop):
        # print('crossover')
        origin_pop = self.population.current_vector
        updated_pop = np.zeros_like(origin_pop)
        for de in classified_indices:
            indices = classified_indices[de]
            parameters = configs[indices]
            operator = self.crossover_selector.select_crossover_operator(int(de))
            # print('operator:',operator)
            # print('indices_len:',len(indices))
            updated_sub_pop = operator.crossover(self, indices, mutated_pop, parameters)
            updated_pop[indices] = updated_sub_pop
            # print('after:',updated_sub_pop.shape[0])
        # operator=self.crossover_selector.select_crossover_operator(0)
        # updated_pop=operator.crossover(self, classified_indices, mutated_pop)
        return updated_pop

    def update(self, mutation_list, mutation_parameters, crossover_list, crossover_parameters):
        # print('start update')

        mutation_classified_pop = self.classification_pop(mutation_list)
        mutated_pop = self.apply_mutation(mutation_classified_pop, mutation_parameters)
        # print('mutated_pop:',mutated_pop.shape)

        crossover_classified_pop = self.classification_pop(crossover_list)
        crossover_pop = self.apply_crossover(crossover_classified_pop, crossover_parameters, mutated_pop)
        # print('crossover_pop:',crossover_pop.shape)
        updated_pop = greedy_select().select(self, crossover_pop)

        # print('update finish')
        # print('update',updated_pop.shape)
        self.population.update_population()

    def get_history(self):
        return self.population.history_vector, self.population.history_fitness

    def get_sub_pop_cost(self, x):
        self.fes += len(x)
        problem = self.problem
        if problem.optimum is None:
            out = problem.eval(x)
        else:
            out = problem.eval(x) - problem.optimum
        return out

    def get_individual_costs(self, x):
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


