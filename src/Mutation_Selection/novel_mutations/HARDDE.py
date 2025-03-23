from ..basic_mutation import basic_mutation
import numpy as np
from scipy.stats import cauchy

class HARDDE(basic_mutation):
    def get_parameters_numbers(self):
        #F --scaling factor 0<=F<=1
        #F1 -- 0.9F
        #F2 -- 0.7F
        
        return 3

    # def mutation(self, env, individual_indice):
    #     """
    #     Perform mutation on an individual in the population.
    #     Args:
    #         env (Environment): The environment object containing the population and mutation parameters.
    #         individual_indice (int): The index of the individual to mutate.
    #     Returns:
    #         numpy.ndarray: The mutated individual.
    #     Raises:
    #         None
    #     """
        
    #     population_object = env.population
    #     parameters = env.action['mutation_parameters']
        
    #     Len=population_object.pop_size
    #     population=population_object.current_vector
        
    #     p=0.15
    #     num_to_select = int(p * len(population))
    
    #     selected_indices = np.random.choice(len(population), num_to_select, replace=False)
    #     selected_individuals = population[selected_indices]
    
    #     selected_fitness = population_object.current_fitness[selected_indices]
    #     best_individual_index = selected_indices[np.argmin(selected_fitness)]
    #     XpbestG = population[best_individual_index]
        
    #     JADE_A = np.array(population_object.JADE_A)
    #     HARD_B = np.array(population_object.HARD_B)
        
    #     if JADE_A.size == 0:
    #         PUA = population
    #     else:
    #         PUA = np.vstack((population, population_object.JADE_A))
        
    #     if HARD_B.size == 0:
    #         PUB = population
    #     else:
    #         PUB = np.vstack((population, population_object.HARD_B))
        
    #     F = parameters[0]
    #     F1=0.9*F
    #     F2=0.7*F
    #     Xr1G = population[random.randint(0, Len-1)]
    #     Xr2G = PUA[random.randint(0, len(PUA)-1)]
    #     Xr3G = PUB[random.randint(0, len(PUB)-1)]
        
        
    #     mutation_vector = population[individual_indice]+F*(XpbestG-population[individual_indice])+F1*(Xr1G-Xr2G)+F2*(Xr1G-Xr3G)
        
    #     new_individual=mutation_vector  
    #     return new_individual
    def construct_random_indices(self, env,indices, sub_pop_size, x_num):
        population_size = env.population.pop_size
        random_indices = np.zeros((sub_pop_size, x_num), dtype=int)
        
        for i in range(sub_pop_size):
            available_indices = list(set(range(population_size)) - {indices[i]})
            # print('available_indices',available_indices)
            random_indices[i] = np.random.choice(available_indices, x_num, replace=False)
        
        return random_indices
    # population version
    def mutation(self,env,indexs,parameters):
        """
        Perform the mutation operation for the Differential Evolution (DE) algorithm.
        This method generates a mutation vector based on the current population, 
        best individual, and archive population using the DE/current-to-pbest/1 
        mutation strategy with additional perturbations.
        Args:
            env (object): The environment object containing the population and 
                          mutation parameters.
            indexs (list): List of indices used to construct the mutated vector.
        Returns:
            numpy.ndarray: The resulting mutation vector.
        """
        
        population_object=env.population
        
        sub_pop=self.construct_sub_vector(env,indexs)
        XpbestG = self.construct_qbest(env,parameters)
        Xr1G_indices=self.construct_random_indices(env,indexs,len(indexs),1)
        Xr1G_indices=Xr1G_indices.T[0]
        Xr1G=population_object.current_vector[Xr1G_indices]
        # print('Xr1G_indices',Xr1G_indices)
        # print('Xr1G',Xr1G)
        archive_indices= self.construct_archive_indices(env,Xr1G_indices,indexs,len(indexs),2)
        Xr2G_indices=archive_indices[:,0]
        Xr3G_indices=archive_indices[:,1]
        Xr2G=self.construct_archive_pop(env,indexs,Xr2G_indices)
        Xr3G=self.construct_archive_pop(env,indexs,Xr3G_indices)
        F,F1,F2 = parameters[:,0],parameters[:,1],parameters[:,2]
        F = F[:, np.newaxis]
        F1 = F1[:, np.newaxis]
        F2 = F2[:, np.newaxis]
        # print('F.shape:',F.shape)
        # print('F1.shape',F1.shape)
        # print('F2.shape',F2.shape)
        # print('subpop.shape',sub_pop.shape)
        # print('xbest:',XpbestG.shape)
        # print('Xr1G:',Xr1G.shape)
        # print('Xr2G:',Xr2G.shape)
        # print('Xr3G:',Xr3G.shape)
        
        mutation_vector = sub_pop+F*(XpbestG-sub_pop)+F1*(Xr1G-Xr2G)+F2*(Xr1G-Xr3G)
        mutation_vector= self.re_boudary(env,mutation_vector)
        return mutation_vector
        
    # def construct_XpbestG(self,env,parameters):
    #     """
    #     Constructs the XpbestG vector, which is the best individual from a randomly selected subset of the population.
    #     Args:
    #         env: An environment object that contains the current population and their fitness values.
    #     Returns:
    #         XpbestG: The best individual vector from the selected subset of the population.
    #     """
        
        
        
    #     # todo: modified by network
        
    #     population = env.population.current_vector
    #     num_to_select = int(p * env.population.pop_size)
    #     selected_indices = np.random.choice(env.population.pop_size, num_to_select, replace=False)
    #     selected_individuals = env.population.current_vector[selected_indices]
    #     selected_fitness = env.population.current_fitness[selected_indices]
    #     best_individual_index = selected_indices[np.argmin(selected_fitness)]
    #     XpbestG = population[best_individual_index]
    #     return XpbestG
    
    # def construct_archive_pop(self,env,indices):

        
    #     archive = np.array(env.population.archive)
    #     # HARD_B = np.array(env.population.HARD_B)
    #     if archive.size == 0:
    #         PUA = env.population.current_vector
    #     else:
    #         PUA = np.vstack((env.population.current_vector, env.population.archive))
        
    #     archive_pop_size = len(indices)
        
    #     random_indices1 = np.random.choice(env.population.pop_size, archive_pop_size, replace=False)
    #     random_indices2 = np.random.choice(len(PUA), archive_pop_size, replace=False)
    #     random_indices3 = np.random.choice(len(PUA), archive_pop_size, replace=False)
        
    #     Xr1G = env.population.current_vector[random_indices1]
    #     Xr2G = PUA[random_indices2]
    #     Xr3G = PUA[random_indices3] 
        
    #     return Xr2G, Xr3G
    
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
    
    def construct_archive_pop(self,env,indices,chosen):
        archive = np.array(env.population.archive)
        if archive.size == 0:
            PUA = env.population.current_vector
        else:
            PUA = np.vstack((env.population.current_vector, env.population.archive))
        archive_pop_size = len(indices)
        
        # random_indice=np.random.choice(len(PUA), archive_pop_size, replace=False)
        return PUA[chosen]