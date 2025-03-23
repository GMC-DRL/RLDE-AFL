from ..basic_mutation import basic_mutation
import numpy as np

class MDE(basic_mutation):
    def get_parameters_numbers(self):
        # p --mutation probability
        # eta --obey gaussian distribution(0,1)
        return 1
    

    def mutation(self, env,individual_indice): 
        population_object=env.population
        parameters=env.action['mutation_parameters']
        
        population=population_object.current_vector
        F = parameters[0]
        Len=population_object.pop_size
        # get the best individual and get d
        best_individual_indice=0
        best_individual=1e9
        sumD=[]
        for i in range(Len):
            sumD.append(population_object.current_fitness[i]-population_object.calculate_result())
        
        best_individual_indice=np.argmin(population_object.current_fitness)
        best=population[best_individual_indice]
        # print('sumD:',sumD) 
        
        maxD=max(sumD)
        if(maxD==0):
            dev=1
        else:
            dev=maxD
        finalD=0
        for i in range(len(sumD)):
            finalD=finalD+(sumD[i]/dev)**2
        finalD=finalD**0.5

        dc=2.5
        # print('finalD',finalD)
        if(finalD<dc):
            p=0.5    
        else:
            p=0
        # note: 1<= dc <=2.5
        #       p=k
        #      0.2<=k<=0.5 
        #  so dc,k can be also parameters
        
        eta = np.random.normal(0, 1)  # Generate a random value from a standard normal distribution
        best_mde=(1+0.5*eta)*best
        # print('best_mde',best_mde)
        # for i in range(Len):
        if(np.random.rand()<p):
            # print("change")
            indices = np.random.choice(range(Len), 4, replace=False)
            x1, x2, x3,x4 = population[indices[0]], population[indices[1]], population[indices[2]], population[indices[3]]
            mutated_vector=best_mde+F*(x1-x2)+F*(x3-x4) # is it correct?
            new_individual=mutated_vector
        else:
            new_individual=population[individual_indice]
        # print('origin_individual',population[individual_indice])
        # print('new_individual',new_individual)
        return new_individual
    
    # the optimization ability is too weak?
    # maybe there are some mistakes in my code..... 
