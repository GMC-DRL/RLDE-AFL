from Mutation_Selection.conventional_mutations import (
    best_1,
    best_2,
    current_to_rand_1,
    rand_1,
    rand_1_bin,
    rand_2,
    rand_2_bin,
    rand_to_best_1,
    current_to_best_1
)
from Mutation_Selection.novel_mutations import (
    HARDDE,
    JADE,
    MadDE,
    MDE,
    MDE_pBX,
    pro_rand_1,
    pro_rand_2,
    SA_RM_rand_1,
    SA_RM_best_1,
    TopoMut_DE,
    current_to_rand_1_archive,
    weighted_rand_to_qbest_1

)
    
# import random
import numpy as np

# todo: Mutations selection policy (using normal distribution,and use probability to select the crossover policy)   
# todo: current to best

mutation_operators =["best_1", "best_2", "rand_1", "rand_2","current_to_best_1",
                     "rand_1_bin","rand_2_bin","SA_RM_rand_1",
                    "rand_to_best_1","current_to_rand_1","MDE_pBX", "pro_rand_1",
                    "MadDE","TopoMut_DE","JADE", "HARDDE","weighted_rand_to_qbest_1","current_to_rand_1_archive"]

# def get_n_mutation(mutation_operator):
#     invalid_operators = [op for op in mutation_operator if op not in mutation_operators]
#     if invalid_operators:
#         raise ValueError(f"Invalid mutation operators: {invalid_operators}. Must be one of {mutation_operators}")
#     operators = {}
#     for op_name in mutation_operator:
#         operators[op_name] = globals()[op_name]
#
#     return operators,


class select_mutation:

    # distribution = np.random.normal(loc=0, scale=1, size=len(mutation_operators))

    def __init__(self, mutation_operator):

        invalid_operators = [op for op in mutation_operator if op not in mutation_operators]
        if invalid_operators:
            raise ValueError(f"Invalid mutation operators: {invalid_operators}. Must be one of {mutation_operators}")

        operators = {}
        n_mutation = -1
        print(mutation_operator)
        for operator_name in mutation_operator:
            operators[operator_name] = eval(operator_name)()
            print('operator_name:',operator_name)
            n_mutation = max(n_mutation, operators[operator_name].get_parameters_numbers())

        self.mutation_operator = mutation_operator
        self.selected_operator = operators

        self.n_operator = len(self.mutation_operator)
        self.n_mutation = n_mutation



    # @staticmethod
    # def select_mutation_operator(mutation_operators, distribution):
    #     """
    #     Select a mutation operator based on the given probabilities.
    #     Parameters:
    #     - mutation_operators (list): A list of mutation operators.
    #     - distribution: a normal distribution (is it necessary ?,or just randomly initialize the distribution)
    #     Returns:
    #     - selected_operator: The selected mutation operator.
    #     """
    #     selected_operator={
    #         "best_1": best_1,
    #         "best_2": best_2,
    #         "rand_1": rand_1,
    #         "rand_2": rand_2,
    #         "rand_to_best_1": rand_to_best_1,
    #         "rand_1_bin": rand_1_bin,
    #         "rand_2_bin": rand_2_bin,
    #         "current_to_rand_1": current_to_rand_1,
    #         "JADE": JADE,
    #         "MadDE": MadDE,
    #         "MDE": MDE,
    #         "MDE_pBX": MDE_pBX,
    #         "pro_rand_1": pro_rand_1,
    #         "TopoMut_DE": TopoMut_DE,
    #         "SA_RM_rand_1":SA_RM_rand_1,
    #         "SA_RM_best_1":SA_RM_best_1,
    #         "HARDDE":HARDDE
    #     }
    #     # to be done
    #     return mutation_operators.get(selected_operator, None)


# a temporary solution
# it will be replaced after implementing the selection policy
    def select_mutation_operator(self, mutation_operators):
        """
        Select a mutation operator based on the given probabilities.
        Parameters:
        mutation_operators(str): The name of the mutation operator.
        Returns:
        - selected_operator: The selected mutation operator.
        """
        selected_operator={
            "best_1": best_1,
            "best_2": best_2,
            "rand_1": rand_1,
            "rand_2": rand_2,
            "rand_to_best_1": rand_to_best_1,
            "rand_1_bin": rand_1_bin,
            "rand_2_bin": rand_2_bin,
            "current_to_rand_1": current_to_rand_1,
            "current_to_best_1": current_to_best_1,
            "JADE": JADE,
            "MadDE": MadDE,
            "MDE": MDE,
            "MDE_pBX": MDE_pBX,
            "pro_rand_1": pro_rand_1,
            "pro_rand_2": pro_rand_2,
            "TopoMut_DE": TopoMut_DE,
            "SA_RM_rand_1":SA_RM_rand_1,
            "SA_RM_best_1":SA_RM_best_1,
            "HARDDE":HARDDE,
            "current_to_rand_1":current_to_rand_1_archive,
            "weighted_rand_to_qbest_1":weighted_rand_to_qbest_1
        }
        # return mutation_operators.get(selected_operator, None)


        # Return the selected operator class
        # return selected_operator[mutation_operators]
        operator_class = selected_operator[mutation_operators]
        # print('operator_class:',operator_class)
        if operator_class:
            return operator_class()
        else:
            return None

    # the official solution
    def select_mutation_operator(self, mutation_operator):
        """
        Selects a mutation operator based on the given mutation_operator parameter.
        Parameters:
        mutation_operator (int): The index of the mutation operator to select.
        Returns:
        object or None: The selected mutation operator class if found, None otherwise.
        """

        # mutation_operators=["best_1", "best_2", "rand_1", "rand_2",
        #                     "rand_to_best_1","rand_1_bin","rand_2_bin",
        #                     "current_to_rand_1","MDE_pBX", "pro_rand_1",
        #                     "TopoMut_DE","JADE","SA_RM_rand_1", "HARDDE"]

        mutation_operator_name = self.mutation_operator[mutation_operator]
        # if np.random.rand() < 0.1:
        # print('using mutation operator name:' ,mutation_operator_name)
        # selected_operator={
        #     "best_1": best_1,
        #     "best_2": best_2,
        #     "rand_1": rand_1,
        #     "rand_2": rand_2,
        #     "rand_to_best_1": rand_to_best_1,
        #     "rand_1_bin": rand_1_bin,
        #     "rand_2_bin": rand_2_bin,
        #     "current_to_rand_1": current_to_rand_1,
        #     "JADE": JADE,
        #     "MadDE": MadDE,
        #     "MDE": MDE,
        #     "MDE_pBX": MDE_pBX,
        #     "pro_rand_1": pro_rand_1,
        #     "pro_rand_2": pro_rand_2,
        #     "TopoMut_DE": TopoMut_DE,
        #     "SA_RM_rand_1":SA_RM_rand_1,
        #     "SA_RM_best_1":SA_RM_best_1,
        #     "HARDDE":HARDDE
        # }
        # return mutation_operators.get(selected_operator, None)


        # Return the selected operator class
        # return selected_operator[mutation_operators]
        operator_class = self.selected_operator[mutation_operator_name]
        # print('operator_class:',operator_class)
        if operator_class:
            return operator_class
        else:
            return None



    @staticmethod
    def random_select():
        """
        Randomly select a mutation operator name.
        Returns:
        - mutation_operator_name (str): The name of the selected mutation operator.
        """
        mutation_operator_names = [
            "best_1", "best_2", "rand_1", "rand_2", "rand_to_best_1",
            "rand_1_bin", "rand_2_bin", "current_to_rand_1",
            "MDE_pBX", "pro_rand_1", "TopoMut_DE","JADE",
            "SA_RM_rand_1", "HARDDE"
        ]
        # no mde and sarmbest1 because they are not available

        # print("Available mutation operators:", mutation_operator_names)

        if not mutation_operator_names:
            raise ValueError("No mutation operators available.")

        selected_operator_name = np.random.choice(mutation_operator_names)
        # print("Selected mutation operator:", selected_operator_name)
        return selected_operator_name
    