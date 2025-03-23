# todo: CrossOver selection policy (using normal distribution,and use probability to select the crossover policy)   
# but we haven't collect the crossovers from papers.
from CrossOver_Selection.simple_crossover import simple_crossover

from CrossOver_Selection.conventional_crossover.binomial import binomial
from CrossOver_Selection.conventional_crossover.exponential import exponential
from CrossOver_Selection.novel_crossover.prefential_crossover import prefential
from CrossOver_Selection.novel_crossover.MDE_pBX_crossover import MDE_pBX
# from CrossOver_Selection.novel_crossover.
import numpy as np

crossover_operators = ["binomial", "exponential", "prefential", "MDE_pBX"]

class select_crossover:

    def __init__(self, crossover_operator):
        invalid_operators = [op for op in crossover_operator if op not in crossover_operators]
        if invalid_operators:
            raise ValueError(f"Invalid crossover operators: {invalid_operators}. Must be one of {crossover_operators}")

        operators = {}
        n_crossover = -1
        for operator_name in crossover_operator:
            operators[operator_name] = eval(operator_name)()
            n_crossover = max(n_crossover, operators[operator_name].get_parameters_numbers())

        self.crossover_operator = crossover_operator
        self.selected_operator = operators

        self.n_operator = len(self.crossover_operator)
        self.n_crossover = n_crossover


    def select_crossover_operator(crossover_operators, distribution):
        """
        Select a crossover operator based on the given probabilities.
        Parameters:
        - crossover_operators (list): A list of crossover operators.
        - distribution: a normal distribution (is it necessary ?,or just randomly initialize the distribution)
        Returns:
        - selected_operator: The selected crossover operator.
        """
        selected_operator={
            "binomial_crossover": binomial,
            "exponential_crossover": exponential,
            "MDE_pBX_crossover": MDE_pBX

        }
        # to be done
        return crossover_operators.get(selected_operator, None)

    # a temporary solution
    # it will be replaced after implementing the selection policy
    def select_crossover_operator(self, crossover_operator):
        # todo 这里crossover_operator 应该是int就行
        """
        Select a crossover operator based on the given probabilities.
        Parameters:
        crossover_operators(str): The name of the crossover operator.
        Returns:
        - selected_operator: The selected crossover operator.
        """
        # select_operator_name=["binomial_crossover","MDE_pBX_crossover"]
        #
        # selected_operator={
        #     "binomial_crossover": binomial_crossover,
        #     "exponential_crossover": exponential_crossover,
        #     "MDE_pBX_crossover": MDE_pBX_crossover
        # }
        crossover_operator_name = self.crossover_operator[crossover_operator]
        operator_class = self.selected_operator[crossover_operator_name]
        # print('operator_class:',operator_class)
        if operator_class:
            return operator_class
        else:
            return None

    def random_select():
        """
        Select a crossover operator randomly.
        Returns:
        - selected_operator: The selected crossover operator.
        """
        crossover_operators = ["simple_crossover"]
        selected_operator_name = np.random.choice(crossover_operators)
    # print("Selected mutation operator:", selected_operator_name)
        return selected_operator_name