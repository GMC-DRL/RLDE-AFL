import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import os
from typing import Optional, Union
import argparse
params = {
    'axes.labelsize': '25',
    'xtick.labelsize': '25',
    'ytick.labelsize': '25',
    'lines.linewidth': '3',
    'legend.fontsize': '24',
    'figure.figsize': '20,11',
}
plt.rcParams.update(params)

markers = ['o', '^', '*', 'O', 'v', 'x', 'X', 'd', 'D', '.', '1', '2', '3', '4', '8', 's', 'p', 'P', 'h', 'H']
colors = ['b', 'g', 'orange', 'r', 'purple', 'brown', 'grey', 'limegreen', 'turquoise', 'olivedrab', 'royalblue', 'darkviolet',
          'chocolate', 'crimson', 'teal','seagreen', 'navy', 'deeppink', 'maroon', 'goldnrod',
          ]


def gen_agent_performance_table(results: dict, out_dir: str) -> None:
    total_cost=results['cost']
    table_data = []
    indexs=[]
    columns=['Worst','Best','Median','Mean','Std']
    for problem,value in total_cost.items():
        indexs.append(problem)
        problem_cost=value

        # 拿到的problem_cost 就是agent的
        n_cost = []
        for run in problem_cost:
            n_cost.append(run[-1])
        best = np.min(n_cost)
        best = np.format_float_scientific(best, precision = 3, exp_digits = 3)
        worst = np.max(n_cost)
        worst = np.format_float_scientific(worst, precision = 3, exp_digits = 3)
        median = np.median(n_cost)
        median = np.format_float_scientific(median, precision = 3, exp_digits = 3)
        mean = np.mean(n_cost)
        mean = np.format_float_scientific(mean, precision = 3, exp_digits = 3)
        std = np.std(n_cost)
        std = np.format_float_scientific(std, precision = 3, exp_digits = 3)
        table_data.append([worst, best, median, mean, std])

    dataframe = pd.DataFrame(data = table_data, index = indexs, columns = columns)
    dataframe.to_excel(os.path.join(out_dir, f"concrete_performance_table.xlsx"))
    # for alg,data in table_data.items():
    #     dataframe=pd.DataFrame(data=data,index=indexs,columns=columns)
    #     #print(dataframe)
    #     dataframe.to_excel(os.path.join(out_dir,f'{alg}_concrete_performance_table.xlsx'))


def gen_algorithm_complexity_table(results: dict, out_dir: str) -> None:


    t0 = results['T0']
    t1 = results['T1']
    t2 = results['T2']
    indexs = ["AutoDE"]
    columns = ['T0', 'T1', 'T2', '(T2-T1)/T0']

    data = np.zeros((1, 4))
    data[0, 0] = t0
    data[0, 1] = t1
    data[0, 2] = t2
    data[0, 3] = (t2 - t1) / t0
    table = pd.DataFrame(data = np.round(data, 2), index = indexs, columns = columns)
    table.to_excel(os.path.join(out_dir, 'algorithm_complexity.xlsx'))

def estimate_test_statics(agent_path):
    out_dir = agent_path + 'performance/'
    agent_path = agent_path + 'test.pkl'
    with open(agent_path, 'rb') as f:
        results = pickle.load(f)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    gen_agent_performance_table(results, out_dir)
    gen_algorithm_complexity_table(results, out_dir)


if __name__ == "__main__":
    with open("test.pkl", 'rb') as f:
        results = pickle.load(f)

    gen_agent_performance_table(results, "outputs/")