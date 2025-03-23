import numpy as np
import pickle
import os
import math
import torch
class Memory:
    def __init__(self):
        self.states = {'x': [], 'y': [], 'fes': []}
        self.actions = []
        self.logprobs = []
        self.rewards = []

    def clear_memory(self):
        del self.states['x'][:]
        del self.states['y'][:]
        del self.states['fes'][:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]

    def states_append(self, state):
        x, y, fes = state['x'], state['y'], state['fes']
        self.states['x'].append(x)
        self.states['y'].append(y)
        self.states['fes'].append(fes)

    def return_states(self):
        # todo 这里还是由于bs =1 直接压缩掉了，后边bs加上后要修改
        # todo [t, bs, ps, dim_f]
        x = np.stack(self.states['x']).squeeze(axis = 1) # 4D [t, 1, ps, dim_f] ------> 3D [t, ps, dim_f]
        y = np.stack(self.states['y']).squeeze(axis = 1)
        fes = np.stack(self.states['fes']).squeeze(axis = 1)
        state = {'x' : x, 'y' : y, 'fes' : fes}
        return state

def save_class(dir, file_name, saving_class):
    # print('Saving agent ...')
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(dir + file_name + '.pkl', 'wb') as f:
        pickle.dump(saving_class, f, -1)

def clip_grad_norms(param_groups, max_norm = math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for idx, group in enumerate(param_groups)
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped

if __name__ == "__main__":
    memory = Memory()
    x = 2
    y =3
    memory.states['x'].append(x)
    memory.states['y'].append(y)
    memory.clear_memory()
    print(memory)
