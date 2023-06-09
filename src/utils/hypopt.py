import sys

sys.path.append('..')
sys.path.append('../..')

import os
from itertools import product
import numpy as np
from tqdm import tqdm

from tools.evaluate import get_losses

class HyperOptimizer:

    def __init__(
            self,
            solver,
            data,
            env,
            params_fixed: dict,
            params_to_opt: dict,
            log_file_name='logs'):

        self.solver = solver
        self.data = data
        self.env = env
        self.params_fixed = params_fixed
        self.params_to_opt = params_to_opt
        self.log_file_name = log_file_name

    def optimize(self, verbose=False):

        log_file = open(self.log_file_name, 'w')
        param_names, param_vals = list(zip(*sorted(list(self.params_to_opt.items()))))
        log_file.write(' '.join(param_names) + ' ' + 'total_loss_M\n')
        if verbose:
            print(' '.join(param_names) + ' ' + 'total_loss_M\n')
        terminals, time_matrix = self.data
        for param_val in product(*param_vals):
            
            params = dict(zip(param_names, param_val))
            solver = self.solver(
                    terminals['start_value'].values, 
                    time_matrix, 
                    self.env, 
                    **self.params_fixed, 
                    **params)

            total_losses = []
            try:

                for day in tqdm(range(91)):
                    routes = solver.get_routes()
                    solver.update(terminals[f'day {day + 1}'])
                    losses = get_losses(routes, solver.remains, len(terminals), solver.environment)
                    total_losses.append(losses)
                log_entry = ' '.join(map(str, param_val)) + ' ' + str(np.sum(total_losses) / 1e6) + '\n'
                if verbose:
                    print(log_entry)

                log_file.write(log_entry)

            except AssertionError as msg:
                
                log_entry = ' '.join(map(str, param_val)) + ' ' + str(msg) + '\n'
                if verbose:
                    print(log_entry)
                log_file.write(log_entry)

        log_file.close() 
           
if __name__ == '__main__':

    from solvers import DensitySolver
    from task import Environment
    from utils import get_data

    DATA_ROOT = '../../data'
    TERMINAL_DATA_PATH = os.path.join(DATA_ROOT, 'terminal_data_hackathon v4.xlsx')
    TIMES_DATA_PATH = os.path.join(DATA_ROOT, 'times v4.csv')

    data = get_data(DATA_ROOT)
    env = Environment()
    solver = DensitySolver

    params_fixed = dict(
            armored_num = 5, 
            sigma = 500
    )

    params_to_opt = dict(
            tl = np.arange(20, 31), 
            das_c1 = [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], 
            das_c2 = np.arange(4, 10), 
            das_c3 = np.arange(4, 10)
    )

    opt = HyperOptimizer(solver, data, env, params_fixed, params_to_opt)
    opt.optimize(verbose=True)
