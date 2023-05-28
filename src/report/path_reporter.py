import os
from typing import List
import numpy as np
import pandas as pd
from datetime import datetime


class PathReporter:
    
    def __init__(self, 
                 n_trucks: int, 
                 terminal_ids: List[int],
                 time_matrix: np.ndarray):
        
        self.n_trucks = n_trucks
        self.terminal_ids = terminals
        self.service_time = 10 
        self.time_matrix = time_matrix
    
    def evaluate(self, roots: List[List[int]], day=datetime):
        
        columns = ['порядковый номер броневика',
                   'устройство',
                   'дата-время прибытия',
                   'дата-время отъезда']
        
        day = pd.to_datetime(day)
        
        if not os.path.exists('roots.csv'):
            df_roots = pd.DataFrame(columns=columns)
        else:
            df_roots = pd.read_csv('roots.csv')
        
        for i, root in enumerate(roots):
            
            start_time = pd.to_datetime(day) + pd.Timedelta('09:00:00')
            
            path_time = 0
            
            for j, t in enumerate(root):
                
                terminal_id = self.terminal_ids[t]
                
                arr_time = (start_time + 
                            pd.Timedelta(path_time, unit='m') + 
                            pd.Timedelta(j * self.service_time, unit='m'))
                
                dep_time = arr_time + pd.Timedelta(self.service_time, unit='m')
                
                if j < len(root) - 1:
                    
                    path_time += self.time_matrix[t][root[j + 1]]
                    
                data = dict(zip(columns, [i, self.terminal_ids[t], arr_time, dep_time]))
                
                new_entry = pd.DataFrame(data, index=[len(df_roots)])
                df_roots = pd.concat([df_roots, new_entry])
                
        df_roots.to_csv('roots.csv', index=None)