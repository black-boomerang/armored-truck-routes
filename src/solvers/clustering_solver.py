import numpy as np
import networkx as nx

from src.task import Environment
from src.solvers.base_solver import BaseSolver
from src.clustering import STPClustering

tsp = nx.approximation.traveling_salesman_problem


class ClusteringSolver(BaseSolver):
    """ Решение на основе кластеризации """

    EPSILON = 1e-9

    def __init__(self, remains: np.ndarray, time_matrix: np.ndarray, environment: Environment, n_trucks: int):
        super().__init__(remains, time_matrix, environment, n_trucks)
        self.clustering = STPClustering(max_length=540) 
        self.num_routes_per_day = int(np.ceil(0.1 * len(time_matrix)))

    def tsp_solution(sellf, distances):

        '''
            Решение задачи коммивояжера алгоритмом Кристифодеса
        '''

        sym_distances = np.maximum(np.tril(distances), np.triu(distances).T)
        sym_distances = sym_distances + sym_distances.T
        G = nx.from_numpy_array(sym_distances)

        if sym_distances.shape[0] > 1:
            path = tsp(G, cycle=False)
        else:
            path = [0]
            
        return path
    
    def get_candidates(self):
        
        '''
            Функция получения кандитов для обхода на текущий день
        '''

        remains = (self.environment.terminal_limit - self.remains) / self.environment.terminal_limit

        times = self.days_after_service

        # проверяем, что все укладывается в дедлайн
        assert times.max() < self.environment.non_serviced_days 

        days_before_deadline = 1
        idx = list(set(np.concatenate([
            # выбираем терминалы, делайн обслуживания которых наступает сегодня
            np.where(times == self.environment.non_serviced_days - days_before_deadline)[0], 

            # выбираем все переполнившиеся терминалы
            np.where(remains < 0.1)[0] 
        ])))

        assert len(idx) <= self.num_routes_per_day

        times = times / self.environment.non_serviced_days
        cost = times
        
        # дополняем до нужного процента в порядке дедлайна
        idx = np.concatenate([np.argsort(-cost)[:(self.num_routes_per_day - len(idx))], idx]) 

        return np.array(idx)

    def get_routes(self, idx=None):

        '''
            Функция получения маршрутов для всех броневиков на текущий день
        '''

        idx = self.get_candidates()
        time_matrix = self.time_matrix[np.ix_(idx, idx)]

        # выполняем кластеризацию на основе выбранных кандидатов
        clusters = self.clustering.fit_predict(time_matrix) 

        paths = []
        times = []
  
        for cluster in clusters:
            subset = time_matrix[np.ix_(cluster, cluster)]

            # запускаем решение задачи коммивояжера внутри кластера
            path = self.tsp_solution(subset) 

            # считаем затраченное время для самопроверки
            src, dst = path[:-1], path[1:]
            elapsed = (subset[src, dst]).sum() + 10 * len(path) 

            paths.append(idx[cluster[path]])
            times.append(elapsed)

        # проверяем, что все автомобили уложились в рабочее время
        assert max(times) < self.environment.working_day_time 
        
        self.remains[np.concatenate(paths)] = 0.0
        self.days_after_service[np.concatenate(paths)] = -1.0

        # проверяем, что ничего не переполнилось
        assert (self.environment.terminal_limit - self.remains > 0).all() 

        # проверяем, что уложились в заданное число автомобилей
        assert len(paths) <= self.n_trucks

        return paths
