
import itertools
from typing import Optional, List

import numpy as np
import networkx as nx

from src.task import Environment
from src.solvers.base_solver import BaseSolver
from src.clustering import STPClustering

tsp = nx.approximation.traveling_salesman_problem


class ClusteringSolver(BaseSolver):
    """ Решение на основе кластеризации """

    EPSILON = 1e-9

    def __init__(self, remains: np.ndarray, time_matrix: np.ndarray, environment: Environment, armored_num: int = 10):
        super().__init__(remains, time_matrix, environment)
        self.armored_num = armored_num
        self.clustering = STPClustering(max_length=540)
        self.num_routes_per_day = int(np.ceil(0.1 * len(time_matrix)))

    def tsp_solution(sellf, distances):
        sym_distances = np.maximum(np.tril(distances), np.triu(distances).T)
        sym_distances = sym_distances + sym_distances.T
        G = nx.from_numpy_array(sym_distances)

        if sym_distances.shape[0] > 1:
            path = tsp(G, cycle=False)
        else:
            path = [0]
            
        return path

    def get_routes(self, idx=None) -> List[List[int]]:

        remains = (self.environment.terminal_limit - self.remains) / self.environment.terminal_limit
        # print(remains.min())

        times = self.days_after_service
        assert times.max() < self.environment.non_serviced_days

        days_before_deadline = 1
        idx = list(set(np.concatenate([
            np.where(times == self.environment.non_serviced_days - days_before_deadline)[0],
            np.where(remains < 0.1)[0]
        ])))
        # print(len(idx))
        assert len(idx) <= self.num_routes_per_day

        times = times / self.environment.non_serviced_days
        cost = times
        idx = np.concatenate([np.argsort(-cost)[:(self.num_routes_per_day - len(idx))], idx])
        # cost = np.stack([times, remains, np.arange(len(times))], axis=1)
        # idx = np.array(sorted(cost, key=lambda element: (-element[0], element[1])))[:, 2].astype(int)[:self.num_routes_per_day]
        # print(idx)

        if idx is None:
            time_matrix = self.time_matrix
        else:
            time_matrix = self.time_matrix[np.ix_(idx, idx)]

        clusters = self.clustering.fit_predict(time_matrix)

        paths = []
        times = []
        for cluster in clusters:
            subset = time_matrix[np.ix_(cluster, cluster)]
            path = self.tsp_solution(subset)

            src, dst = path[:-1], path[1:]
            elapsed = (subset[src, dst]).sum() + 10 * len(path)
            paths.append(cluster[path])
            times.append(elapsed)

        assert max(times) < self.environment.working_day_time

        self.remains[idx[np.concatenate(paths)]] = 0.0
        self.days_after_service[idx[np.concatenate(paths)]] = -1.0

        assert (self.environment.terminal_limit - self.remains > 0).all()
        assert len(paths) < 10

        return paths
