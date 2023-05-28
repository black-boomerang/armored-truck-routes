import itertools
from typing import Optional, List

import numpy as np
from scipy.special import logsumexp

from src.solvers.base_solver import BaseSolver
from src.task import Environment
from src.utils import tsp_solution


class DensitySolver(BaseSolver):
    """ Решение на основе плотности """
    EPSILON = 1e-9

    def __init__(self, remains: np.ndarray, time_matrix: np.ndarray, business_logic: Environment,
                 armored_num: int = 10, sigma: float = 7000):
        super().__init__(remains, time_matrix, business_logic)
        self.armored_num = armored_num
        # константа, используемая при расчёте плотности (параметр гауссианы)
        self.sigma = sigma

    def get_density(self, terminals: Optional[np.ndarray] = None) -> np.ndarray:
        """ Функция расчёта плотности """
        if terminals is None:
            terminals = np.arange(self.terminals_num)
        start_density = self.remains[terminals] / self.environment.terminal_limit
        start_density += self.days_after_service[terminals] / self.environment.non_serviced_days
        start_density = (start_density / np.linalg.norm(start_density)).clip(DensitySolver.EPSILON)

        sub_time_matrix = self.time_matrix[np.ix_(terminals, terminals)]
        log_density = logsumexp(-sub_time_matrix / self.sigma * 0 + np.log(start_density)[None, :], axis=1)
        return log_density

    def get_clusters(self, centers: np.ndarray) -> List[np.ndarray]:
        """ Выделяем кластера на основе времени ОТ терминалов-центров """
        k = len(centers)
        terminals = np.arange(self.terminals_num)
        distances = self.time_matrix[centers[None, :], terminals[:, None]]  # (terminals_num x k)
        clusters_inds = distances.argmin(axis=1)
        clusters = []
        for cluster in range(k):
            clusters.append(np.where(clusters_inds == cluster)[0])
        return clusters

    def get_cluster_route(self, cluster: np.ndarray, density: np.ndarray) -> List[int]:
        """ Получить маршрут для одного броневика внутри заданного кластера """

        # сопоставляем
        cluster_ind_to_ind = {cluster_i: i for cluster_i, i in enumerate(cluster)}
        cluster_density = density[cluster]
        cluster_sorted_indecies = (-cluster_density).argsort()

        left = 0
        right = len(cluster)
        while left < right:
            serviced_terminals = right - (right - left) // 2
            subcluster = np.array(
                [cluster_ind_to_ind[terminal] for terminal in cluster_sorted_indecies[:serviced_terminals]])
            route, r_time = tsp_solution(self.time_matrix[subcluster[None, :], subcluster[:, None]])
            r_time += serviced_terminals * self.environment.encashment_time
            if r_time <= self.environment.working_day_time:
                left = serviced_terminals
            else:
                right = serviced_terminals - 1

        subcluster = np.array([cluster_ind_to_ind[terminal] for terminal in cluster_sorted_indecies[:left]])
        best_route, _ = tsp_solution(self.time_matrix[subcluster[None, :], subcluster[:, None]])
        return [cluster_ind_to_ind[cluster_sorted_indecies[terminal]] for terminal in best_route]

    def get_cluster(self, terminals: np.ndarray, center: int) -> np.ndarray:
        """ Выделяем кластер на основе времени ОТ терминала-центра """
        time_to_terminals = np.full(self.terminals_num, np.inf)
        time_to_terminals[terminals] = self.time_matrix[center, terminals]
        return np.argpartition(time_to_terminals, 72)[:72]

    def get_routes(self) -> List[List[int]]:
        """
        Получить маршруты для всех броневиков на текущий день.
        :return: список маршрутов для каждого броневика
        """
        remains = (self.environment.terminal_limit - self.remains) / self.environment.terminal_limit
        times = self.days_after_service
        assert times.max() < self.environment.non_serviced_days

        days_before_deadline = 1
        idx = list(set(np.concatenate([
            np.where(times == self.environment.non_serviced_days - days_before_deadline)[0],
            np.where(remains < 0.1)[0]
        ])))
        assert len(idx) <= self.num_routes_per_day

        times = times / self.environment.non_serviced_days
        cost = times
        idx = np.concatenate([np.argsort(-cost)[:(self.num_routes_per_day - len(idx))], idx])

        cur_terminals = idx
        routes = []
        for i in range(self.armored_num):
            density = np.full(self.terminals_num, -np.inf)
            density[cur_terminals] = self.get_density(cur_terminals)
            best_terminal = cur_terminals[np.argmax(-density)]
            cluster = self.get_cluster(cur_terminals, best_terminal)
            routes.append(self.get_cluster_route(cluster, density))
            cur_terminals = np.setdiff1d(cur_terminals, np.array(routes[-1]))
            if len(cur_terminals) == 0:
                break
        self.remains[list(itertools.chain(*routes))] = 0.0
        self.days_after_service[list(itertools.chain(*routes))] = -1
        return routes
