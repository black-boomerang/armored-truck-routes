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
        start_density += np.exp(25) * (self.remains[terminals] > self.environment.terminal_limit)
        start_density += (self.days_after_service[terminals] + 1) / self.environment.non_serviced_days
        start_density += np.exp(3 * (self.days_after_service[terminals] - 6)) * (self.days_after_service[terminals] > 6)
        start_density = (start_density / np.linalg.norm(start_density)).clip(DensitySolver.EPSILON)

        sub_time_matrix = self.time_matrix[np.ix_(terminals, terminals)]
        sub_time_matrix = (sub_time_matrix / np.linalg.norm(sub_time_matrix, axis=1, keepdims=True, ord=1)).clip(
            DensitySolver.EPSILON)
        log_density = logsumexp(-sub_time_matrix + np.log(start_density)[None, :], axis=1)
        log_density += 1000 * (self.remains[terminals] > self.environment.terminal_limit)
        log_density += 1000 * (self.days_after_service[terminals] >= self.environment.non_serviced_days)
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
        right = min(len(cluster), 72)
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

    def get_cluster(self, terminals: np.ndarray, center: int, cur_armored: int) -> np.ndarray:
        """ Выделяем кластер на основе времени ОТ терминала-центра """
        time_to_terminals = np.full(self.terminals_num, np.inf)
        time_to_terminals[terminals] = self.time_matrix[center, terminals]
        neighbourhood = len(terminals) // (self.armored_num - cur_armored)
        return np.argpartition(time_to_terminals, neighbourhood)[:neighbourhood]

    def get_routes(self) -> List[List[int]]:
        """
        Получить маршруты для всех броневиков на текущий день.
        :return: список маршрутов для каждого броневика
        """

        cur_terminals = np.arange(self.terminals_num)
        routes = []
        for i in range(self.armored_num):
            density = np.full(self.terminals_num, -np.inf)
            density[cur_terminals] = self.get_density(cur_terminals)
            best_terminal = np.argmax(density)
            cluster = self.get_cluster(cur_terminals, best_terminal, i)
            routes.append(self.get_cluster_route(cluster, density))
            cur_terminals = np.setdiff1d(cur_terminals, np.array(routes[-1]))
            if len(cur_terminals) == 0:
                break
        self.remains[list(itertools.chain(*routes))] = 0.0
        self.days_after_service[list(itertools.chain(*routes))] = -1

        max_days_terminals = (self.days_after_service >= self.environment.non_serviced_days).sum()
        bad_limit_terminals = (self.remains > self.environment.terminal_limit).sum()
        assert max_days_terminals == 0, f'Превышено максимальное кол-во дней для {max_days_terminals} терминалов'
        assert bad_limit_terminals == 0, f'Превышена максимальная сумма в терминале для {bad_limit_terminals} терминалов'

        return routes
