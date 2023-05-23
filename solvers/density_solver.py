from typing import Optional, List

import numpy as np
from scipy.special import logsumexp

from task import BusinessLogic
from utils import tsp_solution
from .base_solver import BaseSolver


class DensitySolver(BaseSolver):
    """ Решение на основе плотности """
    EPSILON = 1e-9

    def __init__(self, remains: np.ndarray, time_matrix: np.ndarray, business_logic: BusinessLogic,
                 armored_num: int = 10, sigma: float = 50):
        super().__init__(remains, time_matrix, business_logic)
        self.armored_num = armored_num
        # константа, используемая при расчёте плотности (параметр гауссианы)
        self.sigma = sigma

    def get_density(self, terminals: Optional[np.ndarray] = None) -> np.ndarray:
        """ Функция расчёта плотности """
        if terminals is None:
            terminals = np.arange(self.terminals_num)
        start_density = self.remains[terminals] / self.bl.terminal_limit
        start_density += self.days_after_service[terminals] / \
            self.bl.non_serviced_days
        start_density = (
            start_density / np.linalg.norm(start_density)).clip(DensitySolver.EPSILON)

        sub_time_matrix = self.time_matrix[np.ix_(terminals, terminals)]
        log_density = logsumexp(-sub_time_matrix / self.sigma +
                                np.log(start_density)[None, :], axis=1)
        return log_density

    def get_clusters(self, centers: np.ndarray) -> List[np.ndarray]:
        """ Выделяем кластера на основе времени ОТ терминалов-центров """
        k = len(centers)
        terminals = np.arange(self.terminals_num)
        distances = self.time_matrix[centers[None, :],
                                     terminals[:, None]]  # (terminals_num x k)
        clusters_inds = distances.argmin(axis=1)
        clusters = []
        for cluster in range(k):
            clusters.append(np.where(clusters_inds == cluster)[0])
        return clusters

    def get_cluster_route(self, cluster: np.ndarray, density: np.ndarray) -> List[int]:
        """ Получить маршрут для одного броневика внутри заданного кластера """

        # сопоставляем
        cluster_ind_to_ind = {
            cluster_i: i for cluster_i, i in enumerate(cluster)}
        cluster_density = density[cluster]
        cluster_sorted_indecies = (-cluster_density).argsort()
        cluster_ind_to_ind = {cluster_i: cluster_ind_to_ind[i] for cluster_i, i in enumerate(
            cluster_sorted_indecies)}

        left = 0
        right = len(cluster)
        best_route = []
        while left < right:
            serviced_terminals = right - (right - left) // 2
            subcluster = cluster_sorted_indecies[:serviced_terminals]
            route, r_time = tsp_solution(
                self.time_matrix[subcluster[None, :], subcluster[:, None]])
            r_time += serviced_terminals * self.bl.encashment_time
            if r_time <= self.bl.working_day_time:
                best_route = route
                left = serviced_terminals
            else:
                right = serviced_terminals - 1
        return [cluster_ind_to_ind[terminal] for terminal in best_route]

    def get_routes(self) -> List[List[int]]:
        """
        Получить маршруты для всех броневиков на текущий день.
        :return: список маршрутов для каждого броневика
        """
        density = self.get_density()
        best_terminals = np.argpartition(-density,
                                         self.armored_num)[:self.armored_num]
        clusters = self.get_clusters(best_terminals)
        routes = []
        for cluster in clusters:
            routes.append(self.get_cluster_route(cluster, density))
        return routes
