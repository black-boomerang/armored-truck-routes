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
    BIG_DENSITY = 1000

    def __init__(self,
                 remains: np.ndarray,
                 time_matrix: np.ndarray,
                 business_logic: Environment,
                 n_trucks: int = 10,
                 tl: float = 25,
                 das_c1: float = 3,
                 das_c2: float = 6,
                 das_c3: float = 6,
                 day_part: float = 0.3):
        super().__init__(remains, time_matrix, business_logic, n_trucks)

        # гиперпараметры, используемые при расчёте плотности
        self.tl = tl
        self.das_c1 = das_c1
        self.das_c2 = das_c2
        self.das_c3 = das_c3
        self.big_density = 1000

        # кол-во терминалов
        self.num_routes_per_day = int(np.ceil(day_part * self.terminals_num))

    def get_density(self, terminals: Optional[np.ndarray] = None) -> np.ndarray:
        """ Функция расчёта плотности """
        if terminals is None:
            terminals = np.arange(self.terminals_num)

        # Расчёт стартовой "плотности". Она зависит от:
        # 1) % заполненности терминала (чем больше заполненность, тем больше плотность)
        # 2) кол-ва дней, прошедших с последнего обслуживания
        # Если терминал переполнен, к плотности добавляем
        start_density = self.remains[terminals] / self.environment.terminal_limit
        start_density += (self.days_after_service[terminals] + 1) / self.environment.non_serviced_days
        start_density += np.exp(self.tl) * (self.remains[terminals] > self.environment.terminal_limit)
        start_density += np.exp(self.das_c1 * (self.days_after_service[terminals] - self.das_c2)) * (
                self.days_after_service[terminals] > self.das_c3)
        start_density = (start_density / np.linalg.norm(start_density)).clip(DensitySolver.EPSILON)

        # итоговая плотность рассчитывается на основе стартовой плотности соседних терминалов
        # (чем ближе терминал, тем больше его стартовая плотность влияет на итоговую плотность рассматриваемого)
        sub_time_matrix = self.time_matrix[np.ix_(terminals, terminals)]
        sub_time_matrix = (sub_time_matrix / np.linalg.norm(sub_time_matrix, axis=1, keepdims=True, ord=1)).clip(
            DensitySolver.EPSILON)
        log_density = logsumexp(-sub_time_matrix + np.log(start_density)[None, :], axis=1)
        log_density += DensitySolver.BIG_DENSITY * (self.remains[terminals] > self.environment.terminal_limit)
        log_density += DensitySolver.BIG_DENSITY * (
                self.days_after_service[terminals] >= self.environment.non_serviced_days - 1)
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

        # сортируем терминалы в окрестности на основе "плотности", сохраняя возможность восстановить их индексы
        cluster_ind_to_ind = {cluster_i: i for cluster_i, i in enumerate(cluster)}
        cluster_density = density[cluster]
        cluster_sorted_indecies = (-cluster_density).argsort()

        # при помощи бинпоиска и алгоритма коммивояжёра находим максимально возможное множество самых "плотных"
        # терминалов рядом с выбранным (проверяем можно ли такое множество терминалов обойти за 12 часов)
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

        # окрестность терминала зависит от кол-ва оставшихся броневиков
        # (для последнего окрестность равна множеству всех оставшихся терминалов)
        neighbourhood = len(terminals) // (self.n_trucks - cur_armored)
        return np.argpartition(time_to_terminals, neighbourhood)[:neighbourhood]

    def get_routes(self) -> List[List[int]]:
        """
        Получить маршруты для всех броневиков на текущий день.
        :return: список маршрутов для каждого броневика
        """

        # В начале дня отбираем <= n% "наиболее важных" терминалов, в которые включаем:
        # 1) переполненные терминалы (те терминалы, у которых превышен максимальный лимит)
        # 2) добранные до n% на основе количества дней до дедлайна
        overflowing = np.where(self.remains > self.environment.terminal_limit)[0]
        time_sorted = np.argsort(-self.days_after_service)
        cur_terminals = np.unique(np.concatenate([overflowing, time_sorted])[:self.num_routes_per_day])

        routes = []
        # итеративно для каждого броневика набираем терминалы для объезда
        for i in range(self.n_trucks):
            # оцениваем "плотность" (или "важность") каждого терминала
            density = np.full(self.terminals_num, -np.inf)
            density[cur_terminals] = self.get_density(cur_terminals)

            # выбираем наиболее "важный" терминал
            best_terminal = np.argmax(density)

            # находим максимально возможное множество самых "плотных" терминалов рядом с выбранным
            cluster = self.get_cluster(cur_terminals, best_terminal, i)
            routes.append(self.get_cluster_route(cluster, density))

            # обновляем множества не объеханных за день терминалов
            cur_terminals = np.setdiff1d(cur_terminals, np.array(routes[-1]))
            if len(cur_terminals) == 0:
                break

        # обновляем остатки в терминалах и дни, в течение которых терминалы не обслуживались
        self.remains[list(itertools.chain(*routes))] = 0.0
        self.days_after_service[list(itertools.chain(*routes))] = -1

        # проверяем, что нет терминалов с превышенным лимитом и истёкшим временем не обслуживания
        max_days_terminals = (self.days_after_service >= self.environment.non_serviced_days - 1).sum()
        bad_limit_terminals = (self.remains > self.environment.terminal_limit).sum()
        assert max_days_terminals == 0, f'Превышено максимальное кол-во дней для {max_days_terminals} терминалов'
        assert bad_limit_terminals == 0, f'Превышена максимальная сумма в терминале для {bad_limit_terminals} терминалов'

        return routes
