from abc import ABC, abstractmethod
from typing import List

import numpy as np

from task import BusinessLogic


class BaseSolver(ABC):
    """ Базовый класс решения задачи """

    def __init__(self, remains: np.ndarray, time_matrix: np.ndarray, business_logic: BusinessLogic):
        self.remains = remains
        self.time_matrix = time_matrix
        self.terminals_num = self.time_matrix.shape[0]

        self.days_after_service = np.zeros(self.terminals_num)
        self.bl = business_logic

    def update(self, incomes: np.ndarray) -> None:
        """ Обновить состояние решателя на новый день """
        self.remains += incomes
        self.days_after_service -= 1

    @abstractmethod
    def get_routes(self) -> List[List[int]]:
        """
        Получить маршруты для всех броневиков на текущий день.
        :return: список маршрутов для каждого броневика
        """
