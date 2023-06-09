from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import pandas as pd

from src.task import Environment


class BaseSolver(ABC):
    """ Базовый класс решения задачи """

    def __init__(self, remains: np.ndarray, time_matrix: np.ndarray, environment: Environment, n_trucks: int):
        self.remains = remains
        self.time_matrix = time_matrix
        self.terminals_num = self.time_matrix.shape[0]

        self.days_after_service = np.zeros(self.terminals_num, dtype=int)
        self.environment = environment
        self.n_trucks = n_trucks

    def update(self, incomes: Union[np.ndarray, pd.Series]) -> None:
        """ Обновить состояние решателя на новый день """
        if isinstance(incomes, pd.Series):
            incomes = incomes.values
        self.remains += incomes
        self.days_after_service += 1

    @abstractmethod
    def get_routes(self) -> List[List[int]]:
        """
        Получить маршруты для всех броневиков на текущий день.
        :return: список маршрутов для каждого броневика
        """
