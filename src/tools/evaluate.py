from typing import List, Tuple

import numpy as np
import pandas as pd

from src.report import create_report, create_final_report, PathReporter
from src.solvers import BaseSolver
from src.task import Environment


def get_losses(paths: List[List[int]], remains: np.ndarray, environment: Environment) -> Tuple[
    np.ndarray, np.ndarray, float]:
    """ Вычисление издержек за заданный день с учётом остатков в терминалах и пройденных маршрутов """

    terminals_count = len(remains)
    visited_terminals = set(np.concatenate(paths))
    non_visited_terminals = set(np.arange(terminals_count)).difference(visited_terminals)
    non_visited_terminals = map(lambda x: np.array(list(x)), [visited_terminals, non_visited_terminals])

    # издержки на обслуживание терминалов
    visited_terminals = np.array(list(visited_terminals))
    cashable_loss = np.zeros(terminals_count)
    cashable_loss[visited_terminals] = environment.get_cashable_loss(remains[visited_terminals])

    # издержки на деньги, оставшиеся в терминалах
    non_visited_terminals = np.array(list(non_visited_terminals))
    non_cashable_loss = np.zeros(terminals_count)
    non_cashable_loss[non_visited_terminals] = environment.get_non_cashable_loss(remains[non_visited_terminals])

    total = np.sum(cashable_loss) + np.sum(non_cashable_loss)
    return cashable_loss, non_cashable_loss, total


def evaluate(solver: BaseSolver, terminals: pd.DataFrame, first_day: int = 1, last_day: int = 91) -> None:
    """ Оценка издержек для заданного решателя """
    reported = PathReporter(n_trucks=9, terminal_ids=terminals['TID'], time_matrix=solver.time_matrix)

    remains = []
    cashable = []
    non_cashable = []
    for day in range(first_day, last_day + 1):
        # получаем маршруты на день
        paths = solver.get_routes()

        # посчитываем издержки
        cashable_loss, non_cashable_loss, total = get_losses(paths, solver.remains, solver.environment)
        remains.append(solver.remains)
        cashable.append(cashable_loss)
        non_cashable.append(non_cashable_loss)

        solver.update(terminals[f'day {day}'].values)
        reported.evaluate(paths, day=day)

    create_report(remains, terminals, filename='остатки на конец дня')
    create_report(cashable, terminals, filename='стоимость инкассации')
    create_report(non_cashable, terminals, filename='стоимость фондирования')
    create_final_report(cashable, non_cashable, N=9)
