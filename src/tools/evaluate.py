from typing import List

import numpy as np
import pandas as pd

from src.report import create_report, create_final_report, PathReporter
from src.solvers import BaseSolver
from src.task import Environment


def get_losses(paths: List[List[int]], remains: np.ndarray, terminals_count: int,
               environment: Environment) -> np.ndarray:
    visited_path = set(np.concatenate(paths))
    non_visited_paths = set(np.arange(terminals_count)).difference(visited_path)
    visited_path, non_visited_paths = map(lambda x: np.array(list(x)), [visited_path, non_visited_paths])

    cashable_loss = np.zeros(terminals_count)
    cashable_loss[visited_path] = environment.get_cashable_loss(remains[visited_path])

    non_cashable_loss = np.zeros(terminals_count)
    non_cashable_loss[non_visited_paths] = environment.get_non_cashable_loss(remains[non_visited_paths])

    total = np.sum(cashable_loss) + np.sum(non_cashable_loss)
    return total


def evaluate(solver: BaseSolver, terminals: pd.DataFrame, first_day: int = 1, last_day: int = 91) -> None:
    """ Оценка издержек для заданного решателя """
    reported = PathReporter(n_trucks=9, terminal_ids=terminals['TID'], time_matrix=solver.time_matrix)

    remains = []
    cashable = []
    non_cashable = []
    for day in range(first_day, last_day + 1):
        paths = solver.get_routes()
        visited_path = set(np.concatenate(paths))
        non_visited_paths = set(np.arange(len(terminals))).difference(visited_path)
        visited_path, non_visited_paths = map(lambda x: np.array(list(x)), [visited_path, non_visited_paths])

        remains.append(solver.remains)

        cashable_loss = np.zeros(len(terminals))
        cashable_loss[visited_path] = solver.environment.get_cashable_loss(solver.remains[visited_path])
        cashable.append(cashable_loss)

        non_cashable_loss = np.zeros(len(terminals))
        non_cashable_loss[non_visited_paths] = solver.environment.get_non_cashable_loss(
            solver.remains[non_visited_paths])
        non_cashable.append(non_cashable_loss)

        solver.update(terminals[f'day {day}'].values)
        reported.evaluate(paths, day=day)
        # save_map(paths, terminals, day)

    create_report(remains, terminals, filename='остатки на конец дня')
    create_report(cashable, terminals, filename='стоимость инкассации')
    create_report(non_cashable, terminals, filename='стоимость фондирования')
    create_final_report(cashable, non_cashable, N=9)

    # create_gif()
