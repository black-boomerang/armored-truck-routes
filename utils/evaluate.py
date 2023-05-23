import pandas as pd

from solvers import BaseSolver


def get_day_expenses() -> float:
    """ Оценка издержек за текущий день """
    return 0


def evaluate(solver: BaseSolver, terminals: pd.DataFrame, first_day: int = 1, last_day: int = 91) -> None:
    """ Оценка издержек для заданного решателя """
    expenses = 0.0

    for day in range(first_day, last_day + 1):
        solver.get_routes()
        expenses += get_day_expenses()
        solver.update(terminals[f'day {day}'].values)

    print(f'Издержки: {expenses}')
