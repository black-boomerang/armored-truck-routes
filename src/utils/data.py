import os
from typing import Tuple

import numpy as np
import pandas as pd


def get_data(data_path) -> Tuple[pd.DataFrame, np.ndarray]:
    """ Загрузка данных о терминалах и времени между ними """

    terminal_data_path = os.path.join(data_path, 'terminal_data_hackathon v4.xlsx')

    # местоположения терминалов
    points = pd.read_excel(terminal_data_path, sheet_name='TIDS')
    terminals_num = len(points)

    # считываем доходы + считаем накопленные остатки на конец каждого дня
    incomes_columns = ['TID', 'start_value'] + [f'day {i + 1}' for i in range(91)]
    incomes = pd.read_excel(terminal_data_path, sheet_name='Incomes', names=incomes_columns)
    add_columns = [f'remains {i + 1}' for i in range(91)]
    incomes[add_columns] = incomes[incomes_columns[1:]].values.cumsum(axis=1)[:, 1:]

    # объединяем доходы терминалов и их местоположения
    terminals = points.set_index('TID').join(incomes.set_index('TID')).reset_index()
    tid_to_ind = {tid: i for i, tid in enumerate(terminals.TID)}

    # создаём матрицу времени
    times_data_path = os.path.join(data_path, 'times v4.csv')
    times = pd.read_csv(times_data_path)
    times['Origin_index'] = times['Origin_tid'].apply(lambda tid: tid_to_ind[tid])
    times['Destination_index'] = times['Destination_tid'].apply(lambda tid: tid_to_ind[tid])
    t_pairs = times[['Origin_index', 'Destination_index']].values
    time_matrix = np.zeros((terminals_num, terminals_num))
    time_matrix[t_pairs[:, 0], t_pairs[:, 1]] = times['Total_Time']

    return terminals, time_matrix
