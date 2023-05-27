from typing import List, Tuple

import networkx as nx
import numpy as np

tsp = nx.approximation.traveling_salesman_problem


def tsp_solution(distances: np.ndarray) -> Tuple[List[int], float]:
    """
    Решение задачи коммивояжёра с 1.5-приближением.
    :param distances: матрица расстояний
    :return: (найденный путь, расстояние по нему)
    """
    sym_distances = np.maximum(np.tril(distances), np.triu(distances).T)
    sym_distances = sym_distances + sym_distances.T
    G = nx.from_numpy_array(sym_distances)

    if sym_distances.shape[0] > 1:
        path = tsp(G, cycle=False)
    else:
        path = [0]

    path_distance = sum([G[path[i]][path[i + 1]]['weight']
                        for i in range(len(path) - 1)])
    return path, path_distance
