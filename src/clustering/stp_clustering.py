import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist

from src.clustering.prims_algorithm import mst_prims_algorithm

tsp = nx.approximation.traveling_salesman_problem


class STPClustering:
    '''
        Implementation of Spanning Tree Pruning Clustering Algorithm
    '''
    
    def __init__(self, max_length=720, metric='precomputed', symmetric=False):

        self.max_length = max_length
        self.symmetric = symmetric
        self.metric = metric
        self.mst = None

    def _total_weight(self):

        '''
            Функция подсчета общего веса дерева и формирования списка ребер
        '''

        edges = []
        for i in range(len(self.mst)):
            for j in self.mst[i]:
                edges.append((min(i, j), max(i, j)))

        edges = np.array(list(set(edges)))
        total = (self.distances[edges[:, 0], edges[:, 1]]).sum() + 10 * len(self.distances)
        edges = [tuple(edge) for edge in edges]
        return total, edges

    def _optimal_cutting(self, i, visited, weights, total_weight):

        '''
            Функция для определения оптимального разреза на основе алгоритма DFS
        '''

        visited[i] = True
        total = 0
        for j in self.mst[i]:
            if not visited[j]:
                local = self._optimal_cutting(j, visited, weights, total_weight)
                # print(total_weight - local - self.distances[i, j])
                weights[i, j] = min(local + 10, total_weight - local - 10 - self.distances[i, j])
                weights[j, i] = min(local + 10, total_weight - local - 10 - self.distances[j, i])
                total += local + self.distances[i, j] + 10

        return total

    def _fit(self, X):

        '''
            "Обучение" алгоритма - построение минимального остовного дерева алгоритмом Прима
        '''

        if self.metric == 'precomputed':
            distances = X
        else:
            distances = pdist(X, metric=self.metric)

        self.distances = distances
        self.mst = mst_prims_algorithm(distances)

        return self
    
    def tsp_solution(self, distances):

        '''
            Решение задачи коммивояжера алгоритмом Кристифодеса
        '''

        sym_distances = np.maximum(np.tril(distances), np.triu(distances).T)
        sym_distances = sym_distances + sym_distances.T
        G = nx.from_numpy_array(sym_distances)

        if sym_distances.shape[0] > 1:
            path = tsp(G, cycle=False)
        else:
            path = [0]

        src, dst = path[:-1], path[1:]
        elapsed = (distances[src, dst]).sum() + 10 * len(path)
            
        return path, elapsed
    
    def _find_optimal_cutting(self, max_length):

        '''
            Функция для инициализации и постпроцессинга алгоритма по определению оптимального разреза
        '''

        total, edges = self._total_weight()

        visited = np.zeros(len(self.distances))
        weights = np.zeros_like(self.distances)
        self._optimal_cutting(0, visited, weights, total)
        rows, cols = np.where(weights < max_length)
        opt_idx = weights[weights < max_length].argmax()
        opt_idx = rows[opt_idx], cols[opt_idx]
        assert opt_idx in edges or opt_idx[::-1] in edges

        return opt_idx
    
    def _get_subtree_weight(self, i, bridges, visited, points):

        '''
            Функция для обхода поддерева на основе алгоритма DFS
        '''

        visited[i] = 1
        points.append(i)
        total = 0
        for j in self.mst[i]:
            if not visited[j] and (i, j) not in bridges and (j, i) not in bridges:
                local = self._get_subtree_weight(j, bridges, visited, points)
                total += local + self.distances[i, j] + 10

        return total
    
    def _tree_cutting(self, opt_idx):

        '''
            Разбиение дерева на основе оптимального разреза
        '''

        set1, set2 = [], []
        visited = np.zeros(len(self.distances))
        weight1 = self._get_subtree_weight(opt_idx[0], [opt_idx], visited, set1)
        weight2 = self._get_subtree_weight(opt_idx[1], [opt_idx], visited, set2)

        return weight1, weight2, set1, set2
    
    def _increase_cluster(self, cluster):

        '''
            Функция для увеличения размера кластера
        '''

        while True:
            dst = np.argsort(self.distances[cluster], axis=1)

            candidates = []
            for row in dst:
                for v in row:
                    if v not in cluster:
                        candidates.append(v)
                        break

            dst = np.array(candidates)
            candidate = dst[self.distances[cluster, dst].argmin()]
            cluster_ = np.append(cluster, candidate)
            if self.tsp_solution(self.distances[np.ix_(cluster_, cluster_)])[1] < 720:
                cluster = cluster_
            else:
                break

        return cluster

    def _predict(self, max_length):

        '''
            "Предсказание" алгоритма - выделение кластера наибольшего веса, не превышающего заданного
        '''

        opt_idx = self._find_optimal_cutting(max_length)
        weight1, weight2, set1, set2 = self._tree_cutting(opt_idx)
        
        if weight1 < max_length and weight2 < max_length:
            cluster = np.array(set1 + set2)
            if self.tsp_solution(self.distances[np.ix_(cluster, cluster)])[1] < 720: # если укладываемся в 12 часов, то совмещаем оба множества
                return cluster, None, True
            else:
                return set1, set2, True # ...иначе оставляем 2 исходных

        cluster = np.array(set1) if weight1 < weight2 else np.array(set2)
        cluster = self._increase_cluster(cluster) # пытаемся увеличить кластер
        return cluster, None, False
    

    def _clustering(self, time_matrix):

        '''
            Функция кластеризации
        '''

        time_matrix_copy = time_matrix.copy()
        rests = [np.arange(len(time_matrix_copy))]
        clusters = []

        merge_last = True
        while True:

            if self.symmetric:
                inputs = np.maximum(np.tril(time_matrix_copy), np.triu(time_matrix_copy).T)
                inputs = inputs + inputs.T
            else:
                inputs = time_matrix_copy

            cluster1, cluster2, finish = self._fit(inputs)._predict(max_length=self.max_length)
            if cluster2 is None:
                cluster = cluster1
                clusters.append(cluster1)
            else:
                merge_last = False
                clusters.extend([cluster1, cluster2])
            
            if finish:
                break
            
            rest = np.array(list(set(np.arange(len(time_matrix_copy))).difference(cluster)))
            time_matrix_copy = time_matrix_copy[np.ix_(rest, rest)]
            rests.append(rest)

        clusters = clusters[::-1]
        rests = rests[::-1]
        return clusters, rests, merge_last

    def _remap_idx(self, clusters, rests, merge_last):

        '''
            Функция кластеризации
        '''

        clusters_ = []
        for i in range(len(clusters)):
            cluster = clusters[i]

            if i > 0 and not merge_last:
                i -= 1

            for r in rests[i:]:
                cluster = r[cluster]

            clusters_.append(cluster)
            assert len(list(set().union(*clusters_))) == sum([len(c) for c in clusters_])

        return clusters_
    
    def fit_predict(self, time_matrix):

        '''
            Функция кластеризации
        '''

        clusters, rests, merge_last = self._clustering(time_matrix) # выполняем кластеризацию
        clusters = self._remap_idx(clusters, rests, merge_last) # возвращаемся к исходным индексам

        return clusters
        