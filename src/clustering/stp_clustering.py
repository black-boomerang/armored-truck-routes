import numpy as np
from tqdm import trange
from scipy.spatial.distance import pdist

from src.clustering.prims_algorithm import mst_prims_algorithm


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
        edges = []
        for i in range(len(self.mst)):
            for j in self.mst[i]:
                edges.append((min(i, j), max(i, j)))

        edges = np.array(list(set(edges)))
        total = (self.distances[edges[:, 0], edges[:, 1]]).sum() + 10 * len(self.distances)
        edges = [tuple(edge) for edge in edges]
        return total, edges

    def _optimal_cutting(self, i, visited, weights, total_weight):
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

    def _dfs(self, i, bridges, visited, points):
        visited[i] = 1
        points.append(i)
        total = 0
        for j in self.mst[i]:
            if not visited[j] and (i, j) not in bridges and (j, i) not in bridges:
                local = self._dfs(j, bridges, visited, points)
                total += local + self.distances[i, j] + 10

        return total

    def _fit(self, X):

        if self.metric == 'precomputed':
            distances = X
        else:
            distances = pdist(X, metric=self.metric)

        self.distances = distances
        self.mst = mst_prims_algorithm(distances)

        return self

    def _predict(self, max_length):
        total, edges = self._total_weight()

        visited = np.zeros(len(self.distances))
        weights = np.zeros_like(self.distances)
        self._optimal_cutting(0, visited, weights, total)
        rows, cols = np.where(weights < max_length)
        opt_idx = weights[weights < max_length].argmax()
        opt_idx = rows[opt_idx], cols[opt_idx]
        assert opt_idx in edges or opt_idx[::-1] in edges

        points1, points2 = [], []
        visited = np.zeros(len(self.distances))
        s1 = self._dfs(opt_idx[0], [opt_idx], visited, points1)
        s2 = self._dfs(opt_idx[1], [opt_idx], visited, points2)

        if s1 < max_length and s2 < max_length:
            return np.array(points1), np.array(points2)

        clusters = np.array(points1) if s1 < s2 else np.array(points2)

        return clusters, None
    
    def fit_predict(self, time_matrix):
        time_matrix_copy = time_matrix.copy()
        rests = [np.arange(len(time_matrix_copy))]
        clusters_ = []

        while True:

            if self.symmetric:
                inputs = np.maximum(np.tril(time_matrix_copy), np.triu(time_matrix_copy).T)
                inputs = inputs + inputs.T
            else:
                inputs = time_matrix_copy

            cluster = self._fit(inputs)._predict(max_length=self.max_length)
            if cluster[1] is None:
                cluster = cluster[0]
                clusters_.append(cluster)
            else:
                clusters_.extend(cluster)
                break
            
            rest = np.array(list(set(np.arange(len(time_matrix_copy))).difference(cluster)))
            time_matrix_copy = time_matrix_copy[np.ix_(rest, rest)]
            rests.append(rest)

        clusters_ = clusters_[::-1]
        rests = rests[::-1]

        clusters = []
        for i in range(len(clusters_)):
            cluster = clusters_[i]

            if i > 0:
                i -= 1

            for r in rests[i:]:
                cluster = r[cluster]

            clusters.append(cluster)
            assert len(list(set().union(*clusters))) == sum([len(c) for c in clusters])

        return clusters
        