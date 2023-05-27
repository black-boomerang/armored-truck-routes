import heapq
import numpy as np
from tqdm import trange
from scipy.spatial.distance import pdist


class STPClustering:
    '''
        Implementation of Spanning Tree Pruning Clustering Algorithm
    '''
    
    def __init__(self, metric='precomputed'):

        self.metric = metric
        self.mst = None

    def _construct_tree(self, edges, N):
        tree = [[] for _ in range(N)]
        for i, j in edges:
            tree[i].append(j)
            tree[j].append(i)

        for i in range(len(tree)):
            tree[i] = list(set(tree[i]))

        return tree

    def _mst_prims_algorithm(self):
        mst = []
        visited = [False] * len(self.distances)
        edges = [(self.distances[0, to], 0, to) for to in range(len(self.distances))]
        heapq.heapify(edges)

        while edges:
            cost, frm, to = heapq.heappop(edges)
            if not visited[to]:
                visited[to] = True
                mst.append((frm, to))
                for to_next in range(len(self.distances)):
                    cost = self.distances[to, to_next]
                    if not visited[to_next]:
                        heapq.heappush(edges, (cost, to, to_next))

        tree = self._construct_tree(mst, len(self.distances))
        return tree
    
    def _stp_clustering(self, i, visited, edges, max_length=720):
        visited[i] = 1
        sums = 0
        for j in self.mst[i]:
            if not visited[j]:
                s = self._stp_clustering(j, visited, edges, max_length)

                if s + self.distances[i, j] > max_length:
                    edges.append((i, j))
                else:
                    sums += s + self.distances[i, j] + 10

        return sums

    def _dfs(self, i, bridges, visited, points):
        visited[i] = 1
        points.append(i)
        for j in self.mst[i]:
            if not visited[j] and (i, j) not in bridges and (j, i) not in bridges:
                self._dfs(j, bridges, visited, points)

    def fit(self, X):

        if self.metric == 'precomputed':
            distances = X
        else:
            distances = pdist(X, metric=self.metric)

        self.distances = distances
        self.mst = self._mst_prims_algorithm()

        return self

    def predict(self, max_length):
        bridges = []
        visited = np.zeros(len(self.distances))
        self._stp_clustering(0, visited, bridges, max_length=max_length)
        bridges = list(set(bridges))

        clusters = []
        visited = np.zeros(len(self.distances))
        for i in range(len(self.distances)):
            points = []
            if not visited[i]:
                self._dfs(i, bridges, visited, points)
                clusters.append(np.array(points))

        assert sum([len(cluster) for cluster in clusters]) == len(self.distances), 'some points weren"t clustered'
        assert len(clusters) == len(bridges) + 1

        return clusters
