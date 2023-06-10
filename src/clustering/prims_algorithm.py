import heapq


def construct_tree(edges, N):
    '''
        Функция формирования дерева из списка ребер
    '''

    tree = [[] for _ in range(N)]
    for i, j in edges:
        if i != j:
            tree[i].append(j)
            tree[j].append(i)

    for i in range(len(tree)):
        tree[i] = list(set(tree[i]))

    return tree


def mst_prims_algorithm(distances):
    '''
        based on https://en.wikipedia.org/wiki/Prim%27s_algorithm
    '''

    mst = []
    visited = [False] * len(distances)
    edges = [(distances[0, to], 0, to) for to in range(len(distances))]
    heapq.heapify(edges)

    while edges:
        cost, frm, to = heapq.heappop(edges)
        if not visited[to]:
            visited[to] = True
            mst.append((frm, to))
            for to_next in range(len(distances)):
                cost = distances[to, to_next]
                if not visited[to_next]:
                    heapq.heappush(edges, (cost, to, to_next))

    tree = construct_tree(mst, len(distances))
    return tree
