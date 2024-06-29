from collections import defaultdict

import numpy as np

inf = 0x3f3f3f3f


def floyd(adjs):
    n = adjs.shape[0]
    dist = np.full((n, n), inf)
    path = np.full((n, n), -1)
    for i in range(n):
        for j in range(n):
            if i == j:
                dist[i, j] = 0
            elif adjs[i, j] != 0:
                dist[i, j] = adjs[i, j]
                path[i, j] = i

    all_path_result = defaultdict(dict)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] + dist[k, j] < dist[i, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
                    path[i, j] = k

    for i in range(n):
        for j in range(n):
            all_path_result[i][j] = get_single_path(i, j, path)

    return dist, all_path_result


def get_single_path(u, v, path):
    k = path[u, v]
    if k == -1 or path[u, v] == u:
        return []
    else:
        return get_single_path(u, k, path) + [k] + get_single_path(k, v, path)


if __name__ == "__main__":
    A = np.array([[0, 0, 0, 0, 1],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0],
                  [1, 0, 0, 0, 0]])
    floyd(A)
