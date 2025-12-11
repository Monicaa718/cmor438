import numpy as np
from collections import deque
from ..supervised_learning.distance_metrics import euclidean_distance


def pairwise_distances(X: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances using the shared distance function."""
    n = X.shape[0]
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        D[i, i] = 0.0
        for j in range(i + 1, n):
            d = euclidean_distance(X[i], X[j])
            D[i, j] = d
            D[j, i] = d
    return D


def neighbors_within_eps(D: np.ndarray, i: int, eps: float) -> np.ndarray:
    return np.where(D[i] <= eps)[0]


def dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 5):
    """
    Density-Based Spatial Clustering of Applications with Noise.

    Parameters
    - X: (n_samples, n_features) data matrix
    - eps: neighborhood radius
    - min_samples: minimum number of points to form a dense region

    Returns
    - labels: array of length n_samples with cluster id, -1 for noise
    - n_clusters: number of discovered clusters
    """
    n = X.shape[0]
    D = pairwise_distances(X)
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        nbrs = neighbors_within_eps(D, i, eps)
        if nbrs.size < min_samples:
            labels[i] = -1
        else:
            labels[i] = cluster_id
            queue = deque(nbrs.tolist())
            while queue:
                j = queue.popleft()
                if not visited[j]:
                    visited[j] = True
                    j_nbrs = neighbors_within_eps(D, j, eps)
                    if j_nbrs.size >= min_samples:
                        for k in j_nbrs:
                            if labels[k] == -1:
                                labels[k] = cluster_id
                            if labels[k] == -1 or not visited[k]:
                                queue.append(k)
                if labels[j] == -1:
                    labels[j] = cluster_id
            cluster_id += 1

    return labels, cluster_id
