import numpy as np
from collections import deque
from ..supervised_learning.distance_metrics import euclidean_distance

__all__ = ["dbscan"]

def pairwise_distances(X: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise Euclidean distances between rows of a data matrix.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Input data matrix where each row corresponds to a sample and each column to a feature.

    Returns
    -------
    D : np.ndarray of shape (n_samples, n_samples)
        Symmetric matrix of pairwise Euclidean distances. D[i, j] is the distance between X[i] and X[j].

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[0, 0], [1, 0], [0, 1]])
    >>> pairwise_distances(X)
    array([[0.        , 1.        , 1.        ],
           [1.        , 0.        , 1.41421356],
           [1.        , 1.41421356, 0.        ]])
    """
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
    Density-Based Spatial Clustering of Applications with Noise (DBSCAN).

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features).
    eps : float, optional
        Neighborhood radius. Points within this distance are considered neighbors. Default is 0.5.
    min_samples : int, optional
        Minimum number of points required to form a dense region (core point). Default is 5.

    Returns
    -------
    labels : np.ndarray
        Array of shape (n_samples,) with cluster labels. Noise points are labeled as -1.
    n_clusters : int
        Number of discovered clusters.
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
