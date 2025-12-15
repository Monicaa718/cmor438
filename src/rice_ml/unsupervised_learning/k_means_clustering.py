import numpy as np


def kmeans(X: np.ndarray, k: int, max_iters: int = 100, tol: float = 1e-4):
    """
    K-Means clustering from scratch.

    Parameters
    - X: (n_samples, n_features) data matrix
    - k: number of clusters
    - max_iters: maximum iterations
    - tol: convergence tolerance on center shift

    Returns
    - labels: array of cluster assignments per sample
    - centers: (k, n_features) array of final cluster centers
    """
    # Input validation
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a numpy.ndarray")
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if not np.issubdtype(X.dtype, np.number):
        raise TypeError("X must contain numeric values")
    n, d = X.shape
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError("k must be an integer")
    if k <= 0:
        raise ValueError("k must be positive")
    if k > n:
        raise ValueError("k cannot be greater than the number of samples")
    if not isinstance(max_iters, int) or isinstance(max_iters, bool):
        raise TypeError("max_iters must be an integer")
    if max_iters <= 0:
        raise ValueError("max_iters must be positive")
    if not (isinstance(tol, float) or isinstance(tol, int)):
        raise TypeError("tol must be a float or int")
    if tol <= 0:
        raise ValueError("tol must be positive")
    idx = np.random.choice(n, k, replace=False)
    centers = X[idx].copy()
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iters):
        # distances to centers via efficient squared Euclidean computation
        x_sq = np.sum(X**2, axis=1, keepdims=True)
        c_sq = np.sum(centers**2, axis=1, keepdims=True).T
        d2 = x_sq + c_sq - 2 * (X @ centers.T)
        labels = np.argmin(d2, axis=1)

        new_centers = np.zeros_like(centers)
        for j in range(k):
            pts = X[labels == j]
            if pts.size == 0:
                new_centers[j] = X[np.random.randint(0, n)]
            else:
                new_centers[j] = pts.mean(axis=0)

        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        if shift < tol:
            break

    return labels, centers
