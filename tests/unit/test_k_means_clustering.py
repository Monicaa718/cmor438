import numpy as np

from rice_ml.unsupervised_learning.k_means_clustering import kmeans


def _make_three_clusters(n_per=50, seed=0):
    rng = np.random.default_rng(seed)
    c1 = rng.normal(loc=(-5.0, 0.0), scale=0.5, size=(n_per, 2))
    c2 = rng.normal(loc=(0.0, 5.0), scale=0.5, size=(n_per, 2))
    c3 = rng.normal(loc=(5.0, -5.0), scale=0.5, size=(n_per, 2))
    X = np.vstack([c1, c2, c3])
    return X


def test_kmeans_three_compact_clusters_counts_and_shapes():
    np.random.seed(42)  # control initialization
    X = _make_three_clusters(n_per=40, seed=123)
    labels, centers = kmeans(X, k=3)
    # shapes
    assert labels.shape == (X.shape[0],)
    assert centers.shape == (3, X.shape[1])
    # three clusters discovered
    uniq = np.unique(labels)
    assert len(uniq) == 3
    # reasonably balanced counts (due to synthetic construction)
    counts = np.bincount(labels)
    assert counts.min() >= 30 and counts.max() <= 50
    # centers well separated
    dists = []
    for i in range(3):
        for j in range(i + 1, 3):
            dists.append(np.linalg.norm(centers[i] - centers[j]))
    assert np.min(dists) > 5.0


def test_kmeans_returns_stable_when_tol_large():
    np.random.seed(0)
    X = _make_three_clusters(n_per=10, seed=11)
    labels, centers = kmeans(X, k=3, max_iters=5, tol=1e6)
    # With huge tol, it should stop after first update; still returns proper shapes
    assert labels.shape == (X.shape[0],)
    assert centers.shape == (3, X.shape[1])
