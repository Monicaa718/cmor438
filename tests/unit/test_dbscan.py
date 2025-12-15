import numpy as np

from rice_ml.unsupervised_learning.dbscan import dbscan


def _make_two_clusters_with_noise(n_per=40, n_noise=15, seed=0):
    rng = np.random.default_rng(seed)
    c1 = rng.normal(loc=(-3.0, -3.0), scale=0.4, size=(n_per, 2))
    c2 = rng.normal(loc=(3.0, 3.0), scale=0.4, size=(n_per, 2))
    noise = rng.uniform(low=-6.0, high=6.0, size=(n_noise, 2))
    X = np.vstack([c1, c2, noise])
    return X


def test_dbscan_finds_two_clusters_and_noise():
    X = _make_two_clusters_with_noise(n_per=35, n_noise=20, seed=123)
    labels, n_clusters = dbscan(X, eps=0.7, min_samples=6)
    # Two dense clusters discovered
    assert n_clusters == 2
    # Some noise is expected
    assert np.sum(labels == -1) > 0
    # Non-noise labels count equals cluster sizes returned
    uniq = np.unique(labels[labels >= 0])
    assert len(uniq) == 2


def test_dbscan_parameter_sensitivity_eps_small_more_noise():
    X = _make_two_clusters_with_noise(n_per=35, n_noise=20, seed=321)
    labels_lo, n_lo = dbscan(X, eps=0.3, min_samples=6)
    labels_hi, n_hi = dbscan(X, eps=0.9, min_samples=6)
    # Smaller eps typically yields more noise
    assert np.sum(labels_lo == -1) >= np.sum(labels_hi == -1)
    # Both settings should produce at least one cluster on this synthetic data
    assert n_lo >= 1 and n_hi >= 1
