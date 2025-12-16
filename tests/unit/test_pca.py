# file: test_pca.py
import numpy as np
import pytest

from pca import PCA


def test_pca_fit_attributes_and_shapes():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 5))

    pca = PCA(n_components=3).fit(X)

    assert pca.mean_.shape == (5,)
    assert pca.components_.shape == (3, 5)
    assert pca.explained_variance_.shape == (3,)
    assert pca.explained_variance_ratio_.shape == (3,)
    assert pca.singular_values_.shape == (3,)
    assert pca.n_components_ == 3
    assert pca.n_features_in_ == 5

    # components should be orthonormal: C C^T = I
    G = pca.components_ @ pca.components_.T
    assert np.allclose(G, np.eye(3), atol=1e-10)

    # explained variance should be non-increasing (numerically)
    assert np.all(np.diff(pca.explained_variance_) <= 1e-12)

    # ratios are between 0 and 1, and sum <= 1
    assert np.all(pca.explained_variance_ratio_ >= -1e-12)
    assert np.all(pca.explained_variance_ratio_ <= 1.0 + 1e-12)
    assert pca.explained_variance_ratio_.sum() <= 1.0 + 1e-12


def test_pca_fit_transform_equivalence():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 4))

    pca1 = PCA(n_components=2)
    Z1 = pca1.fit_transform(X)

    pca2 = PCA(n_components=2).fit(X)
    Z2 = pca2.transform(X)

    assert np.allclose(Z1, Z2, atol=1e-12)


def test_pca_inverse_transform_full_reconstruction():
    # If n_components == n_features (and n_samples > n_features), reconstruction should be exact (up to numerical error)
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 6))

    pca = PCA(n_components=6).fit(X)
    Z = pca.transform(X)
    X_hat = pca.inverse_transform(Z)

    assert X_hat.shape == X.shape
    assert np.allclose(X_hat, X, atol=1e-10)


def test_pca_explained_variance_ratio_sum_close_to_one_for_full_components():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 5))

    pca = PCA(n_components=5).fit(X)
    assert np.isclose(pca.explained_variance_ratio_.sum(), 1.0, atol=1e-12)


def test_pca_float_n_components_keeps_enough_variance():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(60, 8))

    pca = PCA(n_components=0.80).fit(X)
    assert 1 <= pca.n_components_ <= min(X.shape)
    assert pca.explained_variance_ratio_.sum() >= 0.80 - 1e-12


def test_pca_whiten_unit_variance_on_training_data():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(200, 5))

    pca = PCA(n_components=3, whiten=True).fit(X)
    Z = pca.transform(X)

    # On training data, whitened components should have ~unit variance (ddof=1)
    var = np.var(Z, axis=0, ddof=1)
    assert np.allclose(var, np.ones(3), atol=1e-2, rtol=1e-2)


def test_pca_errors_and_input_validation():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(10, 3))

    # transform before fit
    with pytest.raises(RuntimeError):
        PCA(n_components=2).transform(X)

    # non-2D input
    with pytest.raises(ValueError):
        PCA().fit(np.array([1.0, 2.0, 3.0]))

    # empty input
    with pytest.raises(ValueError):
        PCA().fit(np.empty((0, 3)))

    # too few samples
    with pytest.raises(ValueError):
        PCA().fit(np.ones((1, 3)))

    # invalid n_components
    with pytest.raises(ValueError):
        PCA(n_components=0)

    with pytest.raises(ValueError):
        PCA(n_components=10).fit(X)  # exceeds min(n_samples, n_features)=3

    with pytest.raises(ValueError):
        PCA(n_components=1.5)

    with pytest.raises(TypeError):
        PCA(n_components="2")  # type: ignore

    # feature mismatch on transform
    pca = PCA(n_components=2).fit(X)
    with pytest.raises(ValueError):
        pca.transform(rng.normal(size=(5, 4)))

    # inverse_transform dimension mismatch
    Z = pca.transform(X)
    with pytest.raises(ValueError):
        pca.inverse_transform(Z[:, :1])
