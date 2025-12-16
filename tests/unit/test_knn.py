import numpy as np
import pytest

from rice_ml.supervised_learning.knn import KNNClassifier, KNNRegressor


def test_classifier_basic_predict_and_proba_uniform_euclidean():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 1, 1])

    clf = KNNClassifier(n_neighbors=3, metric="euclidean", weights="uniform").fit(X, y)

    preds = clf.predict([[0.1, 0.1], [0.9, 0.9]])
    assert preds.tolist() == [0, 1]

    proba = clf.predict_proba([[0.1, 0.1], [0.9, 0.9]])
    assert proba.shape == (2, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.argmax(proba, axis=1).tolist() == [0, 1]


def test_classifier_distance_weights_exact_match():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([7, 8, 9])

    clf = KNNClassifier(n_neighbors=3, metric="euclidean", weights="distance").fit(X, y)

    pred = clf.predict([[0.0]])
    assert pred.tolist() == [7]

    proba = clf.predict_proba([[0.0]])
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.argmax(proba, axis=1).tolist() == [0]  # class 7 is first (sorted)


def test_classifier_kneighbors_shapes_and_order():
    X = np.array([[0.0], [2.0], [5.0], [9.0]])
    y = np.array([0, 0, 1, 1])

    clf = KNNClassifier(n_neighbors=2, metric="euclidean", weights="uniform").fit(X, y)
    dist, idx = clf.kneighbors([[6.0]], return_distance=True)

    assert dist.shape == (1, 2)
    assert idx.shape == (1, 2)
    # nearest to 6.0 are 5.0 then 9.0
    assert idx.tolist() == [[2, 3]]
    assert dist[0, 0] <= dist[0, 1]


def test_classifier_score_accuracy():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 1, 1])

    clf = KNNClassifier(n_neighbors=1, metric="euclidean", weights="uniform").fit(X, y)
    assert clf.score(X, y) == 1.0


def test_regressor_uniform_and_distance():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 10.0, 20.0])

    reg_u = KNNRegressor(n_neighbors=3, metric="euclidean", weights="uniform").fit(X, y)
    yhat_u = reg_u.predict([[1.0]])
    assert np.allclose(yhat_u, [10.0])

    reg_d = KNNRegressor(n_neighbors=3, metric="euclidean", weights="distance").fit(X, y)
    yhat_d = reg_d.predict([[1.0]])
    assert np.allclose(yhat_d, [10.0])  # exact match handled


def test_regressor_score_r2():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 10.0, 20.0])

    reg = KNNRegressor(n_neighbors=1, metric="euclidean", weights="uniform").fit(X, y)
    assert reg.score(X, y) == 1.0


def test_invalid_args_and_not_fitted():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 1, 1])

    with pytest.raises(ValueError):
        KNNClassifier(n_neighbors=0)

    with pytest.raises(ValueError):
        KNNClassifier(metric="cosine")

    with pytest.raises(ValueError):
        KNNClassifier(weights="foo")

    clf = KNNClassifier(n_neighbors=1)
    with pytest.raises(ValueError):
        clf.predict([[0.0]])

    with pytest.raises(ValueError):
        KNNClassifier(n_neighbors=5).fit(X, y)  # k > n_samples

    reg = KNNRegressor(n_neighbors=1)
    with pytest.raises(ValueError):
        reg.predict([[0.0]])
