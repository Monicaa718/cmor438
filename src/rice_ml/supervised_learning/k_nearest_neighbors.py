"""
k-Nearest Neighbors (NumPy-only).

This module provides simple, dependency-free KNN models suitable for teaching
and lightweight usage. It supports:
- Classification (`KNNClassifier`) with `predict` and `predict_proba`
- Regression (`KNNRegressor`) with `predict`
- Distance metrics: 'euclidean', 'manhattan', 'chebyshev'
- Weighting: 'uniform' or 'distance'
- Convenience methods: `kneighbors`, `score`

The API is intentionally similar to scikit-learn's KNeighborsClassifier /
KNeighborsRegressor (fit/predict/predict_proba/kneighbors/score).

Examples
--------
>>> import numpy as np
>>> from rice_ml.supervised_learning.knn import KNNClassifier
>>> X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
>>> y = np.array([0, 0, 1, 1])
>>> clf = KNNClassifier(n_neighbors=3, metric="euclidean", weights="uniform").fit(X, y)
>>> clf.predict([[0.1, 0.1], [0.9, 0.9]]).tolist()
[0, 1]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, Union

import numpy as np

Metric = Literal["euclidean", "manhattan", "chebyshev"]
Weights = Literal["uniform", "distance"]


def _as_2d_float(X, *, name: str) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"{name} must be a 2D array-like, got shape {X.shape}.")
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty, got shape {X.shape}.")
    return X.astype(float, copy=False)


def _as_1d(y, *, name: str) -> np.ndarray:
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"{name} must be a 1D array-like, got shape {y.shape}.")
    if y.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty.")
    return y


def _check_n_neighbors(n_neighbors: int) -> int:
    if not isinstance(n_neighbors, (int, np.integer)):
        raise TypeError("n_neighbors must be an int.")
    n_neighbors = int(n_neighbors)
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be >= 1.")
    return n_neighbors


def _check_metric(metric: str) -> Metric:
    if metric not in ("euclidean", "manhattan", "chebyshev"):
        raise ValueError("metric must be one of {'euclidean','manhattan','chebyshev'}.")
    return metric  # type: ignore[return-value]


def _check_weights(weights: str) -> Weights:
    if weights not in ("uniform", "distance"):
        raise ValueError("weights must be one of {'uniform','distance'}.")
    return weights  # type: ignore[return-value]


def _pairwise_distances(Xq: np.ndarray, Xt: np.ndarray, metric: Metric) -> np.ndarray:
    """
    Compute pairwise distances between query and train samples.

    Xq: (n_query, n_features)
    Xt: (n_train, n_features)
    Returns: (n_query, n_train)
    """
    diff = Xq[:, None, :] - Xt[None, :, :]  # (nq, nt, d)

    if metric == "euclidean":
        return np.sqrt(np.sum(diff * diff, axis=2))
    if metric == "manhattan":
        return np.sum(np.abs(diff), axis=2)
    # chebyshev
    return np.max(np.abs(diff), axis=2)


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape for R^2.")

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))

    # sklearn-like handling for constant y_true:
    # - if ss_tot == 0 and predictions are perfect => 1.0
    # - else => 0.0
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


@dataclass
class _KNNBase:
    n_neighbors: int = 5
    metric: Metric = "euclidean"
    weights: Weights = "uniform"

    def __post_init__(self) -> None:
        self.n_neighbors = _check_n_neighbors(self.n_neighbors)
        self.metric = _check_metric(self.metric)
        self.weights = _check_weights(self.weights)

        self.X_: Optional[np.ndarray] = None
        self.y_: Optional[np.ndarray] = None

    def fit(self, X, y):
        X = _as_2d_float(X, name="X")
        y = _as_1d(y, name="y")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        if self.n_neighbors > X.shape[0]:
            raise ValueError(
                f"n_neighbors={self.n_neighbors} cannot exceed n_samples={X.shape[0]}."
            )

        self.X_ = X
        self.y_ = y
        return self

    def _check_is_fitted(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.X_ is None or self.y_ is None:
            raise ValueError("This estimator is not fitted yet. Call 'fit' first.")
        return self.X_, self.y_

    def kneighbors(self, X, n_neighbors: Optional[int] = None, return_distance: bool = True):
        """
        Find k-nearest neighbors of X.

        Parameters
        ----------
        X : array-like of shape (n_query, n_features)
        n_neighbors : int or None
            If None, uses self.n_neighbors.
        return_distance : bool
            If True, return (distances, indices). Else return indices.

        Returns
        -------
        distances : ndarray of shape (n_query, k)
        indices : ndarray of shape (n_query, k)
        OR
        indices : ndarray of shape (n_query, k)
        """
        Xt, _ = self._check_is_fitted()
        Xq = _as_2d_float(X, name="X")
        if Xq.shape[1] != Xt.shape[1]:
            raise ValueError(
                f"X must have the same n_features as training data: "
                f"{Xq.shape[1]} != {Xt.shape[1]}"
            )

        k = self.n_neighbors if n_neighbors is None else _check_n_neighbors(n_neighbors)
        if k > Xt.shape[0]:
            raise ValueError(f"n_neighbors={k} cannot exceed n_samples={Xt.shape[0]}.")

        D = _pairwise_distances(Xq, Xt, metric=self.metric)  # (nq, nt)

        # argpartition for top-k, then sort those k
        idx_part = np.argpartition(D, kth=k - 1, axis=1)[:, :k]  # (nq, k)
        dist_part = np.take_along_axis(D, idx_part, axis=1)

        order = np.argsort(dist_part, axis=1)
        idx = np.take_along_axis(idx_part, order, axis=1)
        dist = np.take_along_axis(dist_part, order, axis=1)

        if return_distance:
            return dist, idx
        return idx


class KNNClassifier(_KNNBase):
    """
    kNN classifier (NumPy-only), sklearn-like API.

    Attributes after fit
    --------------------
    classes_ : ndarray of shape (n_classes,)
    """

    def fit(self, X, y):
        super().fit(X, y)
        _, y_arr = self._check_is_fitted()
        self.classes_ = np.unique(y_arr)
        return self

    def predict_proba(self, X):
        Xt, y = self._check_is_fitted()
        if not hasattr(self, "classes_"):
            self.classes_ = np.unique(y)

        dist, idx = self.kneighbors(X, return_distance=True)
        y_nn = y[idx]  # (nq, k)

        classes = self.classes_
        # encode y onto 0..C-1 (based on classes_)
        class_to_index = {c: i for i, c in enumerate(classes)}
        y_nn_enc = np.vectorize(class_to_index.get, otypes=[int])(y_nn)

        nq, k = y_nn_enc.shape
        C = classes.shape[0]
        proba = np.zeros((nq, C), dtype=float)

        if self.weights == "uniform":
            for i in range(nq):
                counts = np.bincount(y_nn_enc[i], minlength=C).astype(float)
                proba[i] = counts / counts.sum()
            return proba

        # distance weights
        for i in range(nq):
            di = dist[i]
            yi = y_nn_enc[i]

            zero_mask = di == 0.0
            if np.any(zero_mask):
                yi0 = yi[zero_mask]
                counts = np.bincount(yi0, minlength=C).astype(float)
                proba[i] = counts / counts.sum()
            else:
                w = 1.0 / di
                for cls_idx, wi in zip(yi, w):
                    proba[i, cls_idx] += wi
                proba[i] /= proba[i].sum()

        return proba

    def predict(self, X):
        proba = self.predict_proba(X)
        # tie-breaker: argmax returns first max => smaller class index (sorted classes_)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X, y):
        """
        Classification accuracy.
        """
        y_true = _as_1d(y, name="y")
        y_pred = self.predict(X)
        if y_pred.shape[0] != y_true.shape[0]:
            raise ValueError("X and y must have compatible lengths.")
        return float(np.mean(y_pred == y_true))


class KNNRegressor(_KNNBase):
    """
    kNN regressor (NumPy-only), sklearn-like API.
    """

    def fit(self, X, y):
        X = _as_2d_float(X, name="X")
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError("y must be a 1D array-like for regression.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")
        if self.n_neighbors > X.shape[0]:
            raise ValueError(
                f"n_neighbors={self.n_neighbors} cannot exceed n_samples={X.shape[0]}."
            )
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        Xt, y = self._check_is_fitted()
        dist, idx = self.kneighbors(X, return_distance=True)
        y_nn = y[idx]  # (nq, k)

        if self.weights == "uniform":
            return np.mean(y_nn, axis=1)

        # distance weights (handle exact match like sklearn)
        nq = y_nn.shape[0]
        out = np.empty(nq, dtype=float)
        for i in range(nq):
            di = dist[i]
            yi = y_nn[i]
            zero_mask = di == 0.0
            if np.any(zero_mask):
                out[i] = float(np.mean(yi[zero_mask]))
            else:
                w = 1.0 / di
                out[i] = float(np.sum(w * yi) / np.sum(w))
        return out

    def score(self, X, y):
        """
        R^2 score.
        """
        y_true = np.asarray(y, dtype=float)
        if y_true.ndim != 1:
            raise ValueError("y must be 1D for regression score.")
        y_pred = self.predict(X)
        return _r2_score(y_true, y_pred)
