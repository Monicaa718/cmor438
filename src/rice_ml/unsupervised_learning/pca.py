# file: pca.py
"""
Principal Component Analysis (NumPy-only).

This module provides a lightweight, dependency-free PCA implementation with an
API modeled after scikit-learn.

It supports:
- PCA with SVD ("full" solver via np.linalg.svd)
- fit / transform / fit_transform / inverse_transform
- attributes: components_, mean_, explained_variance_, explained_variance_ratio_,
  singular_values_, n_components_, n_features_in_

Notes
-----
- PCA is computed on mean-centered data.
- Components are orthonormal directions (rows of components_).
- Sign of components is not unique (a component can be multiplied by -1).

Examples
--------
>>> import numpy as np
>>> from pca import PCA
>>> X = np.array([[1., 2.], [3., 4.], [5., 6.]])
>>> pca = PCA(n_components=1).fit(X)
>>> Z = pca.transform(X)
>>> Z.shape
(3, 1)
>>> X_hat = pca.inverse_transform(Z)
>>> X_hat.shape
(3, 2)
"""

from __future__ import annotations
from typing import Optional, Sequence, Union

import numpy as np

__all__ = ["PCA"]

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]]]


def _ensure_2d_float(X: ArrayLike, name: str = "X") -> np.ndarray:
    """Ensure X is a non-empty 2D numeric ndarray of dtype float."""
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got {arr.ndim}D.")
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty.")
    if not np.issubdtype(arr.dtype, np.number):
        try:
            arr = arr.astype(float, copy=False)
        except (TypeError, ValueError) as e:
            raise TypeError(f"All elements of {name} must be numeric.") from e
    else:
        arr = arr.astype(float, copy=False)
    return arr


def _validate_n_components(n_components: Optional[Union[int, float]]) -> None:
    """
    Validate n_components.

    Supported:
    - None: keep all components (min(n_samples, n_features))
    - int:  1..min(n_samples, n_features)
    - float in (0, 1]: choose smallest k with cumulative explained variance ratio >= float
    """
    if n_components is None:
        return
    if isinstance(n_components, (int, np.integer)):
        if int(n_components) < 1:
            raise ValueError("n_components must be >= 1 when an integer.")
        return
    if isinstance(n_components, (float, np.floating)):
        x = float(n_components)
        if not (0.0 < x <= 1.0):
            raise ValueError("n_components as float must be in (0, 1].")
        return
    raise TypeError("n_components must be None, an int, or a float in (0, 1].")


class PCA:
    """
    Principal Component Analysis (sklearn-like).

    Parameters
    ----------
    n_components : int | float | None, default=None
        - None: keep all components (k = min(n_samples, n_features))
        - int: number of components to keep
        - float in (0,1]: keep the smallest k such that cumulative explained variance
          ratio >= n_components
    whiten : bool, default=False
        If True, transform outputs are scaled by 1/sqrt(explained_variance_),
        producing unit-variance components (on training data, approximately).

    Attributes (after fit)
    ----------------------
    components_ : ndarray of shape (n_components_, n_features)
        Principal axes in feature space.
    mean_ : ndarray of shape (n_features,)
        Feature-wise mean of the training data.
    explained_variance_ : ndarray of shape (n_components_,)
        Variance explained by each component.
    explained_variance_ratio_ : ndarray of shape (n_components_,)
        Fraction of total variance explained by each component.
    singular_values_ : ndarray of shape (n_components_,)
        Singular values corresponding to each selected component.
    n_components_ : int
        The number of components kept.
    n_features_in_ : int
        Number of features seen during fit.
    """

    def __init__(self, n_components: Optional[Union[int, float]] = None, *, whiten: bool = False) -> None:
        _validate_n_components(n_components)
        self.n_components = n_components
        self.whiten = bool(whiten)

        # learned attributes
        self.components_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.singular_values_: Optional[np.ndarray] = None
        self.n_components_: Optional[int] = None
        self.n_features_in_: Optional[int] = None

    def _check_is_fitted(self) -> None:
        if (
            self.components_ is None
            or self.mean_ is None
            or self.explained_variance_ is None
            or self.explained_variance_ratio_ is None
            or self.singular_values_ is None
            or self.n_components_ is None
            or self.n_features_in_ is None
        ):
            raise RuntimeError("PCA is not fitted. Call fit(X) first.")

    def fit(self, X: ArrayLike) -> "PCA":
        X_arr = _ensure_2d_float(X, "X")
        n_samples, n_features = X_arr.shape
        if n_samples < 2:
            raise ValueError("PCA requires at least 2 samples.")

        self.n_features_in_ = int(n_features)
        self.mean_ = X_arr.mean(axis=0)

        Xc = X_arr - self.mean_

        # Full SVD of centered data
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)  # Vt shape: (r, n_features)
        r = min(n_samples, n_features)

        # explained variance for each PC (sklearn convention): S^2 / (n_samples - 1)
        explained_var_all = (S ** 2) / (n_samples - 1)
        total_var = explained_var_all.sum()

        # choose k
        if self.n_components is None:
            k = r
        elif isinstance(self.n_components, (int, np.integer)):
            k = int(self.n_components)
            if k > r:
                raise ValueError(f"n_components={k} cannot exceed min(n_samples, n_features)={r}.")
        else:
            # float: smallest k such that cumulative ratio >= threshold
            thresh = float(self.n_components)
            ratios = explained_var_all / total_var if total_var > 0 else np.zeros_like(explained_var_all)
            cumsum = np.cumsum(ratios)
            k = int(np.searchsorted(cumsum, thresh, side="left") + 1)
            k = max(1, min(k, r))

        self.n_components_ = int(k)

        self.components_ = Vt[:k, :].astype(float, copy=False)
        self.singular_values_ = S[:k].astype(float, copy=False)
        self.explained_variance_ = explained_var_all[:k].astype(float, copy=False)
        self.explained_variance_ratio_ = (
            (self.explained_variance_ / total_var).astype(float, copy=False) if total_var > 0 else np.zeros(k, float)
        )

        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        self._check_is_fitted()
        X_arr = _ensure_2d_float(X, "X")
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X_arr.shape[1]} features, expected {self.n_features_in_}.")

        Xc = X_arr - self.mean_
        Z = Xc @ self.components_.T  # (n_samples, k)

        if self.whiten:
            # scale each component by 1/sqrt(var)
            scale = np.sqrt(np.maximum(self.explained_variance_, 1e-12))
            Z = Z / scale[None, :]

        return Z.astype(float, copy=False)

    def fit_transform(self, X: ArrayLike) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_transformed: ArrayLike) -> np.ndarray:
        self._check_is_fitted()
        Z = _ensure_2d_float(X_transformed, "X_transformed")
        if Z.shape[1] != self.n_components_:
            raise ValueError(f"X_transformed has {Z.shape[1]} features, expected {self.n_components_}.")

        if self.whiten:
            scale = np.sqrt(np.maximum(self.explained_variance_, 1e-12))
            Z = Z * scale[None, :]

        X_rec = Z @ self.components_ + self.mean_[None, :]
        return X_rec.astype(float, copy=False)
