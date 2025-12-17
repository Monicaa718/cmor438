"""src.rice_ml.supervised_learning.regression_trees

CART Regression Tree (from scratch) — algorithm only.

What this module includes
- Node data structure
- SSE impurity
- Best categorical split (binary)
- Recursive tree building (greedy SSE minimization)
- Prediction

What this module intentionally does NOT include
- Data loading / cleaning
- Train-test split
- Metrics (MSE/RMSE/MAE/R²)
- Visualization utilities

This keeps the file as “source code of the algorithm” and lets your notebook
handle preprocessing, evaluation, and plotting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------- Impurity (SSE) ----------
def sse(y: np.ndarray) -> float:
    """Sum of Squared Errors around the mean."""
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return 0.0
    m = float(np.mean(y))
    return float(np.sum((y - m) ** 2))


# ---------- Tree Node ----------
@dataclass
class Node:
    # statistics
    prediction: float
    n_samples: int
    node_sse: float

    # split definition (None => leaf)
    feature: Optional[str] = None
    left_values: Optional[Tuple[Any, ...]] = None  # categorical values routed to left

    # children
    left: Optional["Node"] = None
    right: Optional["Node"] = None

    def is_leaf(self) -> bool:
        return self.feature is None


# ---------- Categorical split search ----------
def _unique_values_in_order(s: pd.Series) -> List[Any]:
    seen: List[Any] = []
    for v in s.tolist():
        if v not in seen:
            seen.append(v)
    return seen


def _nontrivial_subsets(values: Sequence[Any]) -> Iterable[Tuple[Any, ...]]:
    """
    Generate non-trivial subsets for a binary split on categorical values.

    We avoid mirrored duplicates (S vs complement) by generating subsets up to size floor(k/2).
    When k is even and r == k-r, only keep subsets containing values[0].
    """
    import itertools

    vals = list(values)
    k = len(vals)
    if k <= 1:
        return []

    out: List[Tuple[Any, ...]] = []
    max_r = k // 2
    for r in range(1, max_r + 1):
        for comb in itertools.combinations(vals, r):
            if r == k - r and vals[0] not in comb:
                continue
            out.append(comb)
    return out


def best_split_categorical(
    X: pd.DataFrame,
    y: np.ndarray,
    feature: str,
    min_samples_leaf: int,
    *,
    max_exhaustive_categories: int = 8,
) -> Optional[Dict[str, Any]]:
    """
    Find the best binary split for a categorical feature under SSE.

    Returns a dict:
      {
        'feature': <feature>,
        'left_values': (<values routed left>),
        'left_mask': boolean np.ndarray,
        'sse_split': float
      }

    Strategy:
    - If #unique values <= max_exhaustive_categories: exhaustive subset enumeration.
    - Else: order categories by mean(y) and try prefix splits (heuristic).
    """
    y = np.asarray(y, dtype=float)
    col = X[feature].to_numpy()
    uniq = _unique_values_in_order(X[feature])

    if len(uniq) <= 1:
        return None

    def eval_subset(left_vals: Sequence[Any]) -> Optional[Dict[str, Any]]:
        left_set = set(left_vals)
        left_mask = np.fromiter((v in left_set for v in col), dtype=bool, count=len(col))
        right_mask = ~left_mask
        if int(left_mask.sum()) < min_samples_leaf or int(right_mask.sum()) < min_samples_leaf:
            return None
        sse_split = sse(y[left_mask]) + sse(y[right_mask])
        return {
            "feature": feature,
            "left_values": tuple(left_vals),
            "left_mask": left_mask,
            "sse_split": float(sse_split),
        }

    best: Optional[Dict[str, Any]] = None
    best_sse = float("inf")

    # 1) Exhaustive search for small cardinality
    if len(uniq) <= max_exhaustive_categories:
        for subset in _nontrivial_subsets(uniq):
            cand = eval_subset(subset)
            if cand is None:
                continue
            if cand["sse_split"] < best_sse:
                best_sse = cand["sse_split"]
                best = cand
        return best

    # 2) Heuristic: order categories by mean(y), try prefix splits
    tmp = pd.DataFrame({"v": col, "y": y})
    stats = tmp.groupby("v", as_index=False)["y"].mean().sort_values("y", ascending=True)
    ordered = stats["v"].tolist()

    for cut in range(1, len(ordered)):
        cand = eval_subset(ordered[:cut])
        if cand is None:
            continue
        if cand["sse_split"] < best_sse:
            best_sse = cand["sse_split"]
            best = cand

    return best


# ---------- Tree building (greedy CART regression tree) ----------
def build_regression_tree(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    max_depth: int,
    min_samples_split: int,
    min_samples_leaf: int,
    min_gain: float = 1e-6,
    max_exhaustive_categories: int = 8,
    _depth: int = 0,
) -> Node:
    """
    Build a CART-style regression tree using greedy SSE minimization.

    Notes:
    - All splits are binary.
    - Categorical splits are of the form: feature in subset -> left else right.
    """
    y = np.asarray(y, dtype=float)
    pred = float(np.mean(y)) if y.size else 0.0
    node_sse = sse(y)
    node = Node(prediction=pred, n_samples=int(len(y)), node_sse=node_sse)

    # stopping rules
    if _depth >= max_depth:
        return node
    if len(y) < min_samples_split:
        return node

    best: Optional[Dict[str, Any]] = None
    best_sse = float("inf")

    for f in X.columns:
        cand = best_split_categorical(
            X, y, f,
            min_samples_leaf=min_samples_leaf,
            max_exhaustive_categories=max_exhaustive_categories,
        )
        if cand is None:
            continue
        if cand["sse_split"] < best_sse:
            best_sse = cand["sse_split"]
            best = cand

    if best is None:
        return node

    gain = node_sse - best_sse
    if gain < float(min_gain):
        return node

    left_mask = best["left_mask"]
    right_mask = ~left_mask

    node.feature = best["feature"]
    node.left_values = best["left_values"]

    node.left = build_regression_tree(
        X.loc[left_mask].reset_index(drop=True),
        y[left_mask],
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_gain=min_gain,
        max_exhaustive_categories=max_exhaustive_categories,
        _depth=_depth + 1,
    )
    node.right = build_regression_tree(
        X.loc[right_mask].reset_index(drop=True),
        y[right_mask],
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_gain=min_gain,
        max_exhaustive_categories=max_exhaustive_categories,
        _depth=_depth + 1,
    )
    return node


# ---------- Prediction ----------
def predict_one(node: Node, row: pd.Series) -> float:
    cur = node
    while not cur.is_leaf():
        if row[cur.feature] in set(cur.left_values):
            cur = cur.left  # type: ignore[assignment]
        else:
            cur = cur.right  # type: ignore[assignment]
    return float(cur.prediction)


def predict(tree: Node, X: pd.DataFrame) -> np.ndarray:
    return np.array([predict_one(tree, X.iloc[i]) for i in range(len(X))], dtype=float)
