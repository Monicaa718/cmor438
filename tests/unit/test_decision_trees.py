import numpy as np
import pytest

from rice_ml.supervised_learning.decision_trees import DecisionTreeClassifier

def test_depth_effect(capsys):
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    y = np.array([0, 1, 1, 0])  # XOR pattern
    # shallow tree
    t1 = DecisionTreeClassifier(max_depth=1, random_state=0)
    t1.fit(X, y)
    p1 = t1.predict(X)
    # deeper tree
    t2 = DecisionTreeClassifier(max_depth=2, random_state=0)
    t2.fit(X, y)
    p2 = t2.predict(X)
    # force printing even when stdout is captured
    with capsys.disabled():
        print("y =", y)
        print("depth1 =", p1)
        print("depth2 =", p2)
    # shallow tree cannot fit XOR
    assert not np.array_equal(p1, y)
    # deeper tree can fit XOR perfectly
    assert np.array_equal(p2, y)

def test_predict_shape_and_values():
    X = np.array([
        [0],
        [1],
        [2],
        [3],
    ])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTreeClassifier(max_depth=2, random_state=42)
    tree.fit(X, y)
    y_pred = tree.predict(X)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == (len(X),)
    assert set(np.unique(y_pred)).issubset(set(np.unique(y)))

def test_predict_proba_shape_and_sum():
    X = np.array([
        [0],
        [1],
        [2],
        [3],
    ])
    y = np.array([0, 0, 1, 1])
    tree = DecisionTreeClassifier(max_depth=2, random_state=0)
    tree.fit(X, y)
    proba = tree.predict_proba(X)
    assert proba.shape == (len(X), tree.n_classes_)
    assert np.allclose(proba.sum(axis=1), 1.0)
