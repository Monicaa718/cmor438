import numpy as np
import pandas as pd
import pytest

from rice_ml.supervised_learning.regression_trees import build_regression_tree, predict

def test_depth_effect(capsys):
    X = pd.DataFrame({"f1": ["A", "A", "B", "B"], "f2": ["X", "Y", "X", "Y"]})
    y = np.array([0.0, 1.0, 1.0, 0.0])
    t1 = build_regression_tree(X, y, max_depth=1, min_samples_split=2, min_samples_leaf=1, min_gain=0.0)
    p1 = predict(t1, X)
    t2 = build_regression_tree(X, y, max_depth=2, min_samples_split=2, min_samples_leaf=1, min_gain=0.0)
    p2 = predict(t2, X)
    # force printing even when stdout is captured (e.g., PyCharm runner)
    with capsys.disabled():
        print("y =", y)
        print("depth1 =", p1)
        print("depth2 =", p2)
    assert not np.allclose(p1, y)
    assert np.allclose(p2, y)

def test_predict_shape():
    X = pd.DataFrame({"f1": ["A", "B", "C"], "f2": ["X", "Y", "X"]})
    y = np.array([0.0, 1.0, 2.0])
    t = build_regression_tree(X, y, max_depth=2, min_samples_split=2, min_samples_leaf=1, min_gain=0.0)
    p = predict(t, X)
    assert p.shape == (len(X),)
