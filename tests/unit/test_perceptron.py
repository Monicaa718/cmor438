import numpy as np
import pytest

import sys
from pathlib import Path

# add project root to python path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from perceptron import PerceptronBinary, PerceptronOVR

# ========================
# Binary Perceptron
# ========================

def test_binary_perceptron_basic_fit_and_predict():
    # linearly separable dataset
    X = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ])
    y = np.array([0, 0, 1, 1])

    clf = PerceptronBinary(
        n_features=2,
        lr=0.1,
        n_epochs=10
    ).fit(X, y)

    preds = clf.predict(X)
    assert preds.shape == y.shape
    assert np.array_equal(preds, y)


def test_binary_perceptron_score_accuracy():
    X = np.array([
        [0.0],
        [1.0],
        [2.0],
        [3.0]
    ])
    y = np.array([0, 0, 1, 1])

    clf = PerceptronBinary(
        n_features=1,
        lr=0.1,
        n_epochs=10
    ).fit(X, y)

    acc = clf.score(X, y)
    assert acc == 1.0


def test_binary_perceptron_wrong_feature_dimension():
    X = np.array([[0.0, 1.0]])
    y = np.array([0])

    clf = PerceptronBinary(
        n_features=2
    ).fit(X, y)

    # wrong input dimension at predict time
    with pytest.raises(ValueError):
        clf.predict([[0.0, 1.0, 2.0]])


def test_binary_perceptron_non_binary_labels_error():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0, 1, 2])  # invalid for binary perceptron

    with pytest.raises(ValueError):
        PerceptronBinary(n_features=1).fit(X, y)


# ========================
# Multiclass Perceptron (OvR)
# ========================

def test_multiclass_perceptron_wrong_feature_dimension():
    X = np.array([[0.0, 1.0]])
    y = np.array([0])

    clf = PerceptronOVR(
        n_classes=1,
        n_features=2
    ).fit(X, y)

    with pytest.raises(ValueError):
        clf.predict([[0.0, 1.0, 2.0]])


def test_multiclass_perceptron_invalid_labels():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array(["a", "b", "c"], dtype=object)

    with pytest.raises(TypeError):
        PerceptronOVR(
            n_classes=3,
            n_features=1
        ).fit(X, y)


# ========================
# Edge cases
# ========================

def test_perceptron_predict_before_fit():
    clf = PerceptronBinary(n_features=2)
    with pytest.raises(RuntimeError):
        clf.predict([[0.0, 0.0]])


def test_multiclass_perceptron_single_sample():
    X = np.array([[1.0, 1.0]])
    y = np.array([0])

    clf = PerceptronOVR(
        n_classes=1,
        n_features=2,
        n_epochs=5
    ).fit(X, y)

    pred = clf.predict([[1.0, 1.0]])
    assert pred.tolist() == [0]
