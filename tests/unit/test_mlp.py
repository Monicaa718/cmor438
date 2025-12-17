# tests/test_mlp.py
import numpy as np
import pytest

import sys
from pathlib import Path

# add project root to python path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from MLP import SimpleMLP, softmax



# ---------- Utility Fixtures ---------- #

@pytest.fixture
def toy_data():
    """
    Small synthetic dataset for testing.
    """
    np.random.seed(42)
    X = np.random.randn(100, 20)
    y = np.random.randint(0, 3, size=100)
    return X, y


# ---------- Softmax Tests ---------- #

def test_softmax_output_shape_and_sum():
    z = np.array([[1.0, 2.0, 3.0],
                  [0.1, 0.2, 0.3]])
    probs = softmax(z)

    assert probs.shape == z.shape
    np.testing.assert_allclose(np.sum(probs, axis=1), np.ones(z.shape[0]), atol=1e-6)


def test_softmax_numerical_stability():
    z = np.array([[1000, 1000, 1000]])
    probs = softmax(z)
    assert not np.any(np.isnan(probs))
    assert not np.any(np.isinf(probs))


# ---------- Forward / Predict ---------- #

def test_forward_output_shape(toy_data):
    X, y = toy_data
    model = SimpleMLP(input_dim=20, hidden_dim=16, output_dim=3)

    probs = model.forward(X)
    assert probs.shape == (X.shape[0], 3)


def test_predict_output_range(toy_data):
    X, y = toy_data
    model = SimpleMLP(input_dim=20, hidden_dim=16, output_dim=3)

    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)
    assert preds.min() >= 0
    assert preds.max() < 3


# ---------- Training Behavior ---------- #

def test_training_updates_parameters(toy_data):
    X, y = toy_data
    model = SimpleMLP(input_dim=20, hidden_dim=16, output_dim=3, lr=0.01)

    W1_before = model.W1.copy()
    model.train(X, y, epochs=1, batch_size=32)
    W1_after = model.W1

    assert not np.allclose(W1_before, W1_after), "Weights should update after training"


def test_training_improves_over_random(toy_data):
    """
    Model should do better than random guessing on training data.
    """
    X, y = toy_data
    model = SimpleMLP(input_dim=20, hidden_dim=32, output_dim=3, lr=0.01)

    model.train(X, y, epochs=5, batch_size=16)
    preds = model.predict(X)

    acc = np.mean(preds == y)
    assert acc > 0.4   # random guess = 0.33


# ---------- Error Handling ---------- #

def test_predict_dimension_mismatch_raises():
    model = SimpleMLP(input_dim=10, hidden_dim=8, output_dim=2)
    X = np.random.randn(5, 20)

    with pytest.raises(ValueError):
        model.predict(X)
