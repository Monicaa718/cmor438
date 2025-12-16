import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from randomforest import RandomForestRegressorScratch
from xgboost_scratch import XGBoostRegressorScratch


@pytest.fixture
def synthetic_data():
    """
    Generate a simple synthetic regression dataset.
    This fixture is shared by all tests.
    """
    rng = np.random.default_rng(42)
    X = rng.normal(size=(200, 3))
    y = 0.5 * X[:, 0] - 0.3 * X[:, 1] + rng.normal(scale=0.1, size=200)
    return X, y


@pytest.fixture
def train_test_split(synthetic_data):
    """
    Train / test split fixture.
    """
    X, y = synthetic_data
    split = int(0.8 * len(X))
    return (
        X[:split],
        X[split:],
        y[:split],
        y[split:]
    )


def test_random_forest_fit_predict(train_test_split):
    """
    Random Forest should:
    - fit without error
    - return predictions of correct shape
    """
    X_train, X_test, y_train, y_test = train_test_split

    rf = RandomForestRegressorScratch(
        n_estimators=50,
        max_depth=3,
        random_state=0
    )

    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)

    assert preds.shape == y_test.shape
    assert np.isfinite(preds).all()


def test_xgboost_fit_predict(train_test_split):
    """
    XGBoost-style model should:
    - fit without error
    - return predictions of correct shape
    """
    X_train, X_test, y_train, y_test = train_test_split

    xgb = XGBoostRegressorScratch(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        random_state=0
    )

    xgb.fit(X_train, y_train)
    preds = xgb.predict(X_test)

    assert preds.shape == y_test.shape
    assert np.isfinite(preds).all()


def test_models_can_be_ensembled(train_test_split):
    """
    RF and XGB predictions should be compatible
    for ensembling and stacking.
    """
    X_train, X_test, y_train, y_test = train_test_split

    rf = RandomForestRegressorScratch(
        n_estimators=30,
        max_depth=3,
        random_state=0
    )
    xgb = XGBoostRegressorScratch(
        n_estimators=30,
        learning_rate=0.1,
        max_depth=3,
        random_state=0
    )

    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    xgb_pred = xgb.predict(X_test)

    # Shape compatibility
    assert rf_pred.shape == xgb_pred.shape == y_test.shape

    # Simple averaging ensemble
    ensemble_pred = 0.5 * rf_pred + 0.5 * xgb_pred
    assert ensemble_pred.shape == y_test.shape
