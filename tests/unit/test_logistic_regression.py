import os
import sys
import unittest
import numpy as np

# 1. Make sure we can import from src/
SRC_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

# 2. NOW IMPORT YOUR MODULE (Must happen AFTER step 1)
from rice_ml.supervised_learning.logistic_regression import LogisticRegression


class TestLogisticRegression(unittest.TestCase):
    def setUp(self):
        """
        Runs before every test.

        We create a simple synthetic binary classification dataset.
        Points are drawn from two Gaussians that are linearly separable
        in 2D. This makes it easy for logistic regression to learn a
        good decision boundary.
        """
        rng = np.random.default_rng(seed=42)

        n_samples_per_class = 100

        # Class 0: centered at (-2, -2)
        mean0 = np.array([-2.0, -2.0])
        cov0 = np.array([[1.0, 0.2], [0.2, 1.0]])
        X0 = rng.multivariate_normal(mean0, cov0, size=n_samples_per_class)
        y0 = np.zeros(n_samples_per_class, dtype=int)

        # Class 1: centered at (+2, +2)
        mean1 = np.array([2.0, 2.0])
        cov1 = np.array([[1.0, -0.2], [-0.2, 1.0]])
        X1 = rng.multivariate_normal(mean1, cov1, size=n_samples_per_class)
        y1 = np.ones(n_samples_per_class, dtype=int)

        # Combine into a single dataset
        self.X = np.vstack([X0, X1])
        self.y = np.concatenate([y0, y1])

        # Shuffle the dataset so classes are mixed
        indices = rng.permutation(self.X.shape[0])
        self.X = self.X[indices]
        self.y = self.y[indices]

        # Create a model instance we can reuse in tests
        self.model = LogisticRegression(
            fit_intercept=True,
            learning_rate=0.1,
            max_iter=2000,
            tol=1e-6,
        )

    def test_fit_runs(self):
        """Test that fit() runs without errors and sets coef_ / intercept_."""
        self.model.fit(self.X, self.y)

        self.assertIsNotNone(self.model.coef_)
        self.assertTrue(
            isinstance(self.model.coef_, np.ndarray),
            msg="coef_ should be a numpy array.",
        )
        self.assertIsInstance(
            self.model.intercept_, (float, int),
            msg="intercept_ should be a scalar float.",
        )

    def test_predicted_probabilities(self):
        """Test that predicted probabilities are in [0, 1] and sum to 1."""
        self.model.fit(self.X, self.y)
        proba = self.model.predict_proba(self.X)

        # shape check: (n_samples, 2)
        self.assertEqual(proba.shape, (self.X.shape[0], 2))

        # all probabilities between 0 and 1
        self.assertTrue(np.all(proba >= 0.0) and np.all(proba <= 1.0))

        # each row sums (approximately) to 1
        row_sums = proba.sum(axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0, atol=1e-6))

    def test_training_accuracy_is_high(self):
        """
        Since the synthetic data are linearly separable, the model
        should achieve high accuracy on the training set.
        """
        self.model.fit(self.X, self.y)
        acc = self.model.accuracy(self.X, self.y, threshold=0.5)

        # We expect at least 0.9 accuracy on this easy problem
        self.assertGreaterEqual(
            acc, 0.9,
            msg=f"Training accuracy too low: {acc:.3f}",
        )

    def test_log_loss_decreases_over_time(self):
        """
        If loss_history_ is recorded, it should generally decrease
        during gradient descent.
        """
        self.model.fit(self.X, self.y)
        history = self.model.loss_history_

        # Make sure we actually recorded something
        self.assertGreater(len(history), 1)

        # Check that the final loss is not larger than the initial loss
        self.assertLessEqual(
            history[-1], history[0] + 1e-6,
            msg="Final loss should be <= initial loss.",
        )


if __name__ == "__main__":
    unittest.main()
