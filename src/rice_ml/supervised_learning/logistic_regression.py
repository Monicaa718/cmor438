import numpy as np


class LogisticRegression:
    """
    Binary Logistic Regression using gradient descent.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to include an intercept term in the model.

    learning_rate : float, default=0.1
        Step size for gradient descent.

    max_iter : int, default=1000
        Maximum number of gradient descent iterations.

    tol : float, default=1e-6
        Convergence tolerance based on parameter change norm.

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features,)
        Learned feature coefficients.

    intercept_ : float
        Learned intercept term (0 if fit_intercept=False).

    weights_ : np.ndarray of shape (n_features + 1,) or (n_features,)
        Full parameter vector including intercept (if used).

    loss_history_ : list of float
        Logistic loss value at each iteration.
    """

    def __init__(
        self,
        fit_intercept: bool = True,
        learning_rate: float = 0.1,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.weights_: np.ndarray | None = None
        self.loss_history_: list[float] = []

    # ----------------- internal helpers -----------------

    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add a column of ones to X if fit_intercept=True."""
        if not self.fit_intercept:
            return X
        n_samples = X.shape[0]
        ones = np.ones((n_samples, 1), dtype=float)
        return np.hstack((ones, X))

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        # clip to avoid overflow / log(0)
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _log_loss(self, y_true: np.ndarray, p_hat: np.ndarray) -> float:
        """Binary cross-entropy loss."""
        eps = 1e-15
        p_hat = np.clip(p_hat, eps, 1 - eps)
        return -np.mean(
            y_true * np.log(p_hat) + (1.0 - y_true) * np.log(1.0 - p_hat)
        )

    # ----------------- public API -----------------

    def fit(self, X, y):
        """
        Fit the logistic regression model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        y : array-like, shape (n_samples,)
            Binary labels {0, 1}.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if y.ndim != 1:
            raise ValueError("y must be a 1D array of binary labels {0, 1}.")

        X_design = self._add_intercept(X)
        n_samples, n_features = X_design.shape

        # initialize weights to zeros for determinism
        w = np.zeros(n_features, dtype=float)
        self.loss_history_ = []

        for _ in range(self.max_iter):
            # predicted probabilities
            z = X_design @ w
            p_hat = self._sigmoid(z)

            # compute loss
            loss = self._log_loss(y, p_hat)
            self.loss_history_.append(loss)

            # gradient of log-loss
            grad = X_design.T @ (p_hat - y) / n_samples

            # gradient descent update
            w_new = w - self.learning_rate * grad

            # check convergence
            if np.linalg.norm(w_new - w, ord=2) < self.tol:
                w = w_new
                break

            w = w_new

        self.weights_ = w

        if self.fit_intercept:
            self.intercept_ = float(w[0])
            self.coef_ = w[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = w

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities for input samples.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, 2)
            Column 0 = P(class 0), column 1 = P(class 1).
        """
        if self.weights_ is None:
            raise RuntimeError("You must call fit before predict_proba.")

        X = np.asarray(X, dtype=float)
        X_design = self._add_intercept(X)
        z = X_design @ self.weights_
        p1 = self._sigmoid(z)
        p0 = 1.0 - p1
        return np.column_stack((p0, p1))

    def predict(self, X, threshold: float = 0.5):
        """
        Predict binary labels {0, 1} for input samples.

        Parameters
        ----------
        threshold : float, default=0.5
            Decision threshold on P(class 1).
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def accuracy(self, X, y, threshold: float = 0.5) -> float:
        """
        Compute classification accuracy on given data.
        """
        y = np.asarray(y)
        y_pred = self.predict(X, threshold=threshold)
        return float(np.mean(y_pred == y))

    def log_loss(self, X, y) -> float:
        """
        Compute logistic loss on given data using current model parameters.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        X_design = self._add_intercept(X)
        z = X_design @ self.weights_
        p_hat = self._sigmoid(z)
        return self._log_loss(y, p_hat)
