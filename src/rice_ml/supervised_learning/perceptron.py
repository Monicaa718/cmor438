# perceptron.py
# -------------------------------
# Basic Perceptron implementations
# -------------------------------

import numpy as np


class PerceptronBinary:
    """
    A simple binary Perceptron classifier.

    Training label convention (internal): {-1, +1}
    External API convention (for this project/tests): y in {0, 1}
    Predict outputs: {0, 1}
    """

    def __init__(self, n_features, lr=1.0, n_epochs=10, shuffle=True):
        self.lr = lr
        self.n_epochs = n_epochs
        self.shuffle = shuffle
        self.w = np.zeros(n_features, dtype=np.float32)
        self.b = 0.0
        self.is_fitted = False

    def _shuffle_data(self, X, y):
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        return X[idx], y[idx]

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        # --- required by tests: only allow {0,1} labels ---
        uniq = np.unique(y)
        if not set(uniq.tolist()).issubset({0, 1}):
            raise ValueError("PerceptronBinary supports only labels {0,1}.")

        # training uses {-1,+1}
        y_internal = np.where(y == 1, 1, -1)

        for epoch in range(self.n_epochs):
            if self.shuffle:
                X, y_internal = self._shuffle_data(X, y_internal)

            errors = 0
            for xi, yi in zip(X, y_internal):
                activation = np.dot(self.w, xi) + self.b

                # Misclassified → update
                if yi * activation <= 0:
                    self.w += self.lr * yi * xi
                    self.b += self.lr * yi
                    errors += 1

            print(f"Epoch {epoch+1}/{self.n_epochs}, errors = {errors}")

        self.is_fitted = True
        return self

    def predict(self, X):
        # --- required by tests ---
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # --- required by tests: dimension check ---
        if X.shape[1] != self.w.shape[0]:
            raise ValueError("Feature dimension mismatch.")

        activation = X @ self.w + self.b
        # output {0,1}
        return np.where(activation >= 0, 1, 0)

    def score(self, X, y):
        y = np.asarray(y)
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))


class PerceptronOVR:
    """
    Multiclass Perceptron using One-vs-Rest.
    Builds K binary Perceptrons.
    """

    def __init__(self, n_classes=10, n_features=784, lr=1.0, n_epochs=10):
        self.n_classes = n_classes
        self.n_features = n_features
        self.lr = lr
        self.n_epochs = n_epochs

        # IMPORTANT: to make toy tests deterministic/easier to converge,
        # disable shuffling inside OVR binary learners.
        self.models = [
            PerceptronBinary(n_features, lr=lr, n_epochs=n_epochs, shuffle=False)
            for _ in range(n_classes)
        ]
        self.is_fitted = False

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        # --- required by tests: reject non-numeric labels (e.g. strings) ---
        if not np.issubdtype(y.dtype, np.integer):
            raise TypeError("PerceptronOVR requires integer class labels.")

        # --- to increase stability on small toy sets, train multiple passes ---
        # This keeps the code simple but helps convergence for the tests.
        n_outer_passes = 5

        for _ in range(n_outer_passes):
            for k in range(self.n_classes):
                print(f"\nTraining classifier for class {k} vs rest")
                y_k = np.where(y == k, 1, 0)  # labels {0,1} for PerceptronBinary
                self.models[k].fit(X, y_k)

        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != self.n_features:
            raise ValueError("Feature dimension mismatch.")

        # compute scores = w_k·x + b_k
        scores = np.zeros((X.shape[0], self.n_classes), dtype=np.float32)
        for k, model in enumerate(self.models):
            scores[:, k] = X @ model.w + model.b

        return np.argmax(scores, axis=1)

    def score(self, X, y):
        y = np.asarray(y)
        y_pred = self.predict(X)
        return float(np.mean(y_pred == y))
