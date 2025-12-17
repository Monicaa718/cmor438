# randomforest.py
# ----------------------------------
# An implementation of Random Forest (Regression)
# ----------------------------------

import numpy as np
from sklearn.tree import DecisionTreeRegressor 


class RandomForestRegressorScratch:
    """
    A minimal Random Forest Regressor (from scratch style).

    Key ideas:
    - Bootstrap sampling
    - Many weak decision trees
    - Prediction = average of tree predictions
    """

    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_leaf=1,
        max_features=None,
        random_state=None
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.trees = []
        self.rng = np.random.default_rng(random_state)

    def _bootstrap_sample(self, X, y):
        """
        Draw a bootstrap sample (sampling with replacement).
        """
        n_samples = X.shape[0]
        indices = self.rng.integers(0, n_samples, size=n_samples)
        return X[indices], y[indices]

    def fit(self, X, y):
        """
        Train all decision trees on bootstrap samples.
        """
        self.trees = []

        X = np.asarray(X)
        y = np.asarray(y)

        for i in range(self.n_estimators):
            # 1. Bootstrap
            X_boot, y_boot = self._bootstrap_sample(X, y)

            # 2. Train a weak decision tree
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=None if self.random_state is None else self.random_state + i
            )

            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

        return self

    def predict(self, X):
        """
        Predict by averaging predictions from all trees.
        """
        X = np.asarray(X)

        # Collect predictions from each tree
        all_preds = np.array([tree.predict(X) for tree in self.trees])

        # Average across trees
        return np.mean(all_preds, axis=0)




# xgboost.py

import numpy as np
from sklearn.tree import DecisionTreeRegressor

class XGBoostRegressorScratch:
    """
    A minimal XGBoost-style regressor (gradient boosting trees).

    Model:
        y_hat = sum_{m=1}^M eta * f_m(x)

    where each f_m is a regression tree
    trained on the negative gradient (residual).
    """

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_leaf=1,
        random_state=None
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.trees = []
        self.init_value = 0.0

    # -----------------------------
    # Loss and gradient (MSE loss)
    # -----------------------------
    def _gradient(self, y_true, y_pred):
        """
        Gradient of squared loss:
            L = 1/2 (y - y_pred)^2
        dL/dy_pred = y_pred - y
        """
        return y_pred - y_true

    # -----------------------------
    # Fit model
    # -----------------------------
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        n_samples = X.shape[0]

        # Initial prediction = mean of y
        self.init_value = np.mean(y)
        y_pred = np.full(n_samples, self.init_value)

        self.trees = []

        for m in range(self.n_estimators):
            # 1. Compute negative gradient (residual)
            grad = -self._gradient(y, y_pred)

            # 2. Fit a weak learner (tree) to gradient
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=None if self.random_state is None else self.random_state + m
            )

            tree.fit(X, grad)

            # 3. Update predictions
            update = tree.predict(X)
            y_pred += self.learning_rate * update

            self.trees.append(tree)

        return self

    # -----------------------------
    # Predict
    # -----------------------------
    def predict(self, X):
        X = np.asarray(X)

        # Start from initial prediction
        y_pred = np.full(X.shape[0], self.init_value)

        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)

        return y_pred
