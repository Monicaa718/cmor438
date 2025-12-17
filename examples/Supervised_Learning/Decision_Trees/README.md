# Decision Tree

A tree model has many real-world analogies and has influenced a wide range of machine learning methods. In the CART framework, a tree can be used for both classification and regression. This project focuses on a decision tree classifier, where the model predicts a categorical class label rather than a numeric value. The tree recursively partitions the feature space using a sequence of if–then rules, choosing each split to minimize node impurity, measured here by the Gini impurity. At each leaf, the model outputs a class probability distribution, and predictions are made by selecting the most probable class.

The decision tree classifier is particularly well-suited for datasets with categorical or discretized features, as it does not require feature scaling or linearity assumptions. Its hierarchical structure also makes the learned decision rules easy to interpret.

## Task

In this notebook, we apply a CART decision tree classifier implemented from scratch to a fully categorical dataset. The objective is to learn an interpretable set of if–then decision rules that map combinations of input attributes to discrete class outcomes. Model performance is evaluated using standard classification metrics, including accuracy, confusion matrices, and predicted class probabilities. In addition, the learned structure of the tree is visualized to aid interpretability.

## Dataset

The data used for this model is the Car Evaluation dataset provided in the project files. The dataset contains categorical features describing car attributes, including buying price, maintenance cost, number of doors, passenger capacity, luggage boot size, and safety level. The target variable is a four-level class label: `unacc`, `acc`, `good`, and `vgood`.

Since the custom decision tree implementation operates on numeric feature values, all categorical predictors are encoded into ordered integer values before training. The class labels are also integer-encoded (`unacc=0`, `acc=1`, `good=2`, `vgood=3`) to satisfy the input requirements of the classifier, while preserving their categorical interpretation.

## Libraries

- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [NumPy](https://numpy.org/)
- [Scikit-learn](https://scikit-learn.org/)