# Logistic Regression

This directory contains example code and documentation for the Logistic Regression algorithm used in supervised learning for binary classification tasks.

## Algorithm

Logistic Regression is a linear classification model that predicts the probability that an input belongs to class 1. It applies a sigmoid activation on a linear combination of input features:

z = β0 + β1 * x1 + β2 * x2 + ... + βd * xd
probability = 1 / (1 + exp(-z))

The output is a value between 0 and 1, representing the predicted probability.

### Objective Function (Binary Cross-Entropy Loss)

The model is trained by minimizing the logistic loss:

Loss = -1/N * Σ [ y * log(p) + (1 - y) * log(1 - p) ]

Where:

y = true label (0 or 1)

p = predicted probability

N = number of samples

Gradient descent is used to update parameters:

β = β - learning_rate * gradient

Key Hyperparameters

learning_rate — step size for gradient descent

max_iter — maximum number of training iterations

tol — stopping threshold

fit_intercept — whether to add an intercept term

Outputs

predict_proba(X) — returns predicted probabilities

predict(X) — returns class labels (0 or 1) based on a threshold

## Data

This example uses the Breast Cancer Wisconsin Dataset from sklearn.datasets.

### Features

The dataset includes 30 numerical features describing cell nucleus characteristics extracted from medical images, such as:

radius

texture

smoothness

compactness

symmetry

### Labels

0 → malignant

1 → benign

Preprocessing Steps

### Load dataset

Convert into a pandas DataFrame for exploration

Split into training and test sets

Standardize input features using StandardScaler

Train the Logistic Regression model using gradient descent
