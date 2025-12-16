# K-Nearest Neighbors

This directory contains an implementation of the K-Nearest Neighbors (KNN) algorithm, a simple, instance-based supervised learning method used for classification. We evaluated KNN's classification capabilities in class, so I am interested in exploring whether using KNN regression is ever predictive. As a result, in this project, the KNN algorithm is applied to a dataset of socioeconomic and public transit indicators to classify what population of a country has completed secondary school based on metrics like GDP, internet access, population, etc.

---

## Overview of KNN

K-Nearest Neighbors is a **distance-based** algorithm that:

- Stores all training data points  
- Computes similarity between a new data point and all training points  
- Uses the **k closest neighbors** (based on distance) to make a prediction  

In this project:

- Similarity is measured using **Euclidean distance**
- KNN is used for **classification** and **regression**

A visual illustration of how KNN works for classification and regression:

<img width="739" alt="KNN illustration" src="https://github.com/user-attachments/assets/38f8c047-e23f-4841-afcd-133156418bfa" />

---

## Algorithm

1. **Choose** the number of neighbors, `k`.
2. For a new data point:
   - Compute the distance between this point and all points in the training set.
   - Sort the distances and select the `k` closest neighbors.
3. **Prediction:**
   - **Classification:** take a majority vote over the labels of the `k` neighbors.
   - **Regression:** compute the average of the neighbors’ target values.
4. Repeat for all points in the test set.

---

## Mathematical Description

Let: 

- $x \in \mathbb{R}^n$: new input vector  
- $D = \{(x_1, y_1), (x_2, y_2), ..., (x_m, y_m)\}$: training dataset  

**Distance metric (Euclidean):**

$d(x, x_i) = \sqrt{ \sum_{j=1}^{n} (x_j - x_{ij})^2 }$
- Select the k smallest distances and assign the class by majority vote or mean of k neighbors

Steps:

1. Compute $d(x, x_i)$ for all training points $x_i$.
2. Select the `k` points with the smallest distances.
3. For:
   - **Classification:** assign the class by majority vote among the `k` neighbors.  
   - **Regression:** predict the value as the mean of the `k` neighbors’ target values.

---

## Evaluation

For the KNN **regression** task (predicting secondary school completion rates), model performance is evaluated using:

- **Error metrics**
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)
- **Goodness of fit**
  - $ R^2 $ (coefficient of determination)
- **Visual diagnostics**
  - Plots of **actual vs. predicted** values to assess how closely predictions match observed data

These evaluations help determine whether KNN regression provides useful predictive performance for this socioeconomic dataset.

## Files Included

- `KNN.ipynb`: KNN model implementation
- `README.md`: This documentation file