# Regression Tree

A tree model has many real-world analogies and has influenced a wide range of machine learning methods. In the CART framework, a tree can be used for both classification and regression. This project focuses on a **regression tree**, where the model predicts a **numeric value** rather than a class label. The tree repeatedly splits the data into two groups and chooses each split to **minimize the within-node squared error (SSE)**. At any leaf, the prediction is a single constant: the **mean** target value of samples that fall into that leaf.

Because the Car Evaluation dataset provides a categorical class label (`unacc`, `acc`, `good`, `vgood`), we convert this label into an **ordered numeric target** and fit a regression tree on that numeric score. This keeps the modelling objective fully within the regression setting and allows performance to be assessed with regression metrics such as MSE, RMSE, MAE, and R².

## Task

In this notebook, we implement a **CART regression tree from scratch** and apply it to a fully categorical dataset. The main goal is to learn an interpretable set of if–then rules that explain how combinations of car attributes shift the expected ordinal score. Model quality is evaluated using standard regression error measures and visual diagnostics, and the learned tree is visualized as a tree diagram.

## Dataset

The data used for this model is the Car Evaluation dataset provided in the project files. The dataset contains categorical features describing car attributes, including buying price, maintenance cost, number of doors, passenger capacity, luggage boot size, and safety level. The original label is a four-level class variable: `unacc`, `acc`, `good`, and `vgood`. For the regression tree, the label is encoded into an ordered numeric target, typically `unacc=0`, `acc=1`, `good=2`, `vgood=3`, so that the model predicts a continuous score on this ordinal scale.

## Libraires

Pandas https://pandas.pydata.org/  
Matplotlib https://matplotlib.org/  
Numpy https://numpy.org/  
Scikit-learn https://scikit-learn.org/
