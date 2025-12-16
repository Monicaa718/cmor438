---
output:
  html_document: default
  word_document: default
---

# Perceptron

This directory contains example code and notes for the **Perceptron algorithm** in supervised learning.

The implementation includes both **binary Perceptron** and a **multiclass extension using One-vs-Rest (OvR)**, written from scratch in NumPy for educational purposes.

---

## Algorithm 

The Perceptron is a **linear classification algorithm** designed for supervised learning tasks.  
Its goal is to learn a weight vector that separates data points of different classes using a linear decision boundary.

---

## Core Idea

Given an input feature vector  
$x \in \mathbb{R}^d$,  
the Perceptron predicts a label based on the sign of a linear score:

$$
\hat{y} = \mathrm{sign}(w^\top x + b)
$$

where:

- $w \in \mathbb{R}^d$ is the weight vector  
- $b \in \mathbb{R}$ is the bias term  

---

## Learning Rule (Binary Case)

For binary classification with labels  
$y \in \{-1, +1\}$,

a data point is misclassified if:

$$
y (w^\top x + b) \le 0
$$

In this case, the parameters are updated as:

$$
w \leftarrow w + \eta y x
$$

$$
b \leftarrow b + \eta y
$$

where $\eta$ is the learning rate.

---

## Training Algorithm (Binary Case)

**Input**: Training data {(x⁽ⁱ⁾, y⁽ⁱ⁾)} where y⁽ⁱ⁾ ∈ {-1, +1}

**Parameters**: Learning rate η, max iterations T

1. Initialize w ← 0 (or small random values), b ← 0
2. For epoch = 1 to T:
3. For each training example (x⁽ⁱ⁾, y⁽ⁱ⁾):
4. Compute prediction: ŷ = sign(wᵀx⁽ⁱ⁾ + b)
5. If y⁽ⁱ⁾(wᵀx⁽ⁱ⁾ + b) ≤ 0 (misclassified):

- Update weights: w ← w + η·y⁽ⁱ⁾·x⁽ⁱ⁾

- Update bias: b ← b + η·y⁽ⁱ⁾

6. Stop when all samples are correctly classified or max epochs reached

---

## Multiclass Extension (One-vs-Rest)

For multiclass problems:

- One binary Perceptron is trained for each class
- Each classifier distinguishes **one class vs. all others**
- At prediction time, the class with the highest score is selected:

$$
\hat{y} = \arg\max_k (w_k^\top x + b_k)
$$

---

## Data

- Input features are real-valued vectors
- For MNIST:
  - Each input has dimension $784$ (flattened $28 \times 28$ image)
  - Pixel values are normalized to $[0,1]$
- Labels are integers $\{0,1,\dots,9\}$ for multiclass classification

---

## Notes and Limitations

- The Perceptron can only learn **linearly separable** decision boundaries
- It does not output probabilities
- Performance is limited compared to multi-layer neural networks
