# Multilayer Perceptron

This directory contains example code and notes for the **Multilayer Perceptron (MLP)** algorithm in supervised learning.  
The focus is on understanding the algorithmic structure, optimization dynamics, and practical design choices that improve performance on image classification tasks such as MNIST.

---

## Algorithm

### Core Idea

A **Multilayer Perceptron (MLP)** is a feedforward neural network composed of:

- An **input layer**
- One or more **hidden layers** with nonlinear activation functions
- An **output layer** that produces class scores or predictions

Unlike a single-layer Perceptron, an MLP introduces **nonlinear transformations** through hidden layers, enabling it to model complex, non-linearly separable decision boundaries.

Formally, for a one-hidden-layer MLP:

\[
\mathbf{h} = \phi(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)
\]

\[
\mathbf{z} = \mathbf{W}_2 \mathbf{h} + \mathbf{b}_2
\]

where:

- \(\mathbf{x} \in \mathbb{R}^d\) is the input feature vector
- \(\mathbf{W}_1, \mathbf{W}_2\) are weight matrices
- \(\mathbf{b}_1, \mathbf{b}_2\) are bias terms
- \(\phi(\cdot)\) is a nonlinear activation function (e.g., `tanh`)
- \(\mathbf{z}\) contains the output class scores

The predicted class is typically obtained via:

\[
\hat{y} = \arg\max_k z_k
\]

---

### Objective Function

For multiclass classification, the MLP is trained to minimize a loss function such as **cross-entropy loss**:

\[
\mathcal{L} = - \sum_{i=1}^{N} \log p(y_i \mid \mathbf{x}_i)
\]

where \(p(y_i \mid \mathbf{x}_i)\) is the predicted probability for the true class.

Training is performed using **gradient-based optimization** with backpropagation.

---

### Key Design Choices and Hyperparameters

#### Activation Function: ReLU vs. tanh

**ReLU (Rectified Linear Unit):**
\[
\text{ReLU}(x) = \max(0, x)
\]

- Advantages: simple, efficient, sparse activations
- Limitations:
  - Outputs are strictly non-negative
  - Zero gradient for \(x < 0\) (dead neuron problem)

**tanh (Hyperbolic Tangent):**
\[
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \in (-1, 1)
\]

- Zero-centered output
- Smooth, continuous gradients
- Symmetric around zero

**Why `tanh` helps in this implementation:**

- MNIST inputs are normalized and largely centered near zero
- Zero-centered activations reduce bias in gradient updates
- Smoother gradients improve stability during backpropagation

Empirically, replacing ReLU with `tanh` leads to more stable optimization and improved feature learning in early layers.

---

#### Optimization with Momentum

Standard stochastic gradient descent (SGD) updates parameters as:

\[
\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla \mathcal{L}_t
\]

This can lead to oscillations and slow convergence, especially in high-dimensional parameter spaces.

**Momentum-based SGD** introduces a velocity term:

\[
\mathbf{v}_{t+1} = \mu \mathbf{v}_t - \eta \nabla \mathcal{L}_t
\]

\[
\mathbf{w}_{t+1} = \mathbf{w}_t + \mathbf{v}_{t+1}
\]

where:

- \(\mu \in [0,1)\) is the momentum coefficient
- \(\mathbf{v}_t\) accumulates past gradients

**Effect of momentum:**

- Dampens oscillations in high-curvature directions
- Accelerates movement along consistent descent directions
- Improves convergence speed and stability

In practice, momentum allows the MLP to reach a better local optimum compared to plain SGD.

---

## Data

### Input and Labels

The MLP is applied to the **MNIST handwritten digit dataset**, where:

- **Input features:**  
  Flattened grayscale images of shape \(28 \times 28\), resulting in a 784-dimensional vector
- **Labels:**  
  Integer class labels in \(\{0, 1, \dots, 9\}\)

---

### Preprocessing

Typical preprocessing steps include:

- Flattening images into 1D feature vectors
- Normalizing pixel values to the range \([0, 1]\)
- Keeping labels as integers for multiclass classification

These steps ensure numerical stability and compatibility with gradient-based optimization.

---

## Summary

The Multilayer Perceptron extends linear models by introducing nonlinear hidden layers and gradient-based optimization.  
In this implementation, performance gains are primarily driven by:

- The use of **`tanh` activation**, which improves gradient symmetry and stability
- **Momentum-based optimization**, which accelerates convergence and reduces oscillations

Together, these design choices allow the MLP to learn meaningful representations from raw pixel data and achieve strong performance on standard image classification benchmarks such as MNIST.
