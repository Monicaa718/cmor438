# mlp.py
# --------------------------------
# Simple MLP implemented in numpy
# --------------------------------

import numpy as np


# -------- Activation Functions -------- #

def softmax(z):
    # Numerical stability: subtract max
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# def relu(z):
#     return np.maximum(0, z)

# def relu_grad(z):
#     return (z > 0).astype(float)
def tanh(z):
    return np.tanh(z)

def tanh_grad(z):
    return 1 - np.tanh(z)**2


# -------- One-hot Encoding -------- #

def one_hot(y, num_classes=10):
    out = np.zeros((len(y), num_classes))
    out[np.arange(len(y)), y] = 1
    return out


# -------- MLP Class -------- #

class SimpleMLP:
    """
    A minimal 1-hidden-layer MLP for MNIST.
    Architecture:
        Input (784)
            -> Dense Layer (hidden_dim)
            -> ReLU
            -> Dense Layer (10)
            -> Softmax
    """

    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10, lr=0.01, momentum=0.9):
        self.lr = lr

        # Xavier Initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = np.random.randn(hidden_dim, output_dim) / np.sqrt(hidden_dim)
        self.b2 = np.zeros((1, output_dim))

        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)

        self.momentum = momentum


    # -------- Forward Pass -------- #

    def forward(self, X):
        # Layer 1
        self.z1 = X @ self.W1 + self.b1
        #self.a1 = relu(self.z1)
        self.a1 = tanh(self.z1)

        # Output layer
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = softmax(self.z2)

        return self.a2


    # -------- Backward Pass -------- #

    def backward(self, X, y_onehot):
        m = X.shape[0]

        # Output layer gradient
        dz2 = self.a2 - y_onehot                 # (batch, 10)
        dW2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Hidden layer gradient
        dz1 = (dz2 @ self.W2.T) * tanh_grad(self.z1)
        dW1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Gradient descent update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        self.vW1 = self.momentum * self.vW1 - self.lr * dW1
        self.vb1 = self.momentum * self.vb1 - self.lr * db1
        self.vW2 = self.momentum * self.vW2 - self.lr * dW2
        self.vb2 = self.momentum * self.vb2 - self.lr * db2

        self.W1 += self.vW1
        self.b1 += self.vb1
        self.W2 += self.vW2
        self.b2 += self.vb2


    # -------- Training Loop -------- #

    def train(self, X, y, epochs=5, batch_size=64):
        y_onehot = one_hot(y, num_classes=10)

        for epoch in range(epochs):
            # Shuffle dataset
            idx = np.random.permutation(len(X))
            X, y_onehot = X[idx], y_onehot[idx]

            # Mini-batch training
            for i in range(0, len(X), batch_size):
                Xb = X[i:i+batch_size]
                yb = y_onehot[i:i+batch_size]

                self.forward(Xb)
                self.backward(Xb, yb)

            # Compute train accuracy at end of epoch
            preds = self.predict(X)
            acc = np.mean(preds == np.argmax(y_onehot, axis=1))
            print(f"Epoch {epoch+1}/{epochs} - Train accuracy: {acc:.4f}")


    # -------- Prediction -------- #

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
