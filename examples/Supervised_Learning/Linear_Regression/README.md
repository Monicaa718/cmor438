# Linear Regression

This directory contains an end-to-end example demonstrating how to use the
`LinearRegression` model implemented in the `rice_ml` library.  
The example walks through loading real-world data, performing exploratory data
analysis (EDA), training a regression model, and evaluating its performance.

## Algorithm

Linear Regression is a fundamental supervised learning algorithm used to model the
relationship between a set of input features \(X\) and a continuous output variable \(y\).
The model assumes a linear relationship of the form:

\[
y = \beta_0 + \beta_1 x_1 + \cdots + \beta_d x_d + \epsilon
\]

where:

- \( \beta_0 \) is the intercept  
- \( \beta_i \) are the learned coefficients (slopes)  
- \( \epsilon \) represents noise or unexplained variance  

The goal of linear regression is to estimate the coefficients \( \beta \) that best
fit the data.

---

### Objective Function

The model is trained by minimizing the **Ordinary Least Squares (OLS)** loss:

\[
\min_{\beta} \| y - X\beta \|_2^2
\]

This optimization problem has a closed-form solution:

\[
\hat{\beta} = (X^\top X)^{-1} X^\top y
\]

In practice, implementations may also:

- add an intercept term  
- standardize features  
- use numerical solvers if \(X^\top X\) is not invertible  

---

### Key Components in Our Implementation

The `LinearRegression` class in this project includes:

- `fit_intercept=True/False`  
  - Determines whether the model learns an intercept term  
- `.fit(X, y)`  
  - Computes coefficients using the OLS closed-form solution  
- `.predict(X)`  
  - Generates predictions for new input data  
- `.coef_` and `.intercept_`  
  - Learned model parameters  
- `.R2()`  
  - Computes the coefficient of determination  
- `.RMSE()`  
  - Computes the root mean squared error  

---

### Notes on Intercept Handling

If `fit_intercept=True`, a column of ones is added to \(X\) before solving OLS:

\[
X' = [\mathbf{1}, X]
\]

This allows the model to learn \( \beta_0 \).

---

### When to Use Linear Regression

Linear Regression works best when:

- The relationship between features and target is approximately linear  
- Features are not highly correlated (low multicollinearity)  
- Noise is approximately Gaussian  
- You want a simple, interpretable model  

It is not suitable when the relationships are strongly nonlinear
unless features are transformed (e.g., polynomial regression).


## Data

This example uses the **Diabetes dataset** from `sklearn.datasets`, a widely used
benchmark for evaluating regression models. The dataset consists of standardized
numeric features collected from real patients and a continuous target variable
representing disease progression.

---

### Dataset Loading

The dataset is loaded via:

from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

### Input Features (X)

The dataset contains 10 baseline medical measurements, each standardized
(mean 0, variance 1):

Feature	Description
age	Age of patient
sex	Sex (binary-coded)
bmi	Body mass index
bp	Average blood pressure
s1	Blood serum measurement 1
s2	Blood serum measurement 2
s3	Blood serum measurement 3
s4	Blood serum measurement 4
s5	Blood serum measurement 5
s6	Blood serum measurement 6

These features are used as predictors in the linear regression model.

### Target Variable (y)

The target is a quantitative measure of disease progression recorded one year
after baseline.

It is stored in:

y = diabetes.target


The goal of Linear Regression is to predict this continuous outcome from the feature matrix 
𝑋
X.

### Preprocessing

In this example:

No additional preprocessing (e.g., scaling) is required because the dataset is already standardized.

The data is split into training (80%) and testing (20%) sets using:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

This allows reliable evaluation of model generalization performance.