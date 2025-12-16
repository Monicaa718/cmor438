# Ensemble Learning for Bitcoin Time-Series Prediction  
(Bagging, Boosting, and Stacking from Scratch)

Ensemble learning is a machine learning paradigm in which multiple models are combined to produce a single predictive system.  
The key motivation is that different models exhibit different inductive biases and error structures, and combining them can improve robustness and generalization.

This project studies ensemble learning in the context of Bitcoin daily return prediction, with a particular emphasis on distinguishing between **simple model averaging** and **true stacking (meta-learning)**.

Rather than treating ensemble methods as black-box tools, the project focuses on conceptual clarity by implementing core algorithms from scratch and analyzing their behavior in a noisy financial time-series environment.

---

## Project Overview

The objective of this project is to explore how different ensemble learning strategies perform when applied to short-horizon Bitcoin return prediction.

Specifically, we compare:

- A buy-and-hold benchmark
- Bagging via a hand-written Random Forest
- Boosting via a hand-written XGBoost-style gradient boosting model
- Fixed-weight model averaging
- A true stacking ensemble with a learned meta-model

The project is designed to highlight the relationship between machine learning objectives and trading outcomes, rather than to optimize raw profitability.

---

## Task Description

The learning task is formulated as a supervised regression problem.

Given information available up to day \( t \), the goal is to predict the next-day Bitcoin return:

$$
\hat{r}_{t+1} = f(X_t),
$$

where \( X_t \) is a feature vector constructed from historical price information.

Predictions are translated into trading decisions using a simple sign-based rule:

$$
w_t =
\begin{cases}
1, & \hat{r}_{t+1} > 0, \\
0, & \hat{r}_{t+1} \le 0.
\end{cases}
$$

This allows model performance to be evaluated both statistically and economically.

---

## Dataset

The dataset consists of daily Bitcoin price data obtained from Yahoo Finance.

Each observation corresponds to one trading day and includes:

- Open price
- High price
- Low price
- Close price
- Trading volume

The data spans multiple market regimes, including prolonged bull markets and periods of elevated volatility.

---

## Feature Engineering

A small set of technical features is constructed from raw prices.

Daily return:

$$
r_t = \frac{P_t - P_{t-1}}{P_{t-1}}
$$

Moving averages:

$$
\text{SMA}_k(t) = \frac{1}{k} \sum_{i=0}^{k-1} P_{t-i}
$$

Trend indicator:

$$
\text{SMA\_diff}_t = \text{SMA}_7(t) - \text{SMA}_{30}(t)
$$

Rolling volatility:

$$
\sigma_t = \sqrt{\frac{1}{k-1} \sum_{i=0}^{k-1} (r_{t-i} - \bar{r}_t)^2}
$$

The prediction target is defined as the next-day return \( r_{t+1} \).

---

## Models Used

### Random Forest (Bagging)

The Random Forest model is implemented from scratch following the bagging principle.

Each decision tree is trained on a bootstrap sample:

$$
(X^{(b)}, y^{(b)}) \sim \text{Bootstrap}(X, y)
$$

Predictions are aggregated by averaging:

$$
\hat{r}_{t+1}^{\text{RF}} = \frac{1}{M} \sum_{m=1}^{M} f_m(X_t)
$$

Bagging primarily reduces variance by averaging many high-variance base learners.

---

### XGBoost-Style Gradient Boosting (Boosting)

The boosting model is implemented as an additive ensemble:

$$
\hat{r}_{t+1}^{(m)} = \hat{r}_{t+1}^{(m-1)} + \eta f_m(X_t)
$$

Each new tree is trained on the negative gradient of the squared loss:

$$
L(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2
$$

Boosting focuses on bias reduction by sequentially correcting errors made by previous models.

---

## Ensemble Methods

### Fixed-Weight Model Averaging

As a baseline ensemble, predictions from Random Forest and XGBoost are combined using fixed weights:

$$
\hat{r}_{t+1}^{\text{AVG}} =
\frac{1}{2}\hat{r}_{t+1}^{\text{RF}} +
\frac{1}{2}\hat{r}_{t+1}^{\text{XGB}}
$$

This approach assumes that both base learners are equally informative and does not learn combination weights from data.

---

### Stacking: A True Ensemble Method

To demonstrate a deeper understanding of ensemble learning, the project implements **stacking**, in which a second-level model learns how to combine base learner predictions.

First, base models generate predictions:

$$
\hat{r}_{t+1}^{(1)} = f_{\text{RF}}(X_t), \quad
\hat{r}_{t+1}^{(2)} = f_{\text{XGB}}(X_t)
$$

These predictions are then used as inputs to a meta-learner:

$$
\hat{r}_{t+1} = g\bigl(\hat{r}_{t+1}^{(1)}, \hat{r}_{t+1}^{(2)}\bigr)
$$

In this project, the meta-learner is a regularized linear regression (Ridge), trained to solve:

$$
\min_{w_0, w_1, w_2}
\sum_t \left(
r_{t+1} - w_0 - w_1 \hat{r}_{t+1}^{(1)} - w_2 \hat{r}_{t+1}^{(2)}
\right)^2
+ \lambda (w_1^2 + w_2^2)
$$

Unlike fixed-weight averaging, stacking learns data-driven combination weights and adapts to the relative reliability of each base model.

---

## Evaluation Methodology

Models are evaluated using both predictive and trading-oriented metrics:

- Mean squared error (MSE)
- Coefficient of determination (\( R^2 \))
- Directional accuracy
- Cumulative returns from a simple trading strategy
- Comparison against a buy-and-hold benchmark

This dual evaluation highlights the distinction between predictive accuracy and economic performance.

---

## Key Insights

The results demonstrate several important points:

- Buy-and-hold dominates due to strong long-run price drift, highlighting the limits of short-horizon prediction.
- Random Forest produces stable but conservative signals due to variance reduction.
- Boosting is more sensitive to short-term noise and can overreact in financial time series.
- Fixed-weight averaging does not guarantee improvement and can reduce effective market exposure.
- Stacking achieves the strongest performance among all machine-learning-based strategies by learning an optimal linear correction of heterogeneous base models.

---

## Conclusion

This project demonstrates how bagging, boosting, and stacking can be implemented from scratch and compared in a financial time-series context.

While stacking cannot outperform buy-and-hold in a strongly trending market—since any timing strategy necessarily reduces exposure to persistent drift—it represents the best achievable performance within the class of predictive machine learning models considered here.

More broadly, the results emphasize a fundamental lesson in financial machine learning: ensemble methods improve robustness and interpretability, but they cannot overcome the intrinsic limits imposed by noise and non-stationarity in asset returns.

---

## Libraries Used

- Pandas: https://pandas.pydata.org/
- NumPy: https://numpy.org/
- Matplotlib: https://matplotlib.org/
- Seaborn: https://seaborn.pydata.org/
- Scikit-learn (Decision Trees and Linear Models only): https://scikit-learn.org/
