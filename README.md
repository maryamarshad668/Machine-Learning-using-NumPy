# Machine Learning using NumPy

## Overview

This project implements two fundamental machine learning algorithms **using NumPy**:

* **Linear Regression** for predicting continuous values.
* **Logistic Regression** for binary classification.

The goal is to understand the mathematical foundations of machine learning by implementing optimization algorithms, loss functions, and gradient descent without relying on high-level machine learning libraries.

The project also compares the custom implementations with **scikit-learn** models to evaluate correctness and performance.

---

# Objectives

* Implement Linear Regression using Gradient Descent.
* Implement Logistic Regression using Binary Cross-Entropy Loss.
* Understand parameter optimization through Gradient Descent.
* Evaluate model performance using multiple metrics.
* Compare custom implementations with scikit-learn.
* Analyze the effects of learning rates and classification thresholds.

---

## Linear Regression

**Dataset:** Boston Housing Dataset

* 506 housing samples
* 13 numerical features
* Predicts median house price

---

## Logistic Regression

**Dataset:** Wisconsin Diagnostic Breast Cancer Dataset

* Binary Classification
* Benign vs Malignant tumors
* Multiple numerical diagnostic features

---

# Project Features

## Linear Regression 

Implemented completely from scratch without using machine learning libraries.

### Features

* Data preprocessing
* Train/Test split
* Parameter initialization
* Gradient Descent optimization
* Mean Squared Error (MSE) loss
* Automatic convergence checking
* Learning rate experimentation
* Prediction on unseen data
* Training loss visualization

---

## Gradient Descent

The model minimizes the squared error objective function:

[
Loss = \frac{1}{n}\sum(target - prediction)^2
]

During each iteration:

* Compute predictions
* Calculate loss
* Compute gradients
* Update weights and bias
* Check convergence

The optimization stops when:

[
||w_{new}-w_{old}|| \leq \epsilon
]

---

## Learning Rate Experiments

The implementation evaluates multiple learning rates to study convergence behavior.

Experiments include:

* Very small learning rates
* Moderate learning rates
* Large learning rates

Observations include:

* Speed of convergence
* Stability
* Oscillation
* Divergence

---

## Model Evaluation

The Linear Regression model is evaluated using:

* Mean Squared Error (MSE)
* Prediction comparison
* Failure case analysis
* Learned parameters
* Training loss curve

Results are compared against **scikit-learn's LinearRegression** implementation.

---

# Logistic Regression 

Implemented completely from scratch using Binary Cross-Entropy Loss.

---

## Features

* Sigmoid activation
* Binary Cross-Entropy Loss
* Gradient computation
* Gradient Descent optimization
* Convergence checking
* Probability prediction
* Binary classification

---

## Binary Cross-Entropy Loss

The optimization objective is:

[
L=-\frac1n\sum[y\log(\hat y)+(1-y)\log(1-\hat y)]
]

---

## Performance Evaluation

The classifier is evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score

---

# K-Fold Cross Validation

To evaluate model robustness, the implementation performs **5-Fold Cross Validation**.

Process:

1. Split dataset into five folds.
2. Train on four folds.
3. Test on the remaining fold.
4. Repeat for all folds.
5. Report average performance metrics.

---

# Threshold Tuning

Instead of using only the default threshold (0.5), the model evaluates multiple thresholds:

* 0.30
* 0.40
* 0.50
* 0.60
* 0.70

For each threshold, the following metrics are computed:

* Accuracy
* Precision
* Recall
* F1-Score

A comparison table and performance curves illustrate how threshold selection affects classification performance.

---

# Visualizations

The project includes multiple plots, including:

* Linear Regression training loss vs iterations
* Logistic Regression training loss vs iterations
* Threshold vs Accuracy
* Threshold vs Precision
* Threshold vs Recall
* Threshold vs F1-Score

---

# 🛠 Technologies Used

* Python
* NumPy
* Matplotlib
* scikit-learn 

---

# Learning Outcomes

This project demonstrates a practical understanding of:

* Linear Regression
* Logistic Regression
* Gradient Descent Optimization
* Binary Cross-Entropy
* Mean Squared Error
* Numerical Optimization
* Machine Learning Evaluation Metrics
* Cross Validation
* Threshold Tuning
* Data Preprocessing
* Model Comparison

---

# Author

**Maryam Arshad**
