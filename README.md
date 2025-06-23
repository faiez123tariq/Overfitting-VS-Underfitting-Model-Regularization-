# 📊 Polynomial Regression: Underfitting, Overfitting, and Regularization (L2)

This project demonstrates **underfitting**, **overfitting**, and **regularization (L2 - Ridge Regression)** using a synthetic dataset generated from a cosine function with added noise. The goal is to understand model complexity, generalization, and the impact of regularization on performance.

---

## 🚀 Project Overview

We build and evaluate three models:

1. **Underfitting Model**: Simple Linear Regression (Degree 1 Polynomial)
2. **Overfitting Model**: High-degree Polynomial Regression (Degree 15)
3. **Regularized Model**: Ridge Regression (L2 Regularization) with Degree 15 Polynomial

---

## 📂 Dataset

We generate **synthetic data** using the following equation:

```text
y = cos(1.5 * π * x) + noise
```

* **Noise:** Random Gaussian noise is added to simulate real-world data.
* **Data Points:** 30 random data points between 0 and 1.
* **Train/Test Split:** 70% Training, 30% Validation.

---

## 🛠️ Project Structure

```text
.
├── polynomial_regression_regularization.py  # Main Python script
└── README.md                                # Project documentation
```

---

## 📦 Requirements

* Python 3.x
* NumPy
* Matplotlib
* Scikit-learn

You can install the required libraries using:

```bash
pip install numpy matplotlib scikit-learn
```

---

## 🔥 How to Run

1. Clone the repository:

```bash
git clone <repository-url>
```

2. Navigate to the project directory:

```bash
cd <project-directory>
```

3. Run the Python file:

```bash
python Model.py
```

---

## 📈 Evaluation Metrics

We use **Mean Squared Error (MSE)** to evaluate model performance on both **Training** and **Validation** datasets.

### Model Performance Summary:

* **Underfitting Model:** High training and validation error (model is too simple).
* **Overfitting Model:** Very low training error, but high validation error (model is too complex and memorizes noise).
* **Regularized Model:** Balanced training and validation error (good generalization due to L2 penalty).

---

## 📊 Visualizations

The project visualizes:

1. **Underfitting Model** → A straight-line fit showing poor approximation.
2. **Overfitting Model** → Highly oscillating curve fitting every training point.
3. **Regularized Model** → Smooth curve that balances bias and variance.

---

## ✅ Key Concepts Covered

* **Underfitting:** Model is too simple, cannot capture the pattern.
* **Overfitting:** Model is too complex, memorizes noise instead of learning.
* **Regularization (L2):** Adds penalty on large coefficients to reduce overfitting.
* **Polynomial Feature Transformation:** Increasing model complexity by adding polynomial terms.

## 🙌 Author

**Faiez Tariq**


