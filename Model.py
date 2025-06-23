# import libraries.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



# Generate Synthetic Data (cosine function + noise).
rng = np.random.RandomState(0)
X = np.sort(rng.rand(30))
y = np.cos(1.5 * np.pi * X) + rng.randn(30) * 0.1
X = X.reshape(-1, 1)



# Split the Data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)



# -------------------------------------------------------------
# UNDERFITTING MODEL (Linear Regression Degree 1)
# -------------------------------------------------------------
poly1 = PolynomialFeatures(degree=1)
X_train_poly1 = poly1.fit_transform(X_train)
X_val_poly1 = poly1.transform(X_val)

model_underfit = LinearRegression()
model_underfit.fit(X_train_poly1, y_train)

y_train_pred_under = model_underfit.predict(X_train_poly1)
y_val_pred_under = model_underfit.predict(X_val_poly1)

mse_train_under = mean_squared_error(y_train, y_train_pred_under)
mse_val_under = mean_squared_error(y_val, y_val_pred_under)




# -------------------------------------------------------------
# OVERFITTING MODEL (Polynomial Degree 15)
# -------------------------------------------------------------
poly15 = PolynomialFeatures(degree=15)
X_train_poly15 = poly15.fit_transform(X_train)
X_val_poly15 = poly15.transform(X_val)

model_overfit = LinearRegression()
model_overfit.fit(X_train_poly15, y_train)

y_train_pred_over = model_overfit.predict(X_train_poly15)
y_val_pred_over = model_overfit.predict(X_val_poly15)

mse_train_over = mean_squared_error(y_train, y_train_pred_over)
mse_val_over = mean_squared_error(y_val, y_val_pred_over)




# -------------------------------------------------------------
# REGULARIZED MODEL (Ridge with Degree 15)
# -------------------------------------------------------------
model_ridge = Ridge(alpha=1.0)
model_ridge.fit(X_train_poly15, y_train)

y_train_pred_ridge = model_ridge.predict(X_train_poly15)
y_val_pred_ridge = model_ridge.predict(X_val_poly15)

mse_train_ridge = mean_squared_error(y_train, y_train_pred_ridge)
mse_val_ridge = mean_squared_error(y_val, y_val_pred_ridge)





# -------------------------------------------------------------
# RESULTS
# -------------------------------------------------------------
print("\nModel Performance Summary:")
print("--------------------------------------------")
print("Underfitting (Linear, Degree 1):")
print(f"Train MSE: {mse_train_under:.4f}, Validation MSE: {mse_val_under:.4f}")
print("--------------------------------------------")
print("Overfitting (Polynomial, Degree 15):")
print(f"Train MSE: {mse_train_over:.4f}, Validation MSE: {mse_val_over:.4f}")
print("--------------------------------------------")
print("Regularized (Ridge, Degree 15, L2):")
print(f"Train MSE: {mse_train_ridge:.4f}, Validation MSE: {mse_val_ridge:.4f}")
print("--------------------------------------------")




# -------------------------------------------------------------
# Prepare Plotting Data
# -------------------------------------------------------------
X_plot = np.linspace(0, 1, 100).reshape(-1, 1)
X_plot_poly1 = poly1.transform(X_plot)
y_plot_under = model_underfit.predict(X_plot_poly1)
X_plot_poly15 = poly15.transform(X_plot)
y_plot_over = model_overfit.predict(X_plot_poly15)
y_plot_ridge = model_ridge.predict(X_plot_poly15)




# -------------------------------------------------------------
# PLOT 1: Underfitting Model
# -------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='black', label='Training Data')
plt.scatter(X_val, y_val, color='blue', label='Validation Data')

plt.plot(X_plot, y_plot_under, color='green', label='Underfit Model (Degree 1)', linewidth=2)

plt.legend()
plt.title('Underfitting Model (Linear Regression)')
plt.xlabel('X')
plt.ylabel('y')
plt.show()




# -------------------------------------------------------------
# PLOT 2: Overfitting Model
# -------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='black', label='Training Data')
plt.scatter(X_val, y_val, color='blue', label='Validation Data')

plt.plot(X_plot, y_plot_over, color='red', label='Overfit Model (Polynomial Degree 15)', linewidth=2)

plt.legend()
plt.title('Overfitting Model (Polynomial Degree 15)')
plt.xlabel('X')
plt.ylabel('y')
plt.show()




# -------------------------------------------------------------
# PLOT 3: Regularized Model
# -------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='black', label='Training Data')
plt.scatter(X_val, y_val, color='blue', label='Validation Data')

plt.plot(X_plot, y_plot_ridge, color='purple', label='Regularized Model (Ridge, L2)', linewidth=2)

plt.legend()
plt.title('Regularized Model (Polynomial Degree 15 with L2)')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
