import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
dataset = pd.read_csv("auto-mpg.csv")
dataset["horsepower"].replace("?", np.nan, inplace=True)
dataset["horsepower"] = dataset["horsepower"].astype(float)
dataset.dropna(subset=["horsepower", "mpg"], inplace=True)
X = dataset["horsepower"].values.reshape(-1, 1)
y = dataset["mpg"].values
X_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='gray', alpha=0.5, label='Data')
colors = ['r', 'g', 'b']
degrees = [1, 2, 3]
for i, degree in enumerate(degrees):
    poly_matrix = PolynomialFeatures(degree)
    X_poly = poly_matrix.fit_transform(X)
    X_range_poly = poly_matrix.transform(X_range)
    lin_reg = LinearRegression()
    lin_reg.fit(X_poly, y)
    y_pred = lin_reg.predict(X_poly)
    y_pred_range = lin_reg.predict(X_range_poly)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2score = r2_score(y, y_pred)
    print(f"Degree {degree} MSE={mse:.2f}, RMSE={rmse:.2f}, R²={r2score:.2f}")
    plt.plot(X_range, y_pred_range, color=colors[i], label=f'Degree {degree} ')
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Polynomial Regression ")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
