import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('housing.csv')
dataset.dropna(inplace=True)
X = dataset[['median_income', 'housing_median_age', 'total_rooms', 'population']].values
y = dataset['median_house_value'].values
print("Selected Features \n", X[:5])
print("Target \n", y[:5])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print("Predicted vs Actual \n", np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))
