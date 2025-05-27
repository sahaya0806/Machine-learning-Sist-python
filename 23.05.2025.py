# Linear Regression only using numpy 
import numpy as np
class LinearRegression:
    def __init__(self):
        self.theta = None  

    def fit(self, X, y):
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((ones, X))  
        XTX = X_b.T @ X_b
        XTy = X_b.T @ y
        self.theta = np.linalg.inv(XTX) @ XTy

    def predict(self, X):
        ones = np.ones((X.shape[0], 1))
        X_b = np.hstack((ones, X))
        return X_b @ self.theta

    def get_params(self):
        return self.theta
    
if __name__ == "__main__":
    # No of hrs studied
    X_train = np.array([[1], [2], [4], [3], [5]])
    # Marks obtained 
    y_train = np.array([12, 13, 16, 15, 19])
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model parameters ", model.get_params())
    X_test = np.array([[6], [7]])
    predictions = model.predict(X_test)
    print("Predictions", predictions)
