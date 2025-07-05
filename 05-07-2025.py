import numpy as np
x = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])
y = np.array([0,1,1,1])
learning_rate = 0.1
epoches = 10 
weights = np.zeros(x.shape[1])
biases = 0
def activation_function(z):
    return 1 if z >= 0  else 0
for epoch in range(epoches):
    print(f"Epoch : {epoches+1}")
    for i in range(len(x)):
        z = np.dot(x[i],weights) + biases
        y_pred = activation_function(z) 
        error = y[i] - y_pred
        weights += learning_rate * error * x[i]
        biases += learning_rate * error
        print(f"Input {x[i]}, Predicted {y_pred}, Actual {y[i]}, Error {error}")
    print(f"Weights {weights}, Bias {biases}\n")

for i in range(len(x)):
    z = np.dot(x[i],weights) + biases
    y_pred = activation_function(z) 
    print(f"Input {x[i]} and the predicted value is  Prediction {y_pred}")