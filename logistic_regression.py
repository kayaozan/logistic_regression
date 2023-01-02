# This is a sample code of logistic regression written from scratch by Ozan KAYA.
# https://github.com/kayaozan

# The dataset is called Titanic.
# It contains several data of passengers such as sex, age, fare, and whether they survived or not.
# The dataset can be obtained from https://www.openml.org/search?type=data&sort=runs&id=40945

# The aim of this code is to estimate the parameters of logistic model
# and predict the survival outcome.

def load_and_prepare():
# Loads the data, divides it into train and test sets and returns them.

    data = pd.read_csv('titanic3.csv',index_col=False)

    # Replacing text variables with binary values
    data.replace({'male': 1, 'female': 0}, inplace=True)
    
    # Dropping the rows that contain NaN and selecting the columns that will be used
    data = data[['sex', 'pclass','age','fare','survived']].dropna().to_numpy()    
    
    # 20% of the data to be selected for test purposes
    test_size = 0.2
    test_range = int(data.shape[0] * test_size)

    # Shuffling the data
    np.random.seed(0)
    np.random.shuffle(data)

    # Dividing the data set into training and test sets
    # Reshaping the arrays such that they work well with matrix operations
    x_test  = data[:test_range, 1:].reshape(test_range, -1)
    x_train = data[test_range:, 1:].reshape(data.shape[0]-test_range, -1)
    
    y_test  = data[:test_range, 0].reshape(-1, 1)
    y_train = data[test_range:, 0].reshape(-1, 1)

    return x_train, y_train, x_test, y_test

def normalize(x_train, x_test):
# Normalizes the feature sets.
# Uses the mean and standard deviation of the training set for both
# so that the regression model is not biased.

    trainMean = np.mean(x_train, axis=0)
    trainStd  = np.std( x_train, axis=0)
    
    x_train = (x_train - trainMean) / trainStd
    x_test  = (x_test  - trainMean) / trainStd

    return x_train, x_test

def sigmoid(z):
# Sigmoid function.

    return 1 / (1 + np.exp(-z))

def h(x, theta):
# The hypothesis of logistic regression.
# Sends the dot product of features and parameters to the sigmoid function.

    return sigmoid(x @ theta)

def cost_function(x, y, theta):
# Calculates the error for each sample.
# Uses the formula: (-1/m) * sum( y.T log(h) + (1-y) log(1-h) )
# where m is the size of samples

    cost = -1/y.shape[0] * (y.T @ np.log(h(x, theta)) + (1-y).T @ np.log(1 - h(x, theta)))
    
    return np.squeeze(cost)

def gradient_descend(x, y, theta, learning_rate=0.1, epochs=500):
# Calculates the gradient descend for each parameter and adjusts them.
# Cost of each run is stored for monitoring purposes.

    J = []
    for i in range(epochs):
        theta = theta - learning_rate/y.shape[0] * x.T @ (h(x, theta) - y)
        J.append(cost_function(x,y,theta))

    return theta, J

def predict(x, theta):
# Predicts the output values for given features and parameters.
# The prediction is 1 if the calculation is equal or greater than 0.5,
# or 0 if it is less than 0.5

     return h(x, theta) >= 0.5

def score(y_prediction, y):
# Compares the prediction values to real values, calculates the ratio of true predictions.

    return np.sum(y_prediction == y) / y.shape[0]

import pandas as pd
import numpy as np
import matplotlib.pyplot as pl

x_train, y_train, x_test, y_test = load_and_prepare()

# Normalizing the feature sets
x_train, x_test = normalize(x_train, x_test)

# Adding a column of ones to the input variables for bias parameters
x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
x_test  = np.hstack((np.ones((x_test.shape[0], 1)), x_test))

# Initializing the parameters with zeros
theta = np.zeros([x_train.shape[1], 1])

# Sending the training set and parameters to start gradient descend
theta, J = gradient_descend(x_train, y_train, theta)

# Plotting the cost to ensure gradient descend has worked as intended
pl.plot(J)
pl.xlabel('number of runs by gradient descend')
pl.ylabel('J (cost)')
pl.show()

# Testing the model with the test set
y_test_prediction = predict(x_test, theta)
accuracy = score(y_test_prediction, y_test)

print('The accuracy measured with test set:\n', accuracy)