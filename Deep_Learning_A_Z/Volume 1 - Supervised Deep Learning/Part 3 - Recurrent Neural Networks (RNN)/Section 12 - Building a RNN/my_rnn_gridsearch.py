#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 18:09:54 2020

@author: bhaskarbiswas
"""



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 2 - RNN with grid search

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

def build_regressor(optimizer):
    regressor = Sequential()
    regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50, return_sequences = True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units = 50))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units = 1))
    regressor.compile(optimizer = optimizer, loss = 'mean_squared_error')
    return regressor

regressor = KerasRegressor(build_fn = build_regressor, batch_size = 250, epochs = 5)
parameters = {'batch_size' : [250, 320],
              'epochs' : [2, 4],
              'optimizer' : ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = regressor, 
                          param_grid = parameters, 
                          scoring = 'neg_mean_squared_error',
                          cv = 2)

grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_score = grid_search.best_score_
best_regressor = grid_search.best_estimator_


# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price1 = best_regressor.predict(X_test).reshape(-1, 1)
predicted_stock_price = sc.inverse_transform(predicted_stock_price1)
MAPE = np.mean((abs(real_stock_price - predicted_stock_price)/real_stock_price))
max_error = max(abs(real_stock_price - predicted_stock_price))
min_error = min(abs(real_stock_price - predicted_stock_price))
avg_error = np.mean(abs(real_stock_price - predicted_stock_price))
print('MAPE = ', MAPE)
print('Max_error = ', max_error[0])
print('Min_error = ', min_error[0])
print('Avg_error = ', avg_error)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.text(60, 0.25, r'MAPE[0]')
plt.legend()
plt.show()

