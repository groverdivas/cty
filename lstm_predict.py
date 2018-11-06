# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 08:10:54 2018

@author: Divas Grover
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
#######################################
#######################################
def get_data():
    df = pd.read_csv("clean_demand.csv", parse_dates=["Date"], infer_datetime_format=True, index_col='Date')
    return df
#######################################
#######################################
def time_to_supervise(series, offset=1):
    df = pd.DataFrame(series)
    columns = [df.shift(i) for i in range (1, offset+1)]
    columns.append(df)
    df = pd.concat(columns, axis = 1)
    df.fillna(0, inplace=True)
    return df
#######################################
#######################################
def scaler(train, test):
    scale = MinMaxScaler(feature_range=(-1,1))
    scale.fit(train)
    train_scale = scale.transform(train)
    test_scale = scale.transform(test)
    return scale, train_scale, test_scale
#######################################
#######################################
def inverse_scaler(scale, X, yhat):
     new_row = [x for x in X] + [yhat]
     temp = np.array(new_row)
     temp = temp.reshape(1, len(temp))
     invert = scale.inverse_transform(temp)
     return invert[0,-1]
#######################################
#######################################
def model_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:,-1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, shuffle=False)
        model.reset_states()
    return model
#######################################
#######################################
def forecast(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]
#######################################
#######################################
demand = get_data()
sprvs_df = time_to_supervise(demand['Value'])
sprvs_df.columns = ['v1','v2']
train, test = sprvs_df.values[0:-288], sprvs_df.values[-288:]
scale, train_scale, test_scale = scaler(train, test)
model = model_lstm(train_scale, 1, 50, 4)
train_reshaped = train_scale[:,0].reshape(len(train_scale),1,1)
model.predict(train_reshaped, batch_size=1)

prediction = []

for i in range(len(test_scale)):
    X,y = test_scale[i, 0:-1], test_scale[i, -1]
    yhat = forecast(model, 1, X)
    yhat = inverse_scaler(scale, X, yhat)
    prediction.append(yhat)
    expected = demand['Value'][len(train) + i]
    print("Time= ",str(demand.index[len(train) + i]), "Predicted=", yhat, "Expected=", expected)
    

rmse = sqrt(mean_squared_error(test[:,1:2], prediction))
print("Error =", rmse)
plt.plot(test[:,1:2],"r")
plt.plot(prediction,"g")
plt.show()
    
#######################################
#######################################
