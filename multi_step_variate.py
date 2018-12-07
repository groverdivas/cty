# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 08:03:46 2018

@author: Divas Grover
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt


def lagger(df, c, ft):
    df_lag = pd.DataFrame(index=df.index, columns = [ft+"_lag_"+str(g) for g in range(0, c)])
    for i in range(0,c):
        v = df.shift(-i)
        df_lag[ft+"_lag_"+str(i)] = v       
    df_lag.dropna(inplace=True)
    
    return df_lag
#######################################
#######################################
def scale(train, test):
    scaler  = MinMaxScaler(feature_range=(0,1))
    scaler.fit(train)
    train_scale = scaler.transform(train)
    test_scale = scaler.transform(test)
    return scaler, train_scale, test_scale
#######################################
#######################################
def fit_lstm(trainX, trainy, epoch, batch_size, nb_epoch, n_neurons):
    trainX = trainX.reshape(trainX.shape[0], 1, trainX.shape[1])
    
    model = Sequential()
    model.add(LSTM(n_neurons, batch_input_shape=(batch_size, trainX.shape[1], trainX.shape[2]),stateful=True))
    
    model.add(Dense(trainy.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary)
    
    for i in range(nb_epoch):
        model.fit(trainX, trainy, batch_size=batch_size, shuffle=False)
        model.reset_states()
    return model
        
    
#######################################
#######################################
def forecast_lstm(model, X, n_batch):
    X = X.reshape(1,1, len(X))
    forecast = model.predict(X, batch_size=n_batch)
    return [x for x in forecast[0, :]]
#######################################
#######################################
def forecast(model, testX, n_batch):
    forecasts =[]
    for i in range(len(testX)):
        forecast_ = forecast_lstm(model, testX, n_batch)
        forecasts = forecasts + forecast_
    return forecasts
#######################################
#######################################
def inverse_scale(forecasts, scaler):
    inverted = list()
    #for i in range(0, len(forecasts)):
        # create array from forecast
        #forecast = np.array(forecasts[i])
    forecast = forecasts.reshape(1, forecasts.shape[0])
        # invert scaling
    inv_scale = scaler.inverse_transform(forecast)
    inv_scale = inv_scale[0, :]
    inverted.append(inv_scale)
    return inverted
#######################################
#######################################
def evaluate_(actual, forecasts):
    for i in range(0,23):
        a = [row[i] for row in actual]
        predicted = [forecast[i] for forecast in forecasts]
        rmse = sqrt(mean_squared_error(a, predicted))
        print('t+%d RMSE: %f' % ((i+1), rmse))
    return 0
#######################################
#######################################
def plot_forecasts(series, forecasts, n_test):
    # plot the entire dataset in blue
    plt.plot(series.values)
    # plot the forecasts in red
    for i in range(len(forecasts)):
        print(i)
        off_s = len(series) - n_test + i - 1
#        print(off_s)
        off_e = off_s + len(forecasts[i]) + 1
#        print(off_e)
        xaxis = [x for x in range(off_s, off_e)]
#        print(len(forecasts[i]))
#        print(forecast[i])
        yaxis = [series.values[off_s]]# + forecasts[i]
        print(type(yaxis))
        for e in forecasts[i]:
            yaxis.append(e)
        print(type(yaxis), len(yaxis))
        print(yaxis)
#        print(yaxis)
        plt.plot(xaxis, yaxis, color='red')
        # show the plot
    plt.show()
#######################################
#######################################   
def moving_average(df, window, plot_intervals=False, plot_anomalies=False, scale=1.96):
    
    rolling_mean = pd.DataFrame(index=df.index, columns=['Value', 'Temp'])

    
    rolling_mean['Value'] = df['Value'].rolling(window=window).mean()
    rolling_mean['Temp'] = df['Temp'].rolling(window=window).mean()
    #rolling_mean.dropna(inplace=True)
    
    return rolling_mean
#######################################
#######################################
    
#load and arrange
data = pd.read_csv('data.csv', parse_dates=["Date"], infer_datetime_format=True, index_col='Date')
#temp = data['Temp']
#val = data['Value']
data = data[['Hour', 'Temp', 'Weekday', 'Value']]

#smooth and update
data_smooth = moving_average(data, window = 24)
data['Temp'] = data_smooth['Temp']
data['Value'] = data_smooth['Value']
data.dropna(inplace=True)
del data_smooth
#lage and update
lag = 24
data_lag = pd.DataFrame(index=data.index)
data_lag_ = pd.DataFrame(index=data.index)


for ft in data.columns:
    data_lag_ = lagger(data[ft], lag, ft)
    data_lag = pd.concat([data_lag, data_lag_], axis=1)

data_lag.dropna(inplace=True)
del data_lag_

feature = data_lag.values

train = feature[:feature.shape[0]-2,:]
test = feature[feature.shape[0]-2:,:]

scaler = MinMaxScaler(feature_range=(0,1))
scale = scaler.fit(train)
train_s = scale.transform(train)
test_s = scale.transform(test)

trainX = train_s[:,:-2]
trainy = train_s[:,-2:]

testX = test_s[:,:-2]
testy = test_s[:,-2:]


model = fit_lstm(trainX, trainy, 1, 1, 1, 4)

forecasts = forecast_lstm(model, testX, 1)
forecasts = inverse_scale(forecasts, scale)


rmse = sqrt(mean_squared_error(testy, forecasts))

plot_forecasts(data_lag['Value_lag_0'], forecasts, 1+1)