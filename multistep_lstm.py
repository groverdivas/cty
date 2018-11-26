# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 09:57:09 2018

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
#def split_sequence(seq, step_in, step_out):
#    X, y = list(), list()
#    for i in range(len(seq)):
#        end_ix = i + step_in
#        out_ix = end_ix + step_out
#        
#        if out_ix > len(seq):
#            break
#        seq_x, seq_y = seq[i:end_ix], seq[end_ix:out_ix]
#        X.append(seq_x)
#        y.append(seq_y)
#    return array(X), array(y)
#######################################
#######################################
def lagger(df, c):
    df_lag = pd.DataFrame(index=df.index, columns = ["lag_"+str(g) for g in range(c)])
    for i in range(c):
        v = df.shift(-i)
        df_lag["lag_"+str(i)] = v       
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
def fit_lstm(train, n_in, n_out, batch_size, nb_epoch, n_neurons):
    X, y = train[:,0:n_in], train[:,n_in:]
    print(X.shape)
    
    X = X.reshape(X.shape[0], 1, X.shape[1])
    #model
    model = Sequential()
    model.add(LSTM(n_neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    print(y.shape)
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    # fit network
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, shuffle=False)
        model.reset_states()
    return model
#######################################
#######################################
def forecast_lstm(model, X, n_batch):
    X = X.reshape(1,1,len(X))
    forecast = model.predict(X, batch_size=n_batch)
    return [x for x in forecast[0, :]]
#######################################
#######################################
def forecast(model, n_batch, train, test, n_in, n_out):
    forecasts =[]
    for i in range(len(test)):
        X, y = test[i, 0:n_in], test[i, n_in:]
        forecast_ = forecast_lstm(model, X, n_batch)
        forecasts.append(forecast_)
    return forecasts
#######################################
#######################################
def inverse_scale(series, forecasts, scaler):
    inverted = list()
    for i in range(0, len(forecasts)):
        # create array from forecast
        forecast = np.array(forecasts[i])
        forecast = forecast.reshape(1, forecast.shape[0])
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

df = pd.read_csv("clean_demand.csv", parse_dates=["Date"], infer_datetime_format=True, index_col='Date')

train = df[:-2]
test = df[-2:]
scaler, train_scale, test_scale = scale(train, test)
#train_scale = list(train_scale)
#test_scale = list(test_scale)
#train_scale.append(test_scale)
train_scale = np.append(train_scale, test_scale)
df_ = pd.DataFrame(index=df.index, columns=['val'])
df_['val']=train_scale
lag = 24
df_lag = lagger(df_, lag)
train_lag_scaled = df_lag[:-2]
test_lag_scaled = df_lag[-2:]

model = fit_lstm(np.array(train_lag_scaled.values), 1, lag-1, 1, 15, 4)
forecasts = forecast(model, 1, np.array(train_lag_scaled.values), np.array(test_lag_scaled.values), n_in=1, n_out=lag-1)
#print(forecasts)
forecasts = inverse_scale(train, forecasts, scaler)

actual = [row[1:] for row in test_lag_scaled.values]
#print(actual)
actual = inverse_scale(train, actual, scaler)

evaluate_(actual, forecasts)
plot_forecasts(df['Value'], forecasts, 2+2)




#seq = list(df["Value"])
#raw_seq = seq[0:-144]
#x_input = array(seq[-144:]) 
#
#n_steps_in, n_steps_out =144, 143
## split into samples
#X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
## reshape from [samples, timesteps] into [samples, timesteps, features]
#n_features = 1
#X = X.reshape((X.shape[0], X.shape[1], n_features))



# define model
#model = Sequential()
#model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
#model.add(LSTM(100, activation='relu'))
#model.add(Dense(n_steps_out))
#model.compile(optimizer='adam', loss='mse')
## fit model
#print(model.summary())
#model.fit(X, y, epochs=10)
## demonstrate prediction
##x_input = array([70, 80, 90])
#x_input = x_input.reshape((1, n_steps_in, n_features))
#yhat = model.predict(x_input)
#print(yhat)
#print("actual #############")
#print(x_input)