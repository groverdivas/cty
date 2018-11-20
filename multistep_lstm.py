# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 09:57:09 2018

@author: Divas Grover
"""
from numpy import array
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

#######################################
#######################################
def split_sequence(seq, step_in, step_out):
    X, y = list(), list()
    for i in range(len(seq)):
        end_ix = i + step_in
        out_ix = end_ix + step_out
        
        if out_ix > len(seq):
            break
        seq_x, seq_y = seq[i:end_ix], seq[end_ix:out_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)
#######################################
#######################################
df = pd.read_csv("clean_demand.csv", parse_dates=["Date"], infer_datetime_format=True, index_col='Date')
seq = list(df["Value"])
raw_seq = seq[0:-10]
x_input = array(seq[-10:]) 

n_steps_in, n_steps_out = 10, 9
# split into samples
X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))
model.add(LSTM(100, activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam', loss='mse')
# fit model
print(model.summary())
model.fit(X, y, epochs=10)
# demonstrate prediction
#x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps_in, n_features))
yhat = model.predict(x_input)
print(yhat)
print("actual #############")
print(x_input)