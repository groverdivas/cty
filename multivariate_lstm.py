# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 12:25:43 2018

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

##load and shift for time series
data = pd.read_csv('data.csv', parse_dates=["Date"], infer_datetime_format=True, index_col='Date')
data['Value+1'] = data['Value'].shift(-1)
data.dropna(inplace = True)

##preprocess
values = data.values
values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(values)

##train test split
train = scaled[:-288,:]
test = scaled[data.shape[0]-288:,:]

train_X, train_y = train[:,:-1], train [:,-1]
test_X, test_y = test[:,:-1], test[:,-1]

train_X = train_X.reshape(train_X.shape[0], 1, train_X.shape[1])
test_X = test_X.reshape(test_X.shape[0], 1, test_X.shape[1])


##Model design
model = Sequential()
model.add(LSTM(50,batch_input_shape=(1, train_X.shape[1], train_X.shape[2]), stateful=True))
model.add(Dense(1))
model.compile(loss = 'mae', optimizer = 'adam')

print(model.summary())

history = model.fit(train_X, train_y, epochs=50, batch_size=1, validation_data=(test_X, test_y), shuffle=False)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
