
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt


# In[2]:


data = pd.read_csv('data.csv', parse_dates=["Date"], infer_datetime_format=True, index_col='Date')
data = data[['Hour', 'Temp', 'Weekday', 'Value']]
data = data.drop(['Weekday'], axis=1)
data.head()


# In[3]:


def moving_average(df, window):

    rolling_mean = pd.DataFrame(index=df.index, columns=['Value', 'Temp'])
    rolling_mean['Value'] = df['Value'].rolling(window=window).mean()
    rolling_mean['Temp'] = df['Temp'].rolling(window=window).mean()
    #rolling_mean.dropna(inplace=True)

    return rolling_mean


# In[4]:


smooth = moving_average(data, window=24)
data['Temp'] = smooth['Temp']
data['Value'] = smooth['Value']
data.dropna(inplace=True)
data.head()


# In[5]:


def lagger(df, c, ft):
    df_lag = pd.DataFrame(index=df.index, columns = [ft+"_lag_"+str(g) for g in range(0, c)])
    for i in range(0,c):
        v = df.shift(-i)
        df_lag[ft+"_lag_"+str(i)] = v
    df_lag.dropna(inplace=True)
    return df_lag


# In[6]:


lag = 24
data_lag = pd.DataFrame(index=data.index)
data_lag_ = pd.DataFrame(index=data.index)
for ft in data.columns:
    data_lag_ = lagger(data[ft], lag, ft)
    data_lag = pd.concat([data_lag, data_lag_], axis=1)

data_lag.dropna(inplace=True)
del data_lag_
data_lag.head()


# In[7]:


# for j in ['Hour', 'Temp']:
#     for i in range(24,48):
#         data_lag = data_lag.drop([j+"_lag_"+str(i)], axis=1)
        
# data_lag.head()


# In[8]:


data_lag.columns


# In[9]:


values = data_lag.values

scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(values)
#scaled = scale.transform(values)

scaler_ = MinMaxScaler(feature_range=(0,1))
scale_ = scaler_.fit_transform(values[:,-12:])


# In[10]:


train = scaled[:-10,:]
test = scaled[-10:,:]

train_Xh = train[:,:24]#, train[:,12:13]
test_Xh = test[:,:24]

train_Xt = train[:,24:48]
test_Xt = test[:,24:48]

train_X, train_y = train[:,48:-12], train[:,-12:]
test_X, test_y = test[:,48:-12], test[:,-12:]



train_Xh = np.reshape(train_Xh, (train_Xh.shape[0], 1, train_Xh.shape[1]))
test_Xh = np.reshape(test_Xh, (test_Xh.shape[0], 1, test_Xh.shape[1]))

train_Xt = np.reshape(train_Xt, (train_Xt.shape[0], 1, train_Xt.shape[1]))
test_Xt = np.reshape(test_Xt, (test_Xt.shape[0], 1, test_Xt.shape[1]))


train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

print('shape of trainh', train_Xh.shape)
print('shape of traint', train_Xt.shape)

print('shape of train', train_X.shape, train_y.shape)
print('shape of test', test_X.shape, test_y.shape)


# In[11]:


from keras.layers import Input, Dense
from keras.models import Model
from keras.layers import concatenate

hour_in = Input(shape=(train_Xh.shape[1] ,train_Xh.shape[2]), name='hour_in')
temp_in = Input(shape=(train_Xt.shape[1], train_Xt.shape[2]), name='temp_in')
value_in = Input(shape=(train_X.shape[1], train_X.shape[2]), name='value_in')


# In[12]:


hour = LSTM(64, batch_input_shape=(1, train_Xh.shape[1], train_Xh.shape[2]), activation='relu')(hour_in)
temp = LSTM(64, batch_input_shape=(1, train_Xt.shape[1], train_Xt.shape[2]), activation='relu')(temp_in)
value = LSTM(64, batch_input_shape=(1, train_X.shape[1],train_X.shape[2]), activation='relu')(value_in)


# In[13]:


mid = concatenate([hour, temp, value])
mid = Dense(12, activation='relu')(mid)


# In[14]:


model = Model(inputs=[hour_in, temp_in, value_in], outputs=[mid])


# In[15]:


model.summary()


# In[ ]:


model.compile(optimizer='adam', loss='mean_absolute_error')
model.fit([train_Xh, train_Xt, train_X], train_y, epochs=50 , batch_size=1)


# In[ ]:


test_Xh = test[:,:24]
test_Xt = test[:,24:48]

test_Xh = np.reshape(test_Xh, (test_Xh.shape[0], 1, test_Xh.shape[1]))
test_Xt = np.reshape(test_Xt, (test_Xt.shape[0], 1, test_Xt.shape[1]))


# In[ ]:


pred = model.predict([test_Xh, test_Xt, test_X], batch_size=1)


# In[ ]:


# test_Xh = test[:,:24]
# test_Xt = test[:,24:48]

# test_Xh = np.reshape(test_Xh, (test_Xh.shape[0], 1, test_Xh.shape[1]))
# test_Xt = np.reshape(test_Xt, (test_Xt.shape[0], 1, test_Xt.shape[1]))


# In[ ]:


act = test_y


# In[ ]:


import matplotlib.pyplot as plt


plt.plot(pred[0], 'r')
plt.plot(act[0])


# In[ ]:


rsme = np.sqrt(np.mean((pred[0]-act[0])**2))
rsme


# In[ ]:


scal = MinMaxScaler(feature_range=(0,1))
scald = scal.fit_transform(values[:, -12:])

pred_inv = scal.inverse_transform(pred)
act_inv = scal.inverse_transform(act)


# In[ ]:


rsme = np.sqrt(np.mean((pred_inv[0]-act_inv[0])**2))
rsme


# In[ ]:




clr = ['r','b','c','g','y','m', 'k', 'r', 'b','g']
def plot_forecasts(series, forecasts, n_test):
    s = 0
    plt.style.use('seaborn-dark')
#     plt.style.use('fivethirtyeight')
    plt.figure(figsize=(20,10))
    plt.plot(series.values, linewidth=10, alpha=0.3,label='Actual')
    for i in range(len(forecasts)):
        #print(i)
        off_s = len(series) - n_test + i - 1
        off_e = off_s + len(forecasts[i]) + 1
        xaxis = [x for x in range(off_s-9, off_e-9)]

        yaxis = [series.values[off_s-9]]# + forecasts[i]
        #print(type(yaxis))
#         yaxis=[]
        for e in forecasts[i]:
            yaxis.append(e)
        #print(type(yaxis), len(yaxis))
        #print(yaxis)
        s = s +0.1
#         if i%2 != 0:
        plt.plot(xaxis, yaxis, label="test"+str(i+1))#, color=clr[i])#, alpha = s)

    plt.legend(loc = 'upper left', fontsize=20)
    plt.rc('xtick',labelsize=20)
    plt.rc('ytick',labelsize=20)
    plt.xlabel('Sample No.', fontsize=20)
    plt.ylabel('Demand (kW)', fontsize=20)
    plt.title('LSTM prediction for an hour ahead', fontsize=20)
    plt.savefig('LSTM.png', dpi=200)


# In[ ]:


plot_forecasts(data['Value'][-25:], pred_inv, 10+2)


# In[ ]:


plt.style.available


# In[ ]:


rsme

