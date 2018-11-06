import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates

import warnings
warnings.filterwarnings('ignore')

#from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import TimeSeriesSplit

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
##########################################
##########################################
def get_data_month():
    demand = pd.read_csv("clean_demand.csv", parse_dates=["Date"], infer_datetime_format=True, index_col='Date')
    #demand = demand[17372:len(demand)]
    return demand
##########################################
##########################################
class HoltWinter:

    def __init__(self, df, slen, alpha, beta, gamma, n_pred, scale =1.96):
        self.df = df
        self.slen = slen
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_pred = n_pred
        self.scale = scale

    def init_trend(self):
        sum = 0.0
        for i in range(self.slen):
            sum += float(self.df['Value'][i+self.slen]-self.df['Value'][i])/self.slen
        return sum/self.slen

    def init_season(self):
        seasonal = {}
        season_avg = []
        n_season = int(len(self.df['Value'])/self.slen)

        for i in range(n_season):
            season_avg.append(sum(self.df['Value'][self.slen*i:self.slen*i+self.slen])/float(self.slen))

        for i in range(self.slen):
            sum_over_avg =0.0
            for j in range(n_season):
                sum_over_avg += self.df['Value'][self.slen*j+i]-season_avg[j]
            seasonal[i] = sum_over_avg/n_season
        return seasonal

    def triple_expo(self):
        self.result = []
        self.smooth = []
        self.season = []
        self.trend = []
        self.deviation = []
        self.upper = []
        self.lower = []

        seasonal = self.init_season()

        for i in range(len(self.df['Value'])+self.n_pred):
            if i == 0:
                smooth_ = self.df['Value'][0]
                trend_ = self.init_trend()
                self.result.append(self.df['Value'][0])
                self.trend.append(trend_)
                self.smooth.append(seasonal[i%self.slen])

                self.deviation.append(0)
                self.upper.append(self.result[0] + self.scale*self.deviation)
                self.lower.append(self.result[0] - self.scale*self.deviation)

                continue

            if i >= len(self.df['Value']):
                m = i - len(self.df['Value']) + 1
                self.result.append((smooth_ + m*trend_) + seasonal[i%self.slen])
                self.deviation.append(self.deviation[-1]*1.01)

            else:
                val = self.df['Value'][i]
                last_smooth, smooth_ = smooth_, self.alpha*(val-seasonal[i%self.slen]) + (1-self.alpha)*(smooth_+trend_)
                trend_ = self.beta*(smooth_-last_smooth) + (1-self.beta)*trend_
                seasonal[i%self.slen] = self.gamma*(val-smooth_) + (1-self.gamma)*seasonal[i%self.slen]
                self.result.append(smooth_+trend_+seasonal[i%self.slen])

                self.deviation.append(self.gamma*np.abs(self.df['Value'][i] - self.result[i]) + (1-self.gamma)*self.deviation[-1])

            self.upper.append(self.result[-1] + self.scale*self.deviation[-1])
            self.lower.append(self.result[-1] - self.scale*self.deviation[-1])

            self.smooth.append(smooth_)
            self.trend.append(trend_)
            self.season.append(seasonal[i%self.slen])
##########################################
##########################################
def plot_holt_winter(df, plot_intervals = False):
    
    dg = 288
    plt.figure(figsize=(20,10))
    plt.subplot(4,1,1)
    plt.plot(list(df['Value'][len(df['Value'])-dg:len(df['Value'])]),label='actual')
    
    plt.subplot(4,1,2)
    plt.plot(model.result[len(model.trend)-dg:len(model.season)],"r")
    plt.subplot(4,1,3)
    plt.plot(model.trend[len(model.trend)-dg:len(model.season)],"c")
    plt.subplot(4,1,4)
    plt.plot(model.season[len(model.trend)-dg:len(model.season)],"g")
    plt.show()
##########################################
##########################################
def cv_(alpha, beta, gamma, df,loss_function=mean_squared_error, slen=2016):
    error = []
    val = df.values
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    for train, test in tscv.split(val):
        model = HoltWinter(df=val[train], slen=slen, alpha=alpha, beta=beta, gamma=gamma, n_pred=len(test))
        model.triple_expo()
        predict = model.result[-len(test)]
        actual = val[test]
        err = loss_function(predict, actual)
        error.append(err)
    
    return np.mean(np.array(error))
##########################################
##########################################
# def main():
demand = get_data_month()
model = HoltWinter(demand, slen=2016, alpha=0.5, beta = 0.5, gamma = 0.5, n_pred=144, scale=1)
model.triple_expo()
re = model.result
tr = model.trend
se = model.season
plot_holt_winter(demand)

#par = np.array([i for i in range(0,100,5)])/100
#alpha = par
#beta = par
#gamma = par
#dem = demand['Value'][0:len(demand['Value'])-1000]
#er = {}
#a_ = 0
#for a in alpha:
#    b_ = 0 
#    for b in beta:
#        g_ = 0
#        for g in gamma:
#            print(a_,b_,g_)
#            #np.put(er, [a_,b_,g_],cv_(a,b,g,dem))
#            er[a_,b_,g_] = cv_(a,b,g,dem)
#            g_ = g_ + 1
#        b_ = b_+1
#    a_ = a_ + 1
#
##print(er.shape)
##########################################
##########################################
# if __name__ == "__main__":
#     main()
