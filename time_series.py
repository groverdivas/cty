import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates

import warnings
warnings.filterwarnings('ignore')

#from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

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
def moving_average(df, window, plot_intervals=False, plot_anomalies=False, scale=1.96):

    fg = 2016
    rolling_mean = df['Value'].rolling(window=window).mean()

    dte = np.array(df.index)
    #dt = np.array(df.index.apply(lambda x: dates.datestr2num(x)*1000000))

    plt.figure(figsize=(30,15))
    plt.plot(dte[len(df['Value'])-fg:len(df['Value'])-1],rolling_mean[len(rolling_mean)-fg:len(rolling_mean)-1], "g", label = "Rolling Mean trend")

    #dt = new_x = [x lambda x : dates.datestr2num(df['Date'])]
    #print(rolling_mean[1:10])
    #plt.plot(df['Value'], "b", label = "Rolling Mean trend",alpha=0.4)

    if plot_intervals:
        mae = mean_absolute_error(df['Value'][window:], rolling_mean[window:])
        deviation = np.std(df['Value'][window:] - rolling_mean[window:])
        lower_bond = rolling_mean - (mae + scale * deviation)
        upper_bond = rolling_mean + (mae + scale * deviation)
        plt.plot(dte[len(df['Value'])-fg:len(df['Value'])-1],upper_bond[len(upper_bond)-fg:len(upper_bond)-1], "r--", label="Upper Bond / Lower Bond")
        plt.plot(dte[len(df['Value'])-fg:len(df['Value'])-1],lower_bond[len(lower_bond)-fg:len(lower_bond)-1], "r--")

        # if plot_anomalies:
        #     anomalies = pd.DataFrame(index=df['Value'].index, columns=df['Value'].columns)
        #     anomalies[df['Value']<lower_bond] = df['Value'][df['Value']<lower_bond]
        #     anomalies[df['Value']>upper_bond] = df['Value'][df['Value']>upper_bond]
        #     plt.plot(anomalies, "ro", markersize=10)

    plt.plot(dte[len(df['Value'])-fg:len(df['Value'])-1],df['Value'][len(df['Value'])-fg:len(df['Value'])-1], "b",label="Actual values",alpha = 0.4)
    plt.legend(loc="upper left")
    plt.locator_params(axis='x', nbins=45)

    # loc, labels = plt.xticks()
    # loc = np.array(loc)
    # sorter = np.argsort(dt)
    # print(len(dt),len(loc),len(sorter))
    #ind = sorter[np.searchsorted(dt, loc, sorter=sorter)]
    #lbl = dte[ind]

    #plt.xticks(ind, lbl)5



    plt.show()
##########################################
##########################################

##########################################
##########################################
def main():
    demand = get_data_month()
    # print(demand.head())
    # print(demand.index)
    # demand.to_csv("demand_again.csv", sep=',')
    moving_average(demand, plot_intervals=True, window=72, plot_anomalies=True)
##########################################
##########################################
if __name__ == "__main__":
    main()

