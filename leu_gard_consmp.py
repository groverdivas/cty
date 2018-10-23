# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 07:58:27 2018

@author: Divas Grover
"""

from zeep import Client
from zeep.transports import Transport
from requests.auth import HTTPBasicAuth
from requests import Session
from zeep.wsse.username import UsernameToken
import pandas as pd
from datetime import datetime, timedelta
import time


def setup():
    user = ''
    password = ''
    session_ = Session()
    session_.auth = HTTPBasicAuth(user, password)
    client  = Client('http://alcapps/_common/webservices/Trend?wsdl',transport=Transport(session=session_), wsse=UsernameToken(user, password))
    return client

def get_time():
    time_start = datetime.now()
    time_start = time_start - timedelta(days = 1)
    time_start = time_start + timedelta(minutes = 5)
    time_end = time_start + timedelta(minutes = 5)
    time_start = time_start.strftime("%m/%d/%Y %I:%M:%S %p")
    time_end = time_end.strftime("%m/%d/%Y %I:%M:%S %p")
    return time_start, time_end
    

def get_data(client):
    c = 0
    d = 0
    #df_temp = pd.DataFrame()
    df_week = pd.DataFrame(columns=["Time", "Value"])
    df_temp = pd.DataFrame(index = [0], columns=["Time", "Value"])
    while 1:
        time_ = get_time()
        
        temp = list(client.service.getTrendData("#leu_power_meter/m517",time_[0], time_[1], True, 2))
        df_temp["Time"][0] = temp[0]
        df_temp["Value"][0] = float(temp[1])
        frames = [df_week, df_temp]
        df_week = pd.concat(frames)
        
        print(c,"Getting for time:", time_[0])
        
        time.sleep(300)
        c = c+1

        if c == 180:
            d = d + 1
            fl = "test_overnight_leu"+str(d)+".csv"
            df_week.to_csv(fl, sep=",")
            c = 0 
            df_week = pd.DataFrame(columns=["Time", "Value"])
    return df_week
                                     
                                     
                                     
def main():
    client = setup()
    df = get_data(client)
    print(get_time())
    return df
    
    

if __name__ == "__main__":
    df = main()