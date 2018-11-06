import pandas as pd
import matplotlib.pyplot as plt

def get_data():
	df = pd.read_csv('clean_demand_added_hour.csv', parse_dates=['Date'], infer_datetime_format=True, index_col='Date')
	return df


def analysis(df, hours):
	avg_demand_hour = {}
	for h in hours:
		al = df['Hour'].index[df['Hour'] == h].tolist()
		dmd = df.loc[al,:]
		dmd = dmd['Value'].sum()
		#print(type(dmd['Value']), len(dmd))
		avg_demand_hour[h] = dmd/len(al)
	return avg_demand_hour



df=get_data()
df = df['2018-7-22 00:00:00':'2018-10-18 23:55:00']
hours = [i for i in range(0,24)]
a = analysis(df, hours)

plt.bar(a.keys(), a.values())
plt.locator_params(axis='x', nbins=24)
plt.show()
#print(a)