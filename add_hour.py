import pandas as pd

demand = pd.read_csv("clean_demand_hour_day.csv", parse_dates=['Date'], infer_datetime_format=True, index_col='Date')
r = pd.Series(demand.index)
r = r.apply(lambda x: x.time())
r = r.apply(lambda x: x.strftime("%H"))
hour = pd.DataFrame(index=demand.index, columns=['Hour'])

for i in range(0,len(r)):
	hour['Hour'][demand.index[i]] = int(r[i])
	#print(hour['Hour'][demand.index[i]])
result = pd.concat([demand, hour], axis = 1)

result.to_csv("clean_demand_added_hour.csv", sep=',')

print(result.head(), result.shape)
print(type(result['Hour'][result.index[1]]))