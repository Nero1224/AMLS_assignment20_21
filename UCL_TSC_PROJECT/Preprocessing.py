from pandas import Series
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from math import sqrt

data = [10, 20, 30, 40 ,45, 59, 60, 100, 120]
print(data)
data = Series(data)
print(data)

values = data.values
print(values)
values = values.reshape(len(values), 1)
print(values)

# normalization
scaler = MinMaxScaler(feature_range=(0,1))
scaler = scaler.fit(values)
print("Minimum: %f; Maximum: %f" % (scaler.data_min_, scaler.data_max_))

normalized_data = scaler.transform(values)
print(normalized_data)

inversed_data = scaler.inverse_transform(normalized_data)
print(inversed_data)

# standardization
data = [1.3, 2.5, 3.7, 4.0 ,4.5, 9.1, 3.0, 5.7, 1.1]
print(data)
data = Series(data)
print(data)

values = data.values
print(values)
values = values.reshape(len(values), 1)
print(values)

standard_scaler = StandardScaler()
standard_scaler = standard_scaler.fit(values)
print("Average: %f; Standard Deviation: %f" % (standard_scaler.mean_, sqrt(standard_scaler.var_)))

standard_data = standard_scaler.transform(values)
print(standard_data)

inversed_standard_data = standard_scaler.inverse_transform(standard_data)
print(inversed_standard_data)