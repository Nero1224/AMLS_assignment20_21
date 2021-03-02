from pandas import Series
from sklearn.preprocessing import MinMaxScaler

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
from sklearn.preprocessing import StandardScaler
from math import sqrt

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

# one hot encoding
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

classes = ['class1', 'class2', 'class4', 'class4', 'class2', 'class1']
classes = np.array(classes)
print(classes)

integer_encoder = LabelEncoder()
integer_encoder = integer_encoder.fit(classes)
inter_coded_classes = integer_encoder.transform(classes)
print(inter_coded_classes)

onehot_encoder = OneHotEncoder(sparse=False)
onehot_coded_classes = onehot_encoder.fit_transform(inter_coded_classes.reshape(len(inter_coded_classes), 1))
print(onehot_coded_classes)
max_data = integer_encoder.inverse_transform([np.argmax(onehot_coded_classes[0, :])])
print(max_data)

# sequences length unify
from keras.preprocessing.sequence import pad_sequences

sequences = [[0, 1, 2, 3],
             [0, 1, 2, 3, 4, 5],
             [0],
             [0, 1]]

print(pad_sequences(sequences))
print(pad_sequences(sequences, padding='post'))
print(pad_sequences(sequences, maxlen=3))
print(pad_sequences(sequences, maxlen=3, truncating='post'))

# pandas shift
from pandas import DataFrame

sequence = DataFrame()
sequence['t'] = [x for x in range(8)]
print(sequence)
sequence['t-1'] = sequence['t'].shift(1)
print(sequence)
sequence['t+1'] = sequence['t'].shift(-1)
print(sequence)