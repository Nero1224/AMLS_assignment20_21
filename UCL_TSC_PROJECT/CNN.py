import Load_Data as ld
from keras import layers, models, optimizers
import tensorflow as tf
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# GPU memory management
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

dir_path_train = r"C:/Users/ASUS/Desktop/AMLS_Assignment/AMLS_PROJECT/UCI_HAR_Dataset/train"
dir_path_test = r"C:/Users/ASUS/Desktop/AMLS_Assignment/AMLS_PROJECT/UCI_HAR_Dataset/test"

data_train, label_train = ld.build_dataset(dir_path_train, 'Inertial Signals', 'train')
data_test, label_test = ld.build_dataset(dir_path_test, 'Inertial Signals', 'test')
ld.distribution_ana(label_train)
ld.distribution_ana(label_test)
print(data_train.shape)
print(label_train.shape)
print(data_test.shape)
print(label_test.shape)

def evaluate_cnn_1d(data_train, label_train, data_test, label_test):
    label_train = np_utils.to_categorical(label_train - 1)
    label_test = np_utils.to_categorical(label_test - 1)
    #print("Model building begin:")
    cnn_1d = models.Sequential()
    cnn_1d.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu',
                             input_shape=(data_train.shape[1], data_train.shape[2])))
    cnn_1d.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    cnn_1d.add(layers.Dropout(0.5))
    cnn_1d.add(layers.MaxPooling1D(pool_size=2))
    cnn_1d.add(layers.Flatten())
    cnn_1d.add(layers.Dense(100, activation='relu'))
    cnn_1d.add(layers.Dense(6, activation='softmax'))
    #cnn_1d.summary()
    cnn_1d.compile(loss='categorical_crossentropy',
                   optimizer='adam',
                   metrics=['acc'])

    cnn_1d.fit(data_train, label_train, epochs=10, batch_size=32, verbose=0)

    _, acc = cnn_1d.evaluate(data_test, label_test, batch_size=32, verbose=0)

    return acc


def evaluate_repreat(data_train, label_train, data_test, label_test, model, times):
    scores = []
    for n in range(times):
        print("Training %.d begin:" % (n+1))
        acc = model(data_train, label_train, data_test, label_test)
        scores.append(acc)
        print("acc: {:.5%}".format(acc))

    print("Mean score: {:.5}; Std:: +/- {:.5}".format(np.mean(scores), np.std(scores)))


def overlapping_remove(data):
    middle = int(data.shape[1]/2)
    data = data[:, 0:middle, :]
    return data
"""
data = overlapping_remove(data_train)
data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])

features = ['acc ', 'gyro ', 'total ']
axes = ['x', 'y', 'z']
names = []

for feature in features:
    for axis in axes:
        names.append(feature + axis)

print(names)

fig, ax = plt.subplots(figsize=(10,20), ncols=1, nrows=9)
for n in range(len(names)):
    ax[n].hist(data[:, n], bins=200, density=True, histtype='bar', stacked=True, label='{}'.format(names[n]))
    ax[n].set_xlim([-1, 1])
    ax[n].legend()
plt.xlabel('timesteps')
plt.ylabel('values')
plt.tight_layout()
plt.show()
"""
def standardization(data_train, data_test):
    print("Standzrdization begin:")
    data_train_nolap = overlapping_remove(data_train)
    print(data_train_nolap.shape)
    data_train_nolap = data_train_nolap.reshape(data_train_nolap.shape[0]*data_train_nolap.shape[1], data_train_nolap.shape[2])
    data_train_reshape = data_train.reshape(data_train.shape[0]*data_train.shape[1], data_train.shape[2])
    data_test_reshape = data_test.reshape(data_test.shape[0]*data_test.shape[1], data_test.shape[2])

    std = StandardScaler()
    std.fit(data_train_nolap)

    data_train_std = std.transform(data_train_reshape).reshape(data_train.shape)
    data_test_std = std.transform(data_test_reshape).reshape(data_test.shape)

    print("Standardization end.")
    return data_train_std, data_test_std

data_train_std, data_test_std = standardization(data_train, data_test)
evaluate_repreat(data_train_std, label_train, data_test_std, label_test, evaluate_cnn_1d(), 10)
