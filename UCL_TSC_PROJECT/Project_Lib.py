from sklearn.preprocessing import StandardScaler
from keras import models, layers, optimizers
from keras.utils import np_utils
import pandas
import os
import numpy as np

# load nine axes sensor data (the shape for each train axis is (7352, 128) for test is (2947, 128))
# 128 is the number of time steps; 50Hz*2,56s(window width) = 128
def load_file(path):
    data = pandas.read_csv(path, delim_whitespace=True, header=None)
    return data.values


def build_dataset(dir_path, file, type):
    files_list = []
    paths_list = []
    data = []
    label = []

    for root, dirs, file_names in os.walk(dir_path + "/" + file):
        for file_name in file_names:
            files_list.append(file_name)
            paths_list.append(os.path.join(root, file_name))

    for file_path in paths_list:
        data.append(load_file(file_path))
    # np.stack() can combine array in the depth dimension. It is suitable for 3D array
    # for the array only has 1 or 2 dimension like (m,n) or (m,1), can transfer them into
    # (m,n,1) and (1,m,1), then conduct stack operation
    data = np.dstack(data)

    label = load_file(dir_path + '/y_' + type + '.txt')
    print(type + ": data shape: {}; label shape: {}".format(data.shape, label.shape))
    return data, label


def distribution_ana(data):
    data = pandas.DataFrame(data)
    print(data.groupby(0).size())
    quantity = data.groupby(0).size().values
    print(quantity)

    for n in range(len(quantity)):
        print("For type " + "{}: {:.2%}".format(n, quantity[n]/sum(quantity)))


def overlapping_remove(data):
    middle = int(data.shape[1]/2)
    data = data[:, 0:middle, :]
    return data


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


def cnn_1d(data_train, label_train, data_test, label_test):
    label_train = np_utils.to_categorical(label_train - 1)
    label_test = np_utils.to_categorical(label_test - 1)
    #print("Model building begin:")
    cnn_1d = models.Sequential()
    cnn_1d.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu',
                             input_shape=(data_train.shape[1], data_train.shape[2])))
    cnn_1d.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    cnn_1d.add(layers.Dropout(0.5))
    cnn_1d.add(layers.MaxPooling1D(pool_size=2))
    cnn_1d.add(layers.Flatten())
    cnn_1d.add(layers.Dense(75, activation='relu'))
    cnn_1d.add(layers.Dense(6, activation='softmax'))
    #cnn_1d.summary()
    cnn_1d.compile(loss='categorical_crossentropy',
                   optimizer='adam',
                   metrics=['acc'])

    cnn_1d.fit(data_train, label_train, epochs=20, batch_size=16, verbose=0)

    _, acc = cnn_1d.evaluate(data_test, label_test, batch_size=16, verbose=0)

    return acc


def lstm(data_train, label_train, data_test, label_test):
    label_train = np_utils.to_categorical(label_train - 1)
    label_test = np_utils.to_categorical(label_test - 1)
    #print("Model building begin:")
    lstm = models.Sequential()
    lstm.add(layers.LSTM(100, input_shape=(data_train.shape[1], data_train.shape[2])))
    lstm.add(layers.Dropout(0.5))
    lstm.add(layers.Dense(100, activation='relu'))
    lstm.add(layers.Dense(6, activation='softmax'))
    #lstm.summary()
    lstm.compile(loss='categorical_crossentropy',
                   optimizer='adam',
                   metrics=['acc'])

    lstm.fit(data_train, label_train, epochs=15, batch_size=64, verbose=0)

    _, acc = lstm.evaluate(data_test, label_test, batch_size=32, verbose=0)

    return acc


def cnn_lstm(data_train, label_train, data_test, label_test, block_num):
    data_train = data_train.reshape(data_train.shape[0], block_num, int(data_train.shape[1] / block_num), data_train.shape[2])
    data_test = data_test.reshape(data_test.shape[0], block_num, int(data_test.shape[1] / block_num), data_test.shape[2])

    label_train = np_utils.to_categorical(label_train - 1)
    label_test = np_utils.to_categorical(label_test - 1)
    #print("Model building begin:")
    cnn_lstm = models.Sequential()
    cnn_lstm.add(layers.TimeDistributed(layers.Conv1D(filters=64, kernel_size=3,
                                                      activation='relu',
                                                      input_shape=(None, data_train.shape[2], data_train.shape[3]))))
    cnn_lstm.add(layers.TimeDistributed(layers.Conv1D(filters=64, kernel_size=3, activation='relu')))
    cnn_lstm.add(layers.TimeDistributed(layers.Dropout(0.5)))
    cnn_lstm.add(layers.TimeDistributed(layers.MaxPooling1D(pool_size=2)))
    cnn_lstm.add(layers.TimeDistributed(layers.Flatten()))
    cnn_lstm.add(layers.LSTM(100, input_shape=()))
    cnn_lstm.add(layers.Dropout(0.5))
    cnn_lstm.add(layers.Dense(100, activation='relu'))
    cnn_lstm.add(layers.Dense(6, activation='softmax'))
    #cnn_1d.summary()
    cnn_lstm.compile(loss='categorical_crossentropy',
                   optimizer='adam',
                   metrics=['acc'])

    cnn_lstm.fit(data_train, label_train, epochs=10, batch_size=32, verbose=0)

    _, acc = cnn_lstm.evaluate(data_test, label_test, batch_size=32, verbose=0)

    return acc


def conv_lstm(data_train, label_train, data_test, label_test, block_num):
    data_train = data_train.reshape(data_train.shape[0], block_num, 1, int(data_train.shape[1] / block_num), data_train.shape[2])
    data_test = data_test.reshape(data_test.shape[0], block_num, 1, int(data_test.shape[1] / block_num), data_test.shape[2])

    label_train = np_utils.to_categorical(label_train - 1)
    label_test = np_utils.to_categorical(label_test - 1)
    #print("Model building begin:")
    conv_lstm = models.Sequential()
    conv_lstm.add(layers.ConvLSTM2D(filters=64, kernel_size=(1,3),
                                    activation='relu',
                                    input_shape=(block_num, 1, data_train.shape[3], data_train.shape[4])))
    conv_lstm.add(layers.Dropout(0.5))
    conv_lstm.add(layers.Flatten())
    conv_lstm.add(layers.Dense(100, activation='relu'))
    conv_lstm.add(layers.Dense(6, activation='softmax'))
    #cnn_1d.summary()
    conv_lstm.compile(loss='categorical_crossentropy',
                   optimizer='adam',
                   metrics=['acc'])

    conv_lstm.fit(data_train, label_train, epochs=25, batch_size=64, verbose=0)

    _, acc = conv_lstm.evaluate(data_test, label_test, batch_size=32, verbose=0)

    return acc


def evaluate_repreat(data_train, label_train, data_test, label_test, model, times):
    scores = []

    if model == 'cnn':
        print("CNN:")
        for n in range(times):
            print("Training %.d begin:" % (n + 1))
            acc = cnn_1d(data_train, label_train, data_test, label_test)
            scores.append(acc)
            print("acc: {:.5%}".format(acc))
    elif model == 'lstm':
        print("LSTM:")
        for n in range(times):
            print("Training %.d begin:" % (n + 1))
            acc = lstm(data_train, label_train, data_test, label_test)
            scores.append(acc)
            print("acc: {:.5%}".format(acc))
    elif model == 'cnn_lstm':
        print("CNN-LSTM")
        for n in range(times):
            print("Training %.d begin:" % (n + 1))
            acc = cnn_lstm(data_train, label_train, data_test, label_test, 4)
            scores.append(acc)
            print("acc: {:.5%}".format(acc))
    elif model == 'conv_lstm':
        print("ConvLSTM:")
        for n in range(times):
            print("Training %.d begin:" % (n + 1))
            acc = conv_lstm(data_train, label_train, data_test, label_test, 4)
            scores.append(acc)
            print("acc: {:.5%}".format(acc))

    print("Mean score: {:.5}; Std:: +/- {:.5}".format(np.mean(scores), np.std(scores)))

