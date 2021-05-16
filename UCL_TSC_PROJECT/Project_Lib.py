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
    print("size: " + str(data.groupby(0).size()))
    quantity = data.groupby(0).size().values
    print("values: " + str(quantity))

    for n in range(len(quantity)):
        print("For type " + "{}: {:.2%}".format(n, quantity[n]/sum(quantity)))


def overlapping_remove(data):
    middle = int(data.shape[1]/2)
    data = data[:, 0:middle, :]
    return data

def normalization(data_train, data_test):
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
    cnn_1d.add(layers.Dense(64, activation='relu'))
    cnn_1d.add(layers.Dense(6, activation='softmax'))
    #cnn_1d.summary()
    cnn_1d.compile(loss='categorical_crossentropy',
                   optimizer='adam',
                   metrics=['acc'])

    history = cnn_1d.fit(data_train, label_train, epochs=20, batch_size=16, verbose=0, validation_split=0.2)

    _, te_acc = cnn_1d.evaluate(data_test, label_test, batch_size=16, verbose=0)

    return history, history.history['acc'][4], history.history['val_acc'][4],te_acc, history.history['loss'][4], history.history['val_loss'][4]


def lstm(data_train, label_train, data_test, label_test):
    label_train = np_utils.to_categorical(label_train - 1)
    label_test = np_utils.to_categorical(label_test - 1)
    #print("Model building begin:")
    lstm = models.Sequential()
    lstm.add(layers.LSTM(128, input_shape=(data_train.shape[1], data_train.shape[2])))
    lstm.add(layers.Dropout(0.5))
    lstm.add(layers.Dense(64, activation='relu'))
    lstm.add(layers.Dense(6, activation='softmax'))
    #lstm.summary()
    lstm.compile(loss='categorical_crossentropy',
                   optimizer='adam',
                   metrics=['acc'])

    history = lstm.fit(data_train, label_train, epochs=20, batch_size=16, verbose=0, validation_split=0.2)

    _, te_acc = lstm.evaluate(data_test, label_test, batch_size=32, verbose=0)

    return history, history.history['acc'][4], history.history['val_acc'][4], te_acc, history.history['loss'][4], history.history[
        'val_loss'][4]


def cnn_lstm(data_train, label_train, data_test, label_test, block_num):
    data_train = data_train.reshape(data_train.shape[0], block_num, int(data_train.shape[1] / block_num), data_train.shape[2])
    data_test = data_test.reshape(data_test.shape[0], block_num, int(data_test.shape[1] / block_num), data_test.shape[2])

    label_train = np_utils.to_categorical(label_train - 1)
    label_test = np_utils.to_categorical(label_test - 1)
    #print("Model building begin:")
    cnn_lstm = models.Sequential()
    cnn_lstm.add(layers.TimeDistributed(layers.Conv1D(filters=128, kernel_size=3,
                                                      activation='relu',
                                                      input_shape=(None, data_train.shape[2], data_train.shape[3]))))
    cnn_lstm.add(layers.TimeDistributed(layers.Conv1D(filters=64, kernel_size=3, activation='relu')))
    cnn_lstm.add(layers.TimeDistributed(layers.Dropout(0.5)))
    cnn_lstm.add(layers.TimeDistributed(layers.MaxPooling1D(pool_size=2)))
    cnn_lstm.add(layers.TimeDistributed(layers.Flatten()))
    cnn_lstm.add(layers.LSTM(128, input_shape=()))
    cnn_lstm.add(layers.Dropout(0.5))
    cnn_lstm.add(layers.Dense(64, activation='relu'))
    cnn_lstm.add(layers.Dense(6, activation='softmax'))
    #cnn_1d.summary()
    cnn_lstm.compile(loss='categorical_crossentropy',
                   optimizer='adam',
                   metrics=['acc'])

    history = cnn_lstm.fit(data_train, label_train, epochs=10, batch_size=32, verbose=0, validation_split=0.2)

    _, te_acc = cnn_lstm.evaluate(data_test, label_test, batch_size=32, verbose=0)

    return history, history.history['acc'][4], history.history['val_acc'][4], te_acc, history.history['loss'][4], history.history[
        'val_loss'][4]


def conv_lstm(data_train, label_train, data_test, label_test, block_num):
    data_train = data_train.reshape(data_train.shape[0], block_num, 1, int(data_train.shape[1] / block_num), data_train.shape[2])
    data_test = data_test.reshape(data_test.shape[0], block_num, 1, int(data_test.shape[1] / block_num), data_test.shape[2])

    label_train = np_utils.to_categorical(label_train - 1)
    label_test = np_utils.to_categorical(label_test - 1)
    #print("Model building begin:")
    conv_lstm = models.Sequential()
    conv_lstm.add(layers.ConvLSTM2D(filters=128, kernel_size=(1,3),
                                    activation='relu',
                                    input_shape=(block_num, 1, data_train.shape[3], data_train.shape[4])))
    conv_lstm.add(layers.Dropout(0.5))
    conv_lstm.add(layers.Flatten())
    conv_lstm.add(layers.Dense(64, activation='relu'))
    conv_lstm.add(layers.Dense(6, activation='softmax'))
    #cnn_1d.summary()
    conv_lstm.compile(loss='categorical_crossentropy',
                   optimizer='adam',
                   metrics=['acc'])

    history = conv_lstm.fit(data_train, label_train, epochs=25, batch_size=64, verbose=0, validation_split=0.2)
    _, te_acc = conv_lstm.evaluate(data_test, label_test, batch_size=32, verbose=0)

    return history, history.history['acc'][4], history.history['val_acc'][4],te_acc, history.history['loss'][4], history.history['val_loss'][4]


def evaluate_repreat(data_train, label_train, data_test, label_test, model, times):
    tr_scores = []
    val_scores = []
    te_scores = []
    tr_losses = []
    val_losses = []
    histories = []

    if model == 'cnn':
        print("CNN:")
        for n in range(times):
            print("Training %.d begin:" % (n + 1))
            history, tr_acc, val_acc, te_acc, tr_loss, val_loss = cnn_1d(data_train, label_train, data_test, label_test)
            histories.append(history)
            tr_scores.append(tr_acc)
            val_scores.append(val_acc)
            te_scores.append(te_acc)
            tr_losses.append(tr_loss)
            val_losses.append(val_loss)
            print("Tr_acc: {:.5%}".format(tr_acc))
            print("Val_acc: {:.5%}".format(val_acc))
            print("Te_acc: {:.5%}".format(te_acc))
            print("Tr_loss: {:.5%}".format(tr_loss))
            print("Val_loss: {:.5%}".format(val_loss))
            print("/////////////////////////////////////////////////////////////////")
    elif model == 'lstm':
        print("LSTM:")
        for n in range(times):
            print("Training %.d begin:" % (n + 1))
            history, tr_acc, val_acc, te_acc, tr_loss, val_loss = lstm(data_train, label_train, data_test, label_test)
            histories.append(history)
            tr_scores.append(tr_acc)
            val_scores.append(val_acc)
            te_scores.append(te_acc)
            tr_losses.append(tr_loss)
            val_losses.append(val_loss)
            print("Tr_acc: {:.5%}".format(tr_acc))
            print("Val_acc: {:.5%}".format(val_acc))
            print("Te_acc: {:.5%}".format(te_acc))
            print("Tr_loss: {:.5%}".format(tr_loss))
            print("Val_loss: {:.5%}".format(val_loss))
            print("/////////////////////////////////////////////////////////////////")
    elif model == 'cnn_lstm':
        print("CNN-LSTM")
        for n in range(times):
            print("Training %.d begin:" % (n + 1))
            history, tr_acc, val_acc, te_acc, tr_loss, val_loss = cnn_lstm(data_train, label_train, data_test, label_test, 4)
            histories.append(history)
            tr_scores.append(tr_acc)
            val_scores.append(val_acc)
            te_scores.append(te_acc)
            tr_losses.append(tr_loss)
            val_losses.append(val_loss)
            print("Tr_acc: {:.5%}".format(tr_acc))
            print("Val_acc: {:.5%}".format(val_acc))
            print("Te_acc: {:.5%}".format(te_acc))
            print("Tr_loss: {:.5%}".format(tr_loss))
            print("Val_loss: {:.5%}".format(val_loss))
            print("/////////////////////////////////////////////////////////////////")
    elif model == 'conv_lstm':
        print("ConvLSTM:")
        for n in range(times):
            print("Training %.d begin:" % (n + 1))
            history, tr_acc, val_acc, te_acc, tr_loss, val_loss = conv_lstm(data_train, label_train, data_test, label_test, 4)
            histories.append(history)
            tr_scores.append(tr_acc)
            val_scores.append(val_acc)
            te_scores.append(te_acc)
            tr_losses.append(tr_loss)
            val_losses.append(val_loss)
            print("Tr_acc: {:.5%}".format(tr_acc))
            print("Val_acc: {:.5%}".format(val_acc))
            print("Te_acc: {:.5%}".format(te_acc))
            print("Tr_loss: {:.5%}".format(tr_loss))
            print("Val_loss: {:.5%}".format(val_loss))
            print("/////////////////////////////////////////////////////////////////")

    print("Mean tr_score: {:.5}; Std:: +/- {:.5}".format(np.mean(tr_scores), np.std(tr_scores)))
    print("Mean val_score: {:.5}; Std:: +/- {:.5}".format(np.mean(val_scores), np.std(val_scores)))
    print("Mean te_score: {:.5}; Std:: +/- {:.5}".format(np.mean(te_scores), np.std(te_scores)))
    print("Mean tr_loss: {:.5}; Std:: +/- {:.5}".format(np.mean(tr_losses), np.std(tr_losses)))
    print("Mean val_loss: {:.5}; Std:: +/- {:.5}".format(np.mean(val_losses), np.std(val_losses)))
    return histories

