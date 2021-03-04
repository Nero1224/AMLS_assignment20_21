from keras import models, layers, optimizers
import tensorflow as tf
import numpy as np
import  matplotlib.pyplot as plt
from keras.callbacks.callbacks import EarlyStopping
import pandas as pd

# gpu memory management
# GPU memory management
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
"""
model = models.Sequential()
model.add(layers.LSTM(2, input_shape=(2, 1)))
model.add(layers.Dense(2))
model.add(layers.Activation('sigmoid'))

model.compile(loss='mse',
              optimizer=optimizers.SGD(lr=0.4, momentum=0.3),
              metrics=['acc'])

history = model.fit(tr_X, tr_Y, epochs=10, batch_size=10, verbose=0)

loss, acc = model.evaluate(vali_X, vali_Y)

pre_Y = model.predict(te_X)
pre_Y = model.predict_classes(te_X)

for n in range(10):
    model.fit(tr_x, tr_y, epochs=1, batch_size=(10, 10, 3))
    model.reset_states()
"""
"""
s1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
s2 = np.array([0.2, 0.4, 0.6, 0.8, 0.8, 0.6, 0.4, 0.2, 0.0])
print(s1+0.1)
x = np.append(s1.reshape(len(s1), 1), s2.reshape(len(s2), 1), axis=1)
print(x)
x = x.reshape(1, 9, 2)
print(x.shape)
#s1 = s1.reshape()
"""
# get train data
def get_tr():
    tr_x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    tr_y = tr_x + 0.1
    tr_x = tr_x.reshape(len(tr_x), 1, 1)
    return tr_x, tr_y

# get validation data
def get_vali():
    vali_x = np.array([0.6, 0.7, 0.8, 0.9, 1.0])
    vali_y = vali_x + 0.1
    vali_x = vali_x.reshape(len(vali_x), 1, 1)
    return vali_x, vali_y

# define model
def get_model(tr_x, tr_y, vali_x, vali_y, mem_num):
    model = models.Sequential()
    model.add(layers.LSTM(mem_num, input_shape=(1, 1)))
    model.add(layers.Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='adam')

    early_stop = EarlyStopping(monitor="val_loss", min_delta=0, patience=2, mode="auto", baseline=0.1,
                               restore_best_weights=True)

    history = model.fit(tr_x, tr_y, epochs=1500,
                        validation_data=(vali_x, vali_y),
                        shuffle=False,
                        callbacks=[early_stop])
    tr_loss = history.history['loss'][500:]
    vali_loss = history.history['val_loss'][500:]

    return tr_loss, vali_loss

tr_x, tr_y = get_tr()
vali_x, vali_y = get_vali()
mems = [5, 10, 15]
tr_losses = []
vali_losses = []


for mem in mems:
    for n in range(5):
        tr_loss, vali_loss = get_model(tr_x, tr_y, vali_x, vali_y, mem)
        tr_losses.append(tr_loss)
        vali_losses.append(vali_loss)


epochs = range(1, len(tr_loss)+1)
fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(epochs, tr_loss, label='train loss')
ax.plot(epochs, vali_loss, label='validation loss')
ax.set_xlabel('epochs', fontsize=22)
ax.set_ylabel('losses', fontsize=22)
ax.set_title('loss v.s. epochs')
ax.legend(fontsize=22)
plt.show()