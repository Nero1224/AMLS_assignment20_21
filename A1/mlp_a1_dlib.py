# Gender Classification based on MLP
# import necessary API
import numpy as np
from keras.layers import Dense, Input, Flatten, Dropout
from keras.models import Model
from keras import optimizers
import tensorflow as tf
import matplotlib.pyplot as plt
import dlib_feature_extract_a1 as ex

# GPU memory management
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)


def get_tr_te_set(num_tr, num_te, num_vali, n):
    """
    This function will automatically load the images
    inside dataset with given train and test set number

    :param num_tr: number of train set
    :param num_te: number of test set
    :return: train set and test set
    """
    features, labels = ex.extract_features_labels(n)
    features = np.array(features)
    labels = np.array([labels, -(labels - 1)]).T

    features_tr = features[:num_tr]
    features_te = features[num_tr:(num_tr + num_te)]
    features_vali = features[(num_tr + num_te): (num_tr + num_te + num_vali)]
    labels_tr = labels[:num_tr]
    labels_te = labels[num_tr:(num_tr + num_te)]
    labels_vali = labels[(num_tr + num_te): (num_tr + num_te + num_vali)]

    return features_tr, features_te, features_vali, labels_tr, labels_te, labels_vali

features_tr, features_te, features_vali, labels_tr, labels_te, labels_vali = get_tr_te_set(2500, 200, 200, 1500)

#print(features_tr.shape)

# build MLP
inp = Input(shape=(68, 2))
x = Flatten()(inp)

x = Dense(2048, activation='sigmoid')(x)
x = Dense(2048, activation='sigmoid')(x)

x = Dense(2, activation='softmax')(x)

model = Model(inputs=inp, output=x)
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=1e-5),
              metrics=['acc'])

history = model.fit(features_tr, labels_tr, epochs=50, validation_split=0.2, verbose=1)

acc = history.history['acc']
acc_val = history.history['val_acc']
loss = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1, len(acc) + 1)

fig, ax = plt.subplots(1,1, figsize=(10,6))

ax.plot(epochs, acc, 'ro', label='Training acc')
ax.plot(epochs, acc_val, 'b', label='Validation acc')
ax.set_title('Accuracy with data augmentation', fontsize=22)
ax.set_xlabel(r'Epochs', fontsize=22)
ax.set_ylabel(r'Accuracy', fontsize=22)
ax.tick_params(labelsize=22)
ax.legend(fontsize=24)
plt.show()

figL, axL = plt.subplots(1,1, figsize=(10,6))

axL.plot(epochs, loss, 'ro', label='Training loss')
axL.plot(epochs, loss_val, 'b', label='Validation loss')
axL.set_title('Loss with data augmentation', fontsize=22)
axL.set_xlabel(r'Epochs', fontsize=22)
axL.set_ylabel(r'Loss', fontsize=22)
axL.tick_params(labelsize=22)
axL.legend(fontsize=22)
plt.show()
