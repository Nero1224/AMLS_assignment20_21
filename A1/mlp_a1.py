# Gender Classification based on MLP
# import necessary API
import numpy as np
from keras.layers import Dense, Input, Flatten
from keras.models import Model
from keras.utils import np_utils
from keras import optimizers
import tensorflow as tf
import matplotlib.pyplot as plt
import dlib_feature_extract_a1 as ex
import dlib_feature_extract_a1_test as ex_te

# GPU memory management
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)


def get_tr_te_set():
    """
    This function will extract dlib feature and labels for all CelebA images.
    It also provided prepared training, validation and test set with ratio 6:2:2
    :return:
    features_tr:    an array containing flatten 68 landmarks features for training
    features_vali:  an array containing flatten 68 landmarks features for validation
    features_te:    an array containing flatten 68 landmarks features for test
    labels_tr:      an list containing labels for training set
    labels_vali:    an list containing labels for validation set
    labels_te:      anlist containing labels for test set
    """
    print("Extraction begin")
    features, labels = ex.extract_features_labels()
    features_te, labels_te = ex_te.extract_features_labels()
    print("Extraction end")

    features = np.array(features)
    features_te = np.array(features_te)
    labels = np_utils.to_categorical(labels, 2)
    labels_te = np_utils.to_categorical(labels_te, 2)

    features_tr = features[:features_te.shape[0]*3]
    features_vali = features[features_te.shape[0]*3:features_te.shape[0]*4]
    labels_tr = labels[:features_te.shape[0]*3]
    labels_vali = labels[features_te.shape[0]*3:features_te.shape[0]*4]

    return features_tr, features_vali, features_te, labels_tr, labels_vali, labels_te


features_tr, features_vali, features_te, labels_tr, labels_vali, labels_te = get_tr_te_set()

# build ML
inp = Input(shape=(68,2))
x = Flatten()(inp)

x = Dense(2048, activation='sigmoid')(x)
x = Dense(2048, activation='sigmoid')(x)
"""
x = Dense(512, activation='relu')(x)
#x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
#x = Dropout(0.2)(x)
x = Dense(256, activation='relu')(x)
#x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
#x = Dropout(0.2)(x)
"""
x = Dense(2, activation='softmax')(x)

model = Model(inputs=inp, output=x)
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(lr=1e-5),
              metrics=['acc'])

history = model.fit(features_tr, labels_tr, epochs=50, validation_split=0.2, verbose=1)

print(model.evaluate(features_te, labels_te))

acc = history.history['acc']
acc_val = history.history['val_acc']
loss = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1, len(acc) + 1)

fig, ax = plt.subplots(1,1, figsize=(10,6))

ax.plot(epochs, acc, 'ro', label='Training acc')
ax.plot(epochs, acc_val, 'b', label='Validation acc')
ax.set_title('MLP Accuracy', fontsize=22)
ax.set_xlabel(r'Epochs', fontsize=22)
ax.set_ylabel(r'Accuracy', fontsize=22)
ax.tick_params(labelsize=22)
ax.legend(fontsize=24)
plt.show()

figL, axL = plt.subplots(1,1, figsize=(10,6))

axL.plot(epochs, loss, 'ro', label='Training loss')
axL.plot(epochs, loss_val, 'b', label='Validation loss')
axL.set_title('MLP Loss', fontsize=22)
axL.set_xlabel(r'Epochs', fontsize=22)
axL.set_ylabel(r'Loss', fontsize=22)
axL.tick_params(labelsize=22)
axL.legend(fontsize=22)
plt.show()
