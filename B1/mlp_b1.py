# Gender Classification based on MLP
# import necessary API
import cv2
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.layers import Dense, Input, Flatten, Dropout
from keras.models import Model
from keras import optimizers
from keras.utils import np_utils
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import shutil

# GPU memory management
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

base_path = os.path.dirname(os.getcwd())
img_path = os.path.join(base_path, r"Datasets\cartoon_set\img")
jpg_path = os.path.join(base_path, r"Datasets\cartoon_set\img_jpg")
if os.path.exists(jpg_path): pass
else:
    os.makedirs(jpg_path)
    for img in os.listdir(img_path):
        shutil.copyfile(os.path.join(img_path, r"%s" %img), os.path.join(jpg_path, "%s.jpg" %img.split(".")[0]))

images = []
labels = []
test_size = 0.2

data = pandas.read_csv("../Datasets/cartoon_set/labels.csv", header=None)
for n in range(5000):
    image = cv2.imread("../Datasets/cartoon_set/img_jpg/%s.jpg" %n, cv2.IMREAD_COLOR)
    label = np.array(data.loc[n+1]).tolist()
    label = int(str(label[0]).split('\t')[-2])
    images.append(image)
    labels.append(label)

images = np.array(images).reshape(5000, 500, 500, 3) / 255
labels = np_utils.to_categorical(np.array(labels), 5)

images, labels = shuffle(images, labels)

X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=test_size,
                                                    random_state=123,
                                                    stratify=labels)
print(Y_test)
"""
X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

# input normalization and output categorical
X_train = X_train / 255
X_test = X_test / 255
Y_train = np_utils.to_categorical(Y_train, 5)
Y_test = np_utils.to_categorical(Y_test, 5)
print(Y_test)

# build MLP
inp = Input(shape=(218, 178, 3))
x = Flatten()(inp)

x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)

x = Dense(2, activation='sigmoid')(x)

model = Model(inputs=inp, output=x)

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

history = model.fit(X_train, Y_train, batch_size=32, epochs=10, validation_split=0.2, verbose=1)

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

figL, axL = plt.subplots(1,1, figsize=(10,6))

axL.plot(epochs, loss, 'ro', label='Training loss')
axL.plot(epochs, loss_val, 'b', label='Validation loss')
axL.set_title('Loss with data augmentation', fontsize=22)
axL.set_xlabel(r'Epochs', fontsize=22)
axL.set_ylabel(r'Loss', fontsize=22)
axL.tick_params(labelsize=22)
axL.legend(fontsize=22)
plt.show()
"""