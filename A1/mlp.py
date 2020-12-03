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

images = np.zeros(shape=(5000, 218, 178, 3))
labels = np.zeros(shape=(5000, 1))
test_size = 0.2

data = pandas.read_csv("../Datasets/celeba/labels.csv", header=None)
for n in range(5000):
    image = cv2.imread("../Datasets/celeba/img/%s.jpg" %n, cv2.IMREAD_COLOR)
    label = np.array(data.loc[n+1]).tolist()
    label = int(str(label[0]).split('\t')[-2])
    images[n] = image
    if label == -1: label = 0
    labels[n] = label

images, labels = shuffle(images, labels)

X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=test_size,
                                                    random_state=123,
                                                    stratify=labels)

X_train = np.array(X_train)
X_test = np.array(X_test)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

X_train = X_train / 255
X_test = X_test / 255
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)

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

model.fit(X_train, Y_train, batch_size=32, epochs=10, validation_split=0.2, verbose=0)
loss, accuracy = model.evaluate(X_test, Y_test)

print(loss)
print(accuracy)

