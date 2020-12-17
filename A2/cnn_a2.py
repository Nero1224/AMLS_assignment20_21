# emotion Classification based on CNN
# import necessary API
import os
import shutil
from pandas import *
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# GPU memory management
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

# modify original dataset and prepare train, validation, test datasets
base_path = os.path.dirname(os.getcwd())

path_list = []

img_path = os.path.join(base_path, r"Datasets\celeba\img")
img_path_train = os.path.join(base_path, r"Datasets\celeba\a2_dataset\train")
img_path_vali = os.path.join(base_path, r"Datasets\celeba\a2_dataset\validation")
img_path_test = os.path.join(base_path, r"Datasets\celeba\a2_dataset\test")
img_path_smiling_train = os.path.join(base_path, img_path_train, "smiling_set")
path_list.append(img_path_smiling_train)
img_path_none_train = os.path.join(base_path, img_path_train, "none_set")
path_list.append(img_path_none_train)
img_path_smiling_vali = os.path.join(base_path, img_path_vali, "smiling_set")
path_list.append(img_path_smiling_vali)
img_path_none_vali = os.path.join(base_path, img_path_vali, "none_set")
path_list.append(img_path_none_vali)
img_path_smiling_test = os.path.join(base_path, img_path_test, "smiling_set")
path_list.append(img_path_smiling_test)
img_path_none_test = os.path.join(base_path, img_path_test, "none_set")
path_list.append(img_path_none_test)

for path in path_list:
    if os.path.exists(path): pass
    else: os.makedirs(path)

smiling_labels = []
none_labels = []
smiling_labels_te = []
none_labels_te = []

data_info = pandas.read_csv("../Datasets/celeba/labels.csv", header=None)
data_info_te = pandas.read_csv("../Datasets/celeba_test/labels.csv", header=None)
for n in range(5000):
    label = np.array(data_info.loc[n + 1]).tolist()
    emotion_label = int(str(label[0]).split('\t')[-1])
    img_label = int(str(label[0]).split('\t')[0])
    if emotion_label == 1:
        smiling_labels.append(img_label)
    else:
        none_labels.append(img_label)

for n in range(1000):
    label_te = np.array(data_info_te.loc[n + 1]).tolist()
    emotion_label_te = int(str(label_te[0]).split('\t')[-1])
    img_label_te = int(str(label_te[0]).split('\t')[0])
    if emotion_label_te == 1:
        smiling_labels_te.append(img_label_te)
    else:
        none_labels_te.append(img_label_te)

print(len(smiling_labels))
print(len(none_labels))
print(len(smiling_labels_te))
print(len(none_labels_te))

for smiling_label in smiling_labels[0:2000]:
    if os.path.exists(img_path_smiling_train+"\%s.jpg" %smiling_label): pass
    else: shutil.copy(img_path+"\%s.jpg" %smiling_label, img_path_smiling_train)
for smiling_label in smiling_labels[2000:2500]:
    if os.path.exists(img_path_smiling_vali+"\%s.jpg" %smiling_label): pass
    else: shutil.copy(img_path+"\%s.jpg" %smiling_label, img_path_smiling_vali)
for smiling_label in smiling_labels_te:
    if os.path.exists(img_path_smiling_test+"\%s.jpg" %smiling_label): pass
    else: shutil.copy(img_path+"\%s.jpg" %smiling_label, img_path_smiling_test)

for none_label in none_labels[0:2000]:
    if os.path.exists(img_path_none_train+"\%s.jpg" %none_label): pass
    else: shutil.copy(img_path+"\%s.jpg" %none_label, img_path_none_train)
for none_label in none_labels[2000:2500]:
    if os.path.exists(img_path_none_vali+"\%s.jpg" %none_label): pass
    else: shutil.copy(img_path+"\%s.jpg" %none_label, img_path_none_vali)
for none_label in none_labels_te:
    if os.path.exists(img_path_none_test+"\%s.jpg" %none_label): pass
    else: shutil.copy(img_path+"\%s.jpg" %none_label, img_path_none_test)

# build CNN network
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(218, 178, 3)))
model.add(layers.MaxPool2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPool2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPool2D(2, 2))
model.add(layers.Flatten())
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# input data processing
"""
train_data_gen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)
"""
train_data_gen = ImageDataGenerator(rescale=1./255)
vali_data_gen = ImageDataGenerator(rescale=1./255)
test_data_gen = ImageDataGenerator(rescale=1./255)

train_gen = train_data_gen.flow_from_directory(img_path_train,
                                               target_size=(218, 178),
                                               batch_size=25,
                                               class_mode='binary')
vali_gen = vali_data_gen.flow_from_directory(img_path_vali,
                                               target_size=(218, 178),
                                               batch_size=25,
                                               class_mode='binary')
test_gen = test_data_gen.flow_from_directory(img_path_vali,
                                               target_size=(218, 178),
                                               batch_size=25,
                                               class_mode='binary')


for data_batch, labels_batch in train_gen:
    print("Data batch size:", data_batch.shape)
    print("Labels batch size:", labels_batch.shape)
    break
for data_batch, labels_batch in vali_gen:
    print("Data batch size:", data_batch.shape)
    print("Labels batch size:", labels_batch.shape)
    break
for data_batch, labels_batch in test_gen:
    print("Data batch size:", data_batch.shape)
    print("Labels batch size:", labels_batch.shape)
    break

history = model.fit_generator(train_gen,
                              steps_per_epoch=160,
                              epochs=5,
                              validation_data=vali_gen,
                              validation_steps=40)

score = model.evaluate_generator(test_gen)
print(score)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

fig, ax = plt.subplots(1,1, figsize=(10,6))

ax.plot(epochs, acc, 'ro', label='Training acc')
ax.plot(epochs, val_acc, 'b', label='Validation acc')
ax.set_title('CNN performance for a1 (Accuracy)', fontsize=22)
ax.set_xlabel(r'Epochs', fontsize=22)
ax.set_ylabel(r'Accuracy', fontsize=22)
ax.tick_params(labelsize=22)
ax.legend(fontsize=24)
plt.show()
figL, axL = plt.subplots(1, 1, figsize=(10,6))

axL.plot(epochs, loss, 'ro', label='Training loss')
axL.plot(epochs, val_loss, 'b', label='Validation loss')
axL.set_title('CNN performance for a1 (Loss)', fontsize=22)
axL.set_xlabel(r'Epochs', fontsize=22)
axL.set_ylabel(r'Loss', fontsize=22)
axL.tick_params(labelsize=22)
axL.legend(fontsize=22)
plt.show()
