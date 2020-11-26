# Gender Classification based on CNN
# import necessary API
import os
import shutil
import numpy as np
from pandas import *
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# modify original dataset and prepare train, validation, test datasets
base_path = os.path.dirname(os.getcwd())

img_path = os.path.join(base_path, r"Datasets\celeba\img")
img_path_train = os.path.join(base_path, r"Datasets\celeba\a1_dataset\train")
img_path_vali = os.path.join(base_path, r"Datasets\celeba\a1_dataset\validation")
img_path_test = os.path.join(base_path, r"Datasets\celeba\a1_dataset\test")
img_path_female_train = os.path.join(base_path, img_path_train, "female_set")
img_path_male_train = os.path.join(base_path, img_path_train, "male_set")
img_path_female_vali = os.path.join(base_path, img_path_vali, "female_set")
img_path_male_vali = os.path.join(base_path, img_path_vali, "male_set")
img_path_female_test = os.path.join(base_path, img_path_test, "female_set")
img_path_male_test = os.path.join(base_path, img_path_test, "male_set")

if os.path.exists(img_path_female_train): pass
else: os.makedirs(img_path_female_train)
if os.path.exists(img_path_male_train): pass
else: os.makedirs(img_path_male_train)

if os.path.exists(img_path_female_vali): pass
else: os.makedirs(img_path_female_vali)
if os.path.exists(img_path_male_vali): pass
else: os.makedirs(img_path_male_vali)

if os.path.exists(img_path_female_test): pass
else: os.makedirs(img_path_female_test)
if os.path.exists(img_path_male_test): pass
else: os.makedirs(img_path_male_test)

female_labels = []
male_labels = []

data_info = pandas.read_csv("../Datasets/celeba/labels.csv", header=None)
for n in range(5000):
    label = np.array(data_info.loc[n + 1]).tolist()
    gender_label = int(str(label[0]).split('\t')[-2])
    img_label = int(str(label[0]).split('\t')[0])
    if gender_label == -1:
        female_labels.append(img_label)
    else:
        male_labels.append(img_label)

for female_label in female_labels[0:1250]:
    if os.path.exists(img_path_female_train+"\%s.jpg" %female_label): pass
    else: shutil.copy(img_path+"\%s.jpg" %female_label, img_path_female_train)

for male_label in male_labels[0:1250]:
    if os.path.exists(img_path_male_train+"\%s.jpg" %male_label): pass
    else: shutil.copy(img_path+"\%s.jpg" %male_label, img_path_male_train)

for female_label in female_labels[1250:1875]:
    if os.path.exists(img_path_female_vali+"\%s.jpg" %female_label): pass
    else: shutil.copy(img_path+"\%s.jpg" %female_label, img_path_female_vali)

for male_label in male_labels[1250:1875]:
    if os.path.exists(img_path_male_vali+"\%s.jpg" %male_label): pass
    else: shutil.copy(img_path+"\%s.jpg" %male_label, img_path_male_vali)

for female_label in female_labels[1875:2500]:
    if os.path.exists(img_path_female_test+"\%s.jpg" %female_label): pass
    else: shutil.copy(img_path+"\%s.jpg" %female_label, img_path_female_test)

for male_label in male_labels[1875:2500]:
    if os.path.exists(img_path_male_test+"\%s.jpg" %male_label): pass
    else: shutil.copy(img_path+"\%s.jpg" %male_label, img_path_male_test)
"""
print(len(os.listdir(img_path_female_train)))
print(len(os.listdir(img_path_female_vali)))
print(len(os.listdir(img_path_female_test)))
print(len(os.listdir(img_path_male_train)))
print(len(os.listdir(img_path_male_vali)))
print(len(os.listdir(img_path_male_test)))
"""

# build CNN network
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPool2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPool2D(2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPool2D(2, 2))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

# input data processing
train_data_gen = ImageDataGenerator(rescale=1./255)
test_data_gen = ImageDataGenerator(rescale=1./255)

train_gen = train_data_gen.flow_from_directory(img_path_train,
                                               target_size=(150, 150),
                                               batch_size=25,
                                               class_mode='binary')
vali_gen = train_data_gen.flow_from_directory(img_path_test,
                                               target_size=(150, 150),
                                               batch_size=25,
                                               class_mode='binary')

for data_batch, labels_batch in train_gen:
    print("Data batch size:", data_batch.shape)
    print("Labels batch size:", labels_batch.shape)
    break

history = model.fit_generator(train_gen,
                              steps_per_epoch=100,
                              epochs=30,
                              validation_data=vali_gen,
                              validation_steps=50)

model.save("a1_gender_v1.h5")

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

fig, ax = plt.subplots(1,1, figsize=(10,6))

ax.plot(epochs, acc, 'ro', label='Training acc')
ax.plot(epochs, val_acc, 'b', label='Validation acc')
ax.set_title('Accuracy with data augmentation', fontsize=22)
ax.set_xlabel(r'Epochs', fontsize=22)
ax.set_ylabel(r'Accuracy', fontsize=22)
ax.tick_params(labelsize=22)
ax.legend(fontsize=24)
plt.show()
figL, axL = plt.subplots(1,1, figsize=(10,6))

axL.plot(epochs, loss, 'ro', label='Training loss')
axL.plot(epochs, val_loss, 'b', label='Validation loss')
axL.set_title('Loss with data augmentation', fontsize=22)
axL.set_xlabel(r'Epochs', fontsize=22)
axL.set_ylabel(r'Loss', fontsize=22)
axL.tick_params(labelsize=22)
axL.legend(fontsize=22)
plt.show()