# Gender Classification based on CNN
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
img_path_train = os.path.join(base_path, r"Datasets\celeba\a1_dataset\train")
img_path_vali = os.path.join(base_path, r"Datasets\celeba\a1_dataset\validation")
img_path_test = os.path.join(base_path, r"Datasets\celeba\a1_dataset\test")
img_path_female_train = os.path.join(base_path, img_path_train, "female_set")
path_list.append(img_path_female_train)
img_path_male_train = os.path.join(base_path, img_path_train, "male_set")
path_list.append(img_path_male_train)
img_path_female_vali = os.path.join(base_path, img_path_vali, "female_set")
path_list.append(img_path_female_vali)
img_path_male_vali = os.path.join(base_path, img_path_vali, "male_set")
path_list.append(img_path_male_vali)
img_path_female_test = os.path.join(base_path, img_path_test, "female_set")
path_list.append(img_path_female_test)
img_path_male_test = os.path.join(base_path, img_path_test, "male_set")
path_list.append(img_path_male_test)

for path in path_list:
    if os.path.exists(path): pass
    else: os.makedirs(path)

female_labels = []
male_labels = []
female_labels_te = []
male_labels_te = []

data_info = pandas.read_csv("../Datasets/celeba/labels.csv", header=None)
data_info_te = pandas.read_csv("../Datasets/celeba_test/labels.csv", header=None)

for n in range(5000):
    label = np.array(data_info.loc[n + 1]).tolist()
    gender_label = int(str(label[0]).split('\t')[-2])
    img_label = int(str(label[0]).split('\t')[0])
    if gender_label == -1:
        female_labels.append(img_label)
    else:
        male_labels.append(img_label)

for n in range(1000):
    label_te = np.array(data_info_te.loc[n + 1]).tolist()
    gender_label_te = int(str(label_te[0]).split('\t')[-2])
    img_label_te = int(str(label_te[0]).split('\t')[0])
    if gender_label_te == -1:
        female_labels_te.append(img_label_te)
    else:
        male_labels_te.append(img_label_te)

print(len(female_labels))
print(len(male_labels))
print(len(female_labels_te))
print(len(male_labels_te))

for female_label in female_labels[0:2000]:
    if os.path.exists(img_path_female_train+"\%s.jpg" %female_label): pass
    else: shutil.copy(img_path+"\%s.jpg" %female_label, img_path_female_train)
for female_label in female_labels[2000:2500]:
    if os.path.exists(img_path_female_vali+"\%s.jpg" %female_label): pass
    else: shutil.copy(img_path+"\%s.jpg" %female_label, img_path_female_vali)
for female_label in female_labels_te:
    if os.path.exists(img_path_female_test+"\%s.jpg" %female_label): pass
    else: shutil.copy(img_path+"\%s.jpg" %female_label, img_path_female_test)

for male_label in male_labels[0:2000]:
    if os.path.exists(img_path_male_train+"\%s.jpg" %male_label): pass
    else: shutil.copy(img_path+"\%s.jpg" %male_label, img_path_male_train)
for male_label in male_labels[2000:2500]:
    if os.path.exists(img_path_male_vali+"\%s.jpg" %male_label): pass
    else: shutil.copy(img_path+"\%s.jpg" %male_label, img_path_male_vali)
for male_label in male_labels_te:
    if os.path.exists(img_path_male_test+"\%s.jpg" %male_label): pass
    else: shutil.copy(img_path+"\%s.jpg" %male_label, img_path_male_test)

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

"""
from keras.preprocessing import image
train_data_gen = ImageDataGenerator(rescale=1./255,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest')
img = image.load_img(os.path.join(img_path, '0.jpg'))
img = image.img_to_array(img)
print(img.shape)
img = img.reshape((1,) + img.shape)
print(img.shape)
i = 0
for batch in train_data_gen.flow(img, batch_size=1):
    plt.figure(i)
    img_plot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i==8:
        break
plt.show()
"""