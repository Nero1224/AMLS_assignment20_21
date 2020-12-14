# color Classification based on CNN
# import necessary API
import os
import shutil
from pandas import *
import numpy as np
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf

# GPU memory management
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

# modify original dataset and prepare train, validation, test datasets
base_path = os.path.dirname(os.getcwd())

path_list = []

img_path = os.path.join(base_path, r"Datasets\cartoon_set\img")
img_path_add = os.path.join(base_path, r"Datasets\cartoon_set_test\img")
img_path_train = os.path.join(base_path, r"Datasets\cartoon_set\b2_dataset\train")
img_path_vali = os.path.join(base_path, r"Datasets\cartoon_set\b2_dataset\validation")
img_path_test = os.path.join(base_path, r"Datasets\cartoon_set\b2_dataset\test")
img_path_color0_train = os.path.join(base_path, img_path_train, "color0_set")
path_list.append(img_path_color0_train)
img_path_color0_vali = os.path.join(base_path, img_path_vali, "color0_set")
path_list.append(img_path_color0_vali)
img_path_color0_test = os.path.join(base_path, img_path_test, "color0_set")
path_list.append(img_path_color0_test)
img_path_color1_train = os.path.join(base_path, img_path_train, "color1_set")
path_list.append(img_path_color1_train)
img_path_color1_vali = os.path.join(base_path, img_path_vali, "color1_set")
path_list.append(img_path_color1_vali)
img_path_color1_test = os.path.join(base_path, img_path_test, "color1_set")
path_list.append(img_path_color1_test)
img_path_color2_train = os.path.join(base_path, img_path_train, "color2_set")
path_list.append(img_path_color2_train)
img_path_color2_vali = os.path.join(base_path, img_path_vali, "color2_set")
path_list.append(img_path_color2_vali)
img_path_color2_test = os.path.join(base_path, img_path_test, "color2_set")
path_list.append(img_path_color2_test)
img_path_color3_train = os.path.join(base_path, img_path_train, "color3_set")
path_list.append(img_path_color3_train)
img_path_color3_vali = os.path.join(base_path, img_path_vali, "color3_set")
path_list.append(img_path_color3_vali)
img_path_color3_test = os.path.join(base_path, img_path_test, "color3_set")
path_list.append(img_path_color3_test)
img_path_color4_train = os.path.join(base_path, img_path_train, "color4_set")
path_list.append(img_path_color4_train)
img_path_color4_vali = os.path.join(base_path, img_path_vali, "color4_set")
path_list.append(img_path_color4_vali)
img_path_color4_test = os.path.join(base_path, img_path_test, "color4_set")
path_list.append(img_path_color4_test)

for path in path_list:
    if os.path.exists(path): pass
    else: os.makedirs(path)

color_labels = []
color0_labels = []
color1_labels = []
color2_labels = []
color3_labels = []
color4_labels = []

color_labels_te = []
color0_labels_te = []
color1_labels_te = []
color2_labels_te = []
color3_labels_te = []
color4_labels_te = []

data_info = pandas.read_csv("../Datasets/cartoon_set/labels.csv", header=None)
data_info_te = pandas.read_csv("../Datasets/cartoon_set_test/labels.csv", header=None)

for n in range(10000):
    label = np.array(data_info.loc[n + 1]).tolist()
    color_label = int(str(label[0]).split('\t')[-3])

    img_label = int(str(label[0]).split('\t')[0])
    if color_label == 0:
        color0_labels.append(img_label)
    elif color_label == 1:
        color1_labels.append(img_label)
    elif color_label == 2:
        color2_labels.append(img_label)
    elif color_label == 3:
        color3_labels.append(img_label)
    elif color_label == 4:
        color4_labels.append(img_label)

for n in range(2500):
    label_te = np.array(data_info_te.loc[n + 1]).tolist()
    color_label_te = int(str(label_te[0]).split('\t')[-3])

    img_label_te = int(str(label_te[0]).split('\t')[0])
    if color_label_te == 0:
        color0_labels_te.append(img_label_te)
    elif color_label_te == 1:
        color1_labels_te.append(img_label_te)
    elif color_label_te == 2:
        color2_labels_te.append(img_label_te)
    elif color_label_te == 3:
        color3_labels_te.append(img_label_te)
    elif color_label_te == 4:
        color4_labels_te.append(img_label_te)
print(len(color0_labels))
print(len(color1_labels))
print(len(color2_labels))
print(len(color3_labels))
print(len(color4_labels))
print(len(color0_labels_te))
print(len(color1_labels_te))
print(len(color2_labels_te))
print(len(color3_labels_te))
print(len(color4_labels_te))

for color0_label in color0_labels[0:1500]:
    if os.path.exists(img_path_color0_train+"\%s.jpg" %color0_label): pass
    else: shutil.copyfile(img_path+"\%s.png" %color0_label, img_path_color0_train+"\%s.jpg" %color0_label)
for color0_label in color0_labels[1500:2004]:
    if os.path.exists(img_path_color0_vali+"\%s.jpg" %color0_label): pass
    else: shutil.copyfile(img_path+"\%s.png" %color0_label, img_path_color0_vali+"\%s.jpg" %color0_label)
for color0_label_te in color0_labels_te:
    if os.path.exists(img_path_color0_test+"\%s.jpg" %color0_labels_te): pass
    else: shutil.copyfile(img_path_add+"\%s.png" %color0_label_te, img_path_color0_test+"\%s.jpg" %color0_label_te)

for color1_label in color1_labels[0:1500]:
    if os.path.exists(img_path_color1_train+"\%s.jpg" %color1_label): pass
    else: shutil.copyfile(img_path+"\%s.png" %color1_label, img_path_color1_train+"\%s.jpg" %color1_label)
for color1_label in color1_labels[1500:2018]:
    if os.path.exists(img_path_color1_vali+"\%s.jpg" %color1_label): pass
    else: shutil.copyfile(img_path+"\%s.png" %color1_label, img_path_color1_vali+"\%s.jpg" %color1_label)
for color1_label_te in color1_labels_te:
    if os.path.exists(img_path_color1_test+"\%s.jpg" %color1_labels_te): pass
    else: shutil.copyfile(img_path_add+"\%s.png" %color1_label_te, img_path_color1_test+"\%s.jpg" %color1_label_te)

for color2_label in color2_labels[0:1500]:
    if os.path.exists(img_path_color2_train+"\%s.jpg" %color2_label): pass
    else: shutil.copyfile(img_path+"\%s.png" %color2_label, img_path_color2_train+"\%s.jpg" %color2_label)
for color2_label in color2_labels[1500:1969]:
    if os.path.exists(img_path_color2_vali+"\%s.jpg" %color2_label): pass
    else: shutil.copyfile(img_path+"\%s.png" %color2_label, img_path_color2_vali+"\%s.jpg" %color2_label)
for color2_label_te in color2_labels_te:
    if os.path.exists(img_path_color2_test+"\%s.jpg" %color2_labels_te): pass
    else: shutil.copyfile(img_path_add+"\%s.png" %color2_label_te, img_path_color2_test+"\%s.jpg" %color2_label_te)

for color3_label in color3_labels[0:1500]:
    if os.path.exists(img_path_color3_train+"\%s.jpg" % color3_label): pass
    else: shutil.copyfile(img_path + "\%s.png" % color3_label, img_path_color3_train+"\%s.jpg" %color3_label)
for color3_label in color3_labels[1500:1992]:
    if os.path.exists(img_path_color3_vali+"\%s.jpg" % color3_label): pass
    else: shutil.copyfile(img_path+"\%s.png" % color3_label, img_path_color3_vali+"\%s.jpg" %color3_label)
for color3_label_te in color3_labels_te:
    if os.path.exists(img_path_color3_test+"\%s.jpg" %color3_labels_te): pass
    else: shutil.copyfile(img_path_add+"\%s.png" %color3_label_te, img_path_color3_test+"\%s.jpg" %color3_label_te)

for color4_label in color4_labels[0:1500]:
    if os.path.exists(img_path_color4_train+"\%s.jpg" % color4_label): pass
    else: shutil.copyfile(img_path + "\%s.png" % color4_label, img_path_color4_train+"\%s.jpg" %color4_label)
for color4_label in color4_labels[1500:2017]:
    if os.path.exists(img_path_color4_vali + "\%s.jpg" % color4_label): pass
    else: shutil.copyfile(img_path + "\%s.png" % color4_label, img_path_color4_vali+"\%s.jpg" %color4_label)
for color4_label_te in color4_labels_te:
    if os.path.exists(img_path_color4_test+"\%s.jpg" %color4_labels_te): pass
    else: shutil.copyfile(img_path_add+"\%s.png" %color4_label_te, img_path_color4_test+"\%s.jpg" %color4_label_te)


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
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
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
                                               target_size=(150, 150),
                                               batch_size=50,
                                               class_mode='categorical')
vali_gen = vali_data_gen.flow_from_directory(img_path_vali,
                                               target_size=(150, 150),
                                               batch_size=50,
                                               class_mode='categorical')
test_gen = test_data_gen.flow_from_directory(img_path_test,
                                               target_size=(150, 150),
                                               batch_size=50,
                                               class_mode='categorical')

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
                              steps_per_epoch=150,
                              epochs=10,
                              validation_data=vali_gen,
                              validation_steps=50)

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
