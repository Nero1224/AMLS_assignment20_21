# shape Classification based on CNN
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

# modify original dataset and prepare train, validation, test datasets
base_path = os.path.dirname(os.getcwd())

path_list = []

img_path = os.path.join(base_path, r"Datasets\cartoon_set\img")
img_path_train = os.path.join(base_path, r"Datasets\cartoon_set\b1_dataset\train")
img_path_vali = os.path.join(base_path, r"Datasets\cartoon_set\b1_dataset\validation")
img_path_test = os.path.join(base_path, r"Datasets\cartoon_set\b1_dataset\test")
img_path_shape0_train = os.path.join(base_path, img_path_train, "shape0_set")
path_list.append(img_path_shape0_train)
img_path_shape0_vali = os.path.join(base_path, img_path_vali, "shape0_set")
path_list.append(img_path_shape0_vali)
img_path_shape0_test = os.path.join(base_path, img_path_test, "shape0_set")
path_list.append(img_path_shape0_test)
img_path_shape1_train = os.path.join(base_path, img_path_train, "shape1_set")
path_list.append(img_path_shape1_train)
img_path_shape1_vali = os.path.join(base_path, img_path_vali, "shape1_set")
path_list.append(img_path_shape1_vali)
img_path_shape1_test = os.path.join(base_path, img_path_test, "shape1_set")
path_list.append(img_path_shape1_test)
img_path_shape2_train = os.path.join(base_path, img_path_train, "shape2_set")
path_list.append(img_path_shape2_train)
img_path_shape2_vali = os.path.join(base_path, img_path_vali, "shape2_set")
path_list.append(img_path_shape2_vali)
img_path_shape2_test = os.path.join(base_path, img_path_test, "shape2_set")
path_list.append(img_path_shape2_test)
img_path_shape3_train = os.path.join(base_path, img_path_train, "shape3_set")
path_list.append(img_path_shape3_train)
img_path_shape3_vali = os.path.join(base_path, img_path_vali, "shape3_set")
path_list.append(img_path_shape3_vali)
img_path_shape3_test = os.path.join(base_path, img_path_test, "shape3_set")
path_list.append(img_path_shape3_test)
img_path_shape4_train = os.path.join(base_path, img_path_train, "shape4_set")
path_list.append(img_path_shape4_train)
img_path_shape4_vali = os.path.join(base_path, img_path_vali, "shape4_set")
path_list.append(img_path_shape4_vali)
img_path_shape4_test = os.path.join(base_path, img_path_test, "shape4_set")
path_list.append(img_path_shape4_test)

for path in path_list:
    if os.path.exists(path): pass
    else: os.makedirs(path)

shape0_labels = []
shape1_labels = []
shape2_labels = []
shape3_labels = []
shape4_labels = []
shape0_labels_te = []
shape1_labels_te = []
shape2_labels_te = []
shape3_labels_te = []
shape4_labels_te = []

data_info = pandas.read_csv("../Datasets/cartoon_set/labels.csv", header=None)
data_info_te = pandas.read_csv("../Datasets/cartoon_set_test/labels.csv", header=None)
for n in range(10000):
    label = np.array(data_info.loc[n + 1]).tolist()
    shape_label = int(str(label[0]).split('\t')[-2])
    img_label = int(str(label[0]).split('\t')[0])
    if shape_label == 0:
        shape0_labels.append(img_label)
    elif shape_label == 1:
        shape1_labels.append(img_label)
    elif shape_label == 2:
        shape2_labels.append(img_label)
    elif shape_label == 3:
        shape3_labels.append(img_label)
    else:
        shape4_labels.append(img_label)

for n in range(2500):
    label_te = np.array(data_info_te.loc[n + 1]).tolist()
    shape_label_te = int(str(label_te[0]).split('\t')[-2])
    img_label_te = int(str(label_te[0]).split('\t')[0])
    if shape_label_te == 0:
        shape0_labels_te.append(img_label_te)
    elif shape_label_te == 1:
        shape1_labels_te.append(img_label_te)
    elif shape_label_te == 2:
        shape2_labels_te.append(img_label_te)
    elif shape_label_te == 3:
        shape3_labels_te.append(img_label_te)
    else:
        shape4_labels_te.append(img_label_te)
print(len(shape0_labels))
print(len(shape1_labels))
print(len(shape2_labels))
print(len(shape3_labels))
print(len(shape4_labels))
print(len(shape0_labels_te))
print(len(shape1_labels_te))
print(len(shape2_labels_te))
print(len(shape3_labels_te))
print(len(shape4_labels_te))

for shape0_label in shape0_labels[0:1500]:
    if os.path.exists(img_path_shape0_train+"\%s.jpg" %shape0_label): pass
    else: shutil.copyfile(img_path+"\%s.png" %shape0_label, img_path_shape0_train+"\%s.jpg" %shape0_label)
for shape0_label in shape0_labels[1500:2000]:
    if os.path.exists(img_path_shape0_vali+"\%s.jpg" %shape0_label): pass
    else: shutil.copyfile(img_path+"\%s.png" %shape0_label, img_path_shape0_vali+"\%s.jpg" %shape0_label)
for shape0_label in shape0_labels_te:
    if os.path.exists(img_path_shape0_test+"\%s.jpg" %shape0_label): pass
    else: shutil.copyfile(img_path+"\%s.png" %shape0_label, img_path_shape0_test+"\%s.jpg" %shape0_label)

for shape1_label in shape1_labels[0:1500]:
    if os.path.exists(img_path_shape1_train+"\%s.jpg" %shape1_label): pass
    else: shutil.copyfile(img_path+"\%s.png" %shape1_label, img_path_shape1_train+"\%s.jpg" %shape1_label)
for shape1_label in shape1_labels[1500:2000]:
    if os.path.exists(img_path_shape1_vali+"\%s.jpg" %shape1_label): pass
    else: shutil.copyfile(img_path+"\%s.png" %shape1_label, img_path_shape1_vali+"\%s.jpg" %shape1_label)
for shape1_label in shape1_labels_te:
    if os.path.exists(img_path_shape1_test+"\%s.jpg" %shape1_label): pass
    else: shutil.copyfile(img_path+"\%s.png" %shape1_label, img_path_shape1_test+"\%s.jpg" %shape1_label)

for shape2_label in shape2_labels[0:1500]:
    if os.path.exists(img_path_shape2_train+"\%s.jpg" %shape2_label): pass
    else: shutil.copyfile(img_path+"\%s.png" %shape2_label, img_path_shape2_train+"\%s.jpg" %shape2_label)
for shape2_label in shape2_labels[1500:2000]:
    if os.path.exists(img_path_shape2_vali+"\%s.jpg" %shape2_label): pass
    else: shutil.copyfile(img_path+"\%s.png" %shape2_label, img_path_shape2_vali+"\%s.jpg" %shape2_label)
for shape2_label in shape2_labels_te:
    if os.path.exists(img_path_shape2_test+"\%s.jpg" %shape2_label): pass
    else: shutil.copyfile(img_path+"\%s.png" %shape2_label, img_path_shape2_test+"\%s.jpg" %shape2_label)

for shape3_label in shape3_labels[0:1500]:
    if os.path.exists(img_path_shape3_train+"\%s.jpg" % shape3_label): pass
    else: shutil.copyfile(img_path + "\%s.png" % shape3_label, img_path_shape3_train+"\%s.jpg" %shape3_label)
for shape3_label in shape3_labels[1500:2000]:
    if os.path.exists(img_path_shape3_vali+"\%s.jpg" % shape3_label): pass
    else: shutil.copyfile(img_path+"\%s.png" % shape3_label, img_path_shape3_vali+"\%s.jpg" %shape3_label)
for shape3_label in shape3_labels_te:
    if os.path.exists(img_path_shape3_test + "\%s.jpg" % shape3_label): pass
    else: shutil.copyfile(img_path + "\%s.png" % shape3_label, img_path_shape3_test+"\%s.jpg" %shape3_label)

for shape4_label in shape4_labels[0:1500]:
    if os.path.exists(img_path_shape4_train+"\%s.jpg" % shape4_label): pass
    else: shutil.copyfile(img_path + "\%s.png" % shape4_label, img_path_shape4_train+"\%s.jpg" %shape4_label)
for shape4_label in shape4_labels[1500:2000]:
    if os.path.exists(img_path_shape4_vali + "\%s.jpg" % shape4_label): pass
    else: shutil.copyfile(img_path + "\%s.png" % shape4_label, img_path_shape4_vali+"\%s.jpg" %shape4_label)
for shape4_label in shape4_labels_te:
    if os.path.exists(img_path_shape4_test + "\%s.jpg" % shape4_label): pass
    else: shutil.copyfile(img_path + "\%s.png" % shape4_label, img_path_shape4_test+"\%s.jpg" %shape4_label)


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
                                               batch_size=25,
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
                              steps_per_epoch=100,
                              epochs=5,
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