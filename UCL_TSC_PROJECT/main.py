import Project_Lib as pl
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# GPU memory management
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

# Load and analyze train and test dataset
dir_path_train = r"C:/Users/ASUS/Desktop/AMLS_Assignment/AMLS_PROJECT/UCI_HAR_Dataset/train"
dir_path_test = r"C:/Users/ASUS/Desktop/AMLS_Assignment/AMLS_PROJECT/UCI_HAR_Dataset/test"
data_train, label_train = pl.build_dataset(dir_path_train, 'Inertial Signals', 'train')
data_test, label_test = pl.build_dataset(dir_path_test, 'Inertial Signals', 'test')
pl.distribution_ana(label_train)
pl.distribution_ana(label_test)
print(data_train.shape)
print(label_train.shape)
print(data_test.shape)
print(label_test.shape)

# plot one data
"""
print('plot begin')
time_steps= range(0, 128, 1)
fig, ax = plt.subplots(1,1, figsize=(10,6))
ax.plot(time_steps, data_train[1,:,1], label='body acc x')
ax.set_title('Body acceleration for one time window)', fontsize=22)
ax.set_xlabel(r'Time steps', fontsize=22)
ax.set_ylabel(r'Acceleration', fontsize=22)
ax.tick_params(labelsize=22)
ax.legend(fontsize=24)
plt.show()
print('plot finished')
"""

# manually shuffle prepared train set and test set

index_train = [i for i in range(len(data_train))]
index_test = [i for i in range(len(data_test))]
np.random.shuffle(index_train)
np.random.shuffle(index_test)
data_train = data_train[index_train, :, :]
data_test = data_test[index_test, :, :]
label_train = label_train[index_train]
label_test = label_test[index_test]


# manually create and shuffle train set and test set; note: the result is very bad! Why?
"""
data = np.concatenate((data_train, data_test), axis=0)
label = np.concatenate((label_train, label_test), axis=0)
print("New data shape: " + str(data.shape))
print("New label shape: " + str(label.shape))
index = [i for i in range(len(data))]
np.random.shuffle(index)
data_ = data[index, :, :]
label = label[index]
data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.2, random_state=1)
print("Balance after shuffle: ")
pl.distribution_ana(label_train)
pl.distribution_ana(label_test)
print(data_train.shape)
print(label_train.shape)
print(data_test.shape)
print(label_test.shape)
"""

# Train and test all models without standardization
"""
histories_cnn = pl.evaluate_repreat(data_train, label_train, data_test, label_test, 'cnn', 5)
histories_lstm = pl.evaluate_repreat(data_train, label_train, data_test, label_test, 'lstm', 5)
histories_cnn_lstm = pl.evaluate_repreat(data_train, label_train, data_test, label_test, 'cnn_lstm', 5)
histories_convlstm = pl.evaluate_repreat(data_train, label_train, data_test, label_test, 'conv_lstm', 5)
"""

# Train and test all models with standardization
#data_train_std, data_test_std = pl.standardization(data_train, data_test)
data_train_std = data_train
data_test_std = data_test
histories_cnn = pl.evaluate_repreat(data_train_std, label_train, data_test_std, label_test, 'cnn', 5)
histories_lstm = pl.evaluate_repreat(data_train_std, label_train, data_test_std, label_test, 'lstm', 5)
histories_cnn_lstm = pl.evaluate_repreat(data_train_std, label_train, data_test_std, label_test, 'cnn_lstm', 5)
histories_convlstm = pl.evaluate_repreat(data_train_std, label_train, data_test_std, label_test, 'conv_lstm', 5)

# Plot learning curve
fig, ax = plt.subplots(figsize=(15, 50), ncols=1, nrows=4)
histories = [histories_cnn[4], histories_lstm[4], histories_cnn_lstm[4], histories_convlstm[4]]
histories_name = ['cnn', 'lstm', 'cnn-lstm', 'convlstm']
#histories = [histories_cnn[4]]
#histories_name = ['cnn']

for n in range(4):
    print('plot begin')
    time_steps = range(1, len(histories[n].history['acc'])+1)
    ax[n].plot(time_steps, histories[n].history['acc'], 'ro', label=str(histories_name[n])+'acc')
    ax[n].plot(time_steps, histories[n].history['val_acc'], 'b', label=str(histories_name[n])+'val_acc')
    ax[n].set_title(str(histories_name[n]), fontsize=22)
    ax[n].set_xlabel(r'Epochs', fontsize=22)
    ax[n].set_ylabel(r'Accuracy', fontsize=22)
    ax[n].tick_params(labelsize=22)
    ax[n].legend(fontsize=24)
    print('plot finished')
plt.show()
