import Project_Lib as pl
import tensorflow as tf
import matplotlib.pyplot as plt

# GPU memory management
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

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



histories_cnn = pl.evaluate_repreat(data_train, label_train, data_test, label_test, 'cnn', 10)
#histories_lstm = pl.evaluate_repreat(data_train, label_train, data_test, label_test, 'lstm', 5)
#histories_cnn_lstm = pl.evaluate_repreat(data_train, label_train, data_test, label_test, 'cnn_lstm', 5)
#histories_convlstm = pl.evaluate_repreat(data_train, label_train, data_test, label_test, 'conv_lstm', 5)

data_train_std, data_test_std = pl.standardization(data_train, data_test)
histories_cnn = pl.evaluate_repreat(data_train, label_train, data_test, label_test, 'cnn', 10)

"""
fig, ax = plt.subplots(figsize=(15, 50), ncols=1, nrows=4)
histories = [histories_cnn[4], histories_lstm[4], histories_cnn_lstm[4], histories_convlstm[4]]
histories_name = ['cnn', 'lstm', 'cnn-lstm', 'convlstm']
#histories = []
#histories.append(histories_cnn[0])
#histories.append('4')

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
"""