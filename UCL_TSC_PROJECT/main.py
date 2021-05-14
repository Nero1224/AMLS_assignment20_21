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



#pl.evaluate_repreat(data_train, label_train, data_test, label_test, 'cnn', 5)
#pl.evaluate_repreat(data_train, label_train, data_test, label_test, 'lstm', 5)
#pl.evaluate_repreat(data_train, label_train, data_test, label_test, 'cnn_lstm', 5)
#pl.evaluate_repreat(data_train, label_train, data_test, label_test, 'conv_lstm', 5)