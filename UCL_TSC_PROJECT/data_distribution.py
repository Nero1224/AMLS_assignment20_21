import Project_Lib as pl
import matplotlib.pyplot as plt
from DataPreparation import normalizer
import numpy as np

dir_path_train = r"C:/Users/ASUS/Desktop/AMLS_Assignment/AMLS_PROJECT/UCI_HAR_Dataset/train"
dir_path_test = r"C:/Users/ASUS/Desktop/AMLS_Assignment/AMLS_PROJECT/UCI_HAR_Dataset/test"

data_train, label_train = pl.build_dataset(dir_path_train, 'Inertial Signals', 'train')
data_test, label_test = pl.build_dataset(dir_path_test, 'Inertial Signals', 'test')

data_train = pl.overlapping_remove(data_train)
data_test = pl.overlapping_remove(data_test)

data_train = data_train.reshape(data_train.shape[0]*data_train.shape[1], data_train.shape[2])
data_test = data_test.reshape(data_test.shape[0]*data_test.shape[1], data_test.shape[2])

print(data_train.shape)
print(data_test.shape)

data_train.tolist()
data_test.tolist()

for n in range(9):
    data_train[:,n] = normalizer(data_train[:,n].reshape(-1,1))


feature_names = ['body_acc_x_train', 'body_acc_y_train', 'body_acc_z_train', 'body_gyro_x_train', 'body_gyro_y_train', 'body_gyro_z_train', 'total_acc_x_train', 'total_acc_y_train', 'total_acc_z_train']
fig, ax = plt.subplots(figsize=(15, 60), ncols=1, nrows=9)
n = 0

for n in range(9):
    print('plot begin')
    ax[n].set_xlim(-1,1)
    ax[n].hist(data_train[:,n], bins=200, histtype = 'bar')
    ax[n].set_title('{} distribution'.format(feature_names[n]), fontsize=22)
    ax[n].set_xlabel(r'Sensor Values', fontsize=22)
    ax[n].set_ylabel(r'{} data frequency'.format(feature_names[n]), fontsize=22)
    ax[n].tick_params(labelsize=22)
    ax[n].legend(fontsize=24)
    print('plot finished')

    n = n + 1

plt.show()
