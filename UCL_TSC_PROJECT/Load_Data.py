import Project_Lib as pl
import numpy as np
import matplotlib.pyplot as plt

dir_path_train = r"C:/Users/ASUS/Desktop/AMLS_Assignment/AMLS_PROJECT/UCI_HAR_Dataset/train/subject_train.txt"
#dir_path_train = r"C:/Users/ASUS/Desktop/AMLS_Assignment/AMLS_PROJECT/UCI_HAR_Dataset/train/Inertial Signals/body_acc_x_train.txt"
dir_path_test = r"C:/Users/ASUS/Desktop/AMLS_Assignment/AMLS_PROJECT/UCI_HAR_Dataset/test/subject_test.txt"

train_set = pl.load_file(dir_path_train)
test_set = pl.load_file(dir_path_test)

print("The shape of train subject set is {}; The shape of test subject set is {}".format(train_set.shape, test_set.shape))
#print(train_set[1,:])

train_set_1 = []

for n in range(train_set.shape[0]):
    if train_set[n] == 1: train_set_1.append(n)
print("Volunteer 1 collection finished")

print("Vlounteer set size: {}; and content: {}".format(np.array(train_set_1).shape, train_set_1))

feature_names = ['body_acc_x_train', 'body_acc_y_train', 'body_acc_z_train', 'body_gyro_x_train', 'body_gyro_y_train', 'body_gyro_z_train', 'total_acc_x_train', 'total_acc_y_train', 'total_acc_z_train']
fig, ax = plt.subplots(figsize=(15, 60), ncols=3, nrows=3)
n = 0

for feature_name in feature_names:
    dir_path_train = r"C:/Users/ASUS/Desktop/AMLS_Assignment/AMLS_PROJECT/UCI_HAR_Dataset/train/Inertial Signals/{}.txt".format(feature_name)
    train_set = pl.load_file(dir_path_train)

    features = train_set[train_set_1[0], 0:int(train_set.shape[1]/2)]
    del train_set_1[0]
    for set in train_set_1:
        features = np.hstack((features, train_set[set, 0:int(train_set.shape[1]/2)]))

    print('plot begin')
    time_steps = range(features.shape[0])
    ax[n].plot(time_steps, features, label='{}'.format(feature_name))
    ax[n].set_title('{}'.format(feature_name), fontsize=22)
    ax[n].set_xlabel(r'Time steps', fontsize=22)
    ax[n].set_ylabel(r'{}'.format(feature_name), fontsize=22)
    ax[n].tick_params(labelsize=22)
    ax[n].legend(fontsize=24)
    print('plot finished')

    n = n + 1

plt.show()