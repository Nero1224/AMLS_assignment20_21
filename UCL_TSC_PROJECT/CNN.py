import pandas
import os
import numpy as np

# load nine axes sensor data (the shape for each train axis is (7352, 128) for test is (2947, 128))
# 128 is the number of time steps; 50Hz*2,56s(window width) = 128
def load_file(path):
    data = pandas.read_csv(path, delim_whitespace=True, header=None)
    return data.values


def build_dataset(dir_path, file, type):
    files_list = []
    paths_list = []
    data = []
    label = []

    for root, dirs, file_names in os.walk(dir_path + "/" + file):
        for file_name in file_names:
            files_list.append(file_name)
            paths_list.append(os.path.join(root, file_name))

    for file_path in paths_list:
        data.append(load_file(file_path))
    # np.stack() can combine array in the depth dimension. It is suitable for 3D array
    # for the array only has 1 or 2 dimension like (m,n) or (m,1), can transfer them into
    # (m,n,1) and (1,m,1), then conduct stack operation
    data = np.dstack(data)


    label = load_file(dir_path + '/y_' + type + '.txt')
    print(type + ": data shape: {}; label shape: {}".format(data.shape, label.shape))
    return data, label


dir_path_train = r"C:/Users/ASUS/Desktop/AMLS_Assignment/AMLS_PROJECT/UCI_HAR_Dataset/train"
dir_path_test = r"C:/Users/ASUS/Desktop/AMLS_Assignment/AMLS_PROJECT/UCI_HAR_Dataset/test"


data_train, label_train = build_dataset(dir_path_train, 'Inertial Signals', 'train')
data_test, label_test = build_dataset(dir_path_test, 'Inertial Signals', 'test')

def distribution_ana(data):
    data = pandas.DataFrame(data)
    print(data.groupby(0).size())
    quantity = data.groupby(0).size().values
    print(quantity)

    for n in range(len(quantity)):
        print("For type " + "{}: {:.2%}".format(n, quantity[n]/sum(quantity)))

distribution_ana(label_train)

