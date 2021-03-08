import Project_Lib as pl
import tensorflow as tf

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

#pl.evaluate_repreat(data_train, label_train, data_test, label_test, 'cnn', 10)
#pl.evaluate_repreat(data_train, label_train, data_test, label_test, 'lstm', 10)
#pl.evaluate_repreat(data_train, label_train, data_test, label_test, 'cnn_lstm', 10)
pl.evaluate_repreat(data_train, label_train, data_test, label_test, 'conv_lstm', 10)