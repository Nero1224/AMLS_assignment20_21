import Project_Lib as pl

dir_path_train = r"C:/Users/ASUS/Desktop/AMLS_Assignment/AMLS_PROJECT/UCI_HAR_Dataset/train/subject_train.txt"
#dir_path_train = r"C:/Users/ASUS/Desktop/AMLS_Assignment/AMLS_PROJECT/UCI_HAR_Dataset/train/Inertial Signals/body_acc_x_train.txt"
dir_path_test = r"C:/Users/ASUS/Desktop/AMLS_Assignment/AMLS_PROJECT/UCI_HAR_Dataset/test/subject_test.txt"

train_set = pl.load_file(dir_path_train)
test_set = pl.load_file(dir_path_test)

print("The shape of train subject set is {}; The shape of test subject set is {}".format(train_set.shape, test_set.shape))
#print(train_set[1,:])