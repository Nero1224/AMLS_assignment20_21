import model_a1 as a1
import model_a2 as a2
import model_b1 as b1
import model_b2 as b2
import joblib
from sklearn.metrics import accuracy_score


# ======================================================================================================================
# Data preprocessing
# Because cross-validation was used in task A1, A2, B1, no specific validation was given
# Warning: the preprocessing for A1, A2, B1 is dlib extraction. It may takes long time (30 minutes for b1).
print("Task a1 preprocessing begin.")
data_train_a1, data_test_a1, label_train_a1, label_test_a1 = a1.get_tr_te_set()
print("Task a1 preprocessing end.")
print("Task a2 preprocessing begin.")
data_train_a2, data_test_a2, label_train_a2, label_test_a2 = a2.get_tr_te_set()
print("Task a2 preprocessing end.")
print("Task b1 preprocessing begin.")
data_train_b1, data_test_b1, label_train_b1, label_test_b1 = b1.get_tr_te_set()
print("Task b1 preprocessing end.")
print("Task b2 preprocessing begin.")
data_train_b2, data_vali_b2, data_test_b2 = b2.get_tr_vali_te_set()
print("Task b2 preprocessing end.")

# ======================================================================================================================
# Task A1 A2 B1 B2
# Reading the pre-trained models for each task
# If want to re-train model, just uncomment training function below and run, it should return best training size,
# best cross-validation score, corresponding training accruacy, final test accuracy and the learning curve
# ======================================================================================================================
# A1
# Below code is for loading A1 model.

print("=====================================================================================================")
print("A1 load begin")
model_A1 = joblib.load('./A1/model_a1.pkl')
print("A1 load end")
print("A1 accuracy generation begin")
acc_A1_train = accuracy_score(label_train_a1, model_A1.predict(data_train_a1))
acc_A1_test = accuracy_score(label_test_a1, model_A1.predict(data_test_a1))
print("A1:{},{}".format(acc_A1_train, acc_A1_test))
print("A1 accuracy generation end")

# Below code is for re-training A1 model.
"""
print("A1 re-train begin")
acc_A1_train, acc_A1_test = a1.rf_training(data_train_a1, data_test_a1, label_train_a1, label_test_a1)
print("A1:{},{}".format(acc_A1_train, acc_A1_test))
print("A1 re-train end")
"""
# ======================================================================================================================
# A2
# Below code is for loading A2 model.

print("=====================================================================================================")
print("A2 load begin")
model_A2 = joblib.load('./A2/model_a2.pkl')
print("A2 load end")
print("A2 accuracy generation begin")
acc_A2_train = accuracy_score(label_train_a2, model_A2.predict(data_train_a2))
acc_A2_test = accuracy_score(label_test_a2, model_A2.predict(data_test_a2))
print("A2:{},{}".format(acc_A2_train, acc_A2_test))
print("A2 accuracy generation end")

# Below code is for re-training A2 model.
"""
print("=====================================================================================================")
print("A2 re-train begin")
acc_A2_train, acc_A2_test = a2.ada_training(data_train_a2, data_test_a2, label_train_a2, label_test_a2)
print("A2:{},{}".format(acc_A2_train, acc_A2_test))
print("A2 re-train end")
"""
# ======================================================================================================================
# B1
# Below code is for loading B1 model.

print("=====================================================================================================")
print("B1 load begin")
model_B1 = joblib.load('./B1/model_b1.pkl')
print("B1 load end")
print("B1 accuracy generation begin")
acc_B1_train = accuracy_score(label_train_b1, model_B1.predict(data_train_b1))
acc_B1_test = accuracy_score(label_test_b1, model_B1.predict(data_test_b1))
print("B1:{},{}".format(acc_B1_train, acc_B1_test))
print("B1 accuracy generation end")

# Below code is for re-training B1 model.
"""
print("=====================================================================================================")
print("B1 re-train begin")
acc_B1_train, acc_B1_test = b1.svm_training(data_train_b1, data_test_b1, label_train_b1, label_test_b1)
print("B1:{},{}".format(acc_B1_train, acc_B1_test))
print("B1 re-train end")
"""
# ======================================================================================================================
# B2
# Below code is for loading B2 model.

print("=====================================================================================================")
print("B2 begins")
model_B2 = joblib.load('./B2/model_b2.pkl')
print("B2 load end")
print("B2 accuracy generation begin")
acc_B2_train = model_B2.evaluate_generator(data_train_b2)[-1]
acc_B2_test = model_B2.evaluate_generator(data_test_b2)[-1]
print("B2:{},{}".format(acc_B2_train, acc_B2_test))
print("B2 accuracy generation end")

# Below code is for re-training B2 model.
"""
print("=====================================================================================================")
print("B2 re-train begin")
acc_B2_train, acc_B2_test = b2.cnn_training(data_train_b2, data_vali_b2, data_test_b2)
print("B2:{},{}".format(acc_B2_train, acc_B2_test))
print("B2 re-train end")
"""
# ======================================================================================================================
# Training accuracy and test accuracy for each task
print("=====================================================================================================")
print("Conclusion:")
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test))

