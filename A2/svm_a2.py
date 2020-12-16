# Gender Classification based on MLP
# import necessary API
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import dlib_feature_extract_a2 as ex
import dlib_feature_extract_a2_test as ex_te
import matplotlib.pyplot as plt
from keras.utils import np_utils
from sklearn.model_selection import GridSearchCV


def get_tr_te_set():
    """
    This function will automatically load the images
    inside dataset with given train and test set number

    :param num_tr: number of train set
    :param num_te: number of test set
    :return: train set and test set
    """
    print("Extraction begin")
    features, labels = ex.extract_features_labels()
    features_te, labels_te = ex_te.extract_features_labels()
    print("Extraction end")

    features = np.array(features)
    labels = np_utils.to_categorical(labels, 2)
    features_te = np.array(features_te)
    labels_te = np_utils.to_categorical(labels_te, 2)

    features_tr = features[:features_te.shape[0]*3]
    features_vali = features[features_te.shape[0]*3:features_te.shape[0]*4]
    labels_tr = labels[:features_te.shape[0]*3]
    labels_vali = labels[features_te.shape[0]*3:features_te.shape[0]*4]

    features_tr = features_tr.reshape(features_te.shape[0]*3, 68 * 2)
    features_vali = features_vali.reshape(features_te.shape[0], 68 * 2)
    features_te = features_te.reshape(features_te.shape[0], 68 * 2)

    return features_tr, features_vali, features_te, labels_tr, labels_vali, labels_te


def svm(features_tr, features_vali, labels_tr, labels_vali, kernel, c, gamma, degree):
    """

    :param kernel:
    :return:
    """
    if kernel == "linear":
        model = SVC(kernel='%s' %kernel, C=c)
        model.fit(features_tr, labels_tr)
    elif kernel == "linSVC":
        model = LinearSVC(C=c, max_iter=30000)
        model.fit(features_tr, labels_tr)
    elif kernel == "rbf":
        model = SVC(kernel='%s' % kernel, gamma=gamma, C=c)
        model.fit(features_tr, labels_tr)
    elif kernel == "poly":
        model = SVC(kernel='%s' % kernel, degree=degree, C=c)
        model.fit(features_tr, labels_tr)
    else:
        return "Wrong kernel input. Please check."

    score_tr = accuracy_score(labels_tr, model.predict(features_tr))
    score_vali = accuracy_score(labels_vali, model.predict(features_vali))

    print("Train acc:{}; Validation acc:{}".format(round(score_tr,3), round(score_vali,3)))

    return accuracy_score(labels_tr, model.predict(features_tr)),\
           accuracy_score(labels_vali, model.predict(features_vali))


features_tr, features_vali, features_te, labels_tr, labels_vali, labels_te = get_tr_te_set()

acc_trs = []
acc_valis = []

print("SVM_LINEAR training begin")
for c in np.linspace(0.1, 1.0, 100):
    print(c)
    accs = svm(features_tr, features_vali, labels_tr, labels_vali, 'linear', c, None, None)
    acc_trs.append(accs[0])
    acc_valis.append(accs[1])
print("SVM_LINEAR training end")

model = SVC(kernel='linear', C=acc_valis.index(max(acc_valis))+1)
model.fit(features_tr, labels_tr)
acc_te = accuracy_score(labels_te, model.predict(features_te))

print("Best validation accuracy:{}".format(np.amax(acc_valis)))
print("Optimal c value:{}".format(acc_valis.index(max(acc_valis))+1))
print("Test accuracy:{}".format(acc_te))

acc_trs = np.array(acc_trs)
acc_valis = np.array(acc_valis)

c = np.linspace(0.1, 1.0, 100)
fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(c, acc_trs, 'ro', label='Training acc')
ax.plot(c, acc_valis, 'ro', color='b', label='Validation acc')
ax.set_title('Accuracy with different c', fontsize=22)
ax.set_xlabel(r'c_value', fontsize=22)
ax.set_ylabel(r'Accuracy', fontsize=22)
ax.tick_params(labelsize=22)
ax.legend(fontsize=24)
plt.show()

"""
acc_trs = []
acc_valis = []
print("SVM_RBF training begin")
for g in np.linspace(0.00001, 0.0024, 100):
    print(g)
    accs = svm(features_tr, features_vali, labels_tr, labels_vali, 'rbf', 1, g, None)
    acc_trs.append(accs[0])
    acc_valis.append(accs[1])
print("SVM_RBF training begin")

model = SVC(kernel='rbf', gamma=acc_valis.index(max(acc_valis))+1, C=1)
model.fit(features_tr, labels_tr)
acc_te = accuracy_score(labels_te, model.predict(features_te))

print("Best validation accuracy:{}".format(np.amax(acc_valis)))
print("Optimal g value:{}".format(acc_valis.index(max(acc_valis))+1))
print("Test accuracy:{}".format(acc_te))

acc_trs = np.array(acc_trs)
acc_valis = np.array(acc_valis)

g = np.linspace(0.00001, 0.0024, 100)
fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(g, acc_trs, 'ro', label='Training acc')
ax.plot(g, acc_valis, 'ro', color='b', label='Validation acc')
ax.set_title('Accuracy with different gamma', fontsize=22)
ax.set_xlabel(r'g_value', fontsize=22)
ax.set_ylabel(r'Accuracy', fontsize=22)
ax.tick_params(labelsize=22)
ax.legend(fontsize=24)
plt.show()


acc_trs = []
acc_valis = []
print("SVM_POLY training begin")
for p in np.linspace(1, 7, 7):
    print(p)
    accs = svm(features_tr, features_vali, labels_tr, labels_vali, 'poly', 1, None, p)
    acc_trs.append(accs[0])
    acc_valis.append(accs[1])
print("SVM_POLY training begin")

model = SVC(kernel='poly', C=acc_valis.index(max(acc_valis))+1)
model.fit(features_tr, labels_tr)
acc_te = accuracy_score(labels_te, model.predict(features_te))

print("Best validation accuracy:{}".format(np.amax(acc_valis)))
print("Optimal n value:{}".format(acc_valis.index(max(acc_valis))+1))
print("Test accuracy:{}".format(acc_te))

acc_trs = np.array(acc_trs)
acc_valis = np.array(acc_valis)

p = np.linspace(1, 7, 7)
fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(p, acc_trs, 'ro', label='Training acc')
ax.plot(p, acc_valis, 'ro', color='b', label='Validation acc')
ax.set_title('Accuracy with different degree', fontsize=22)
ax.set_xlabel(r'degree_value', fontsize=22)
ax.set_ylabel(r'Accuracy', fontsize=22)
ax.tick_params(labelsize=22)
ax.legend(fontsize=24)
plt.show()
"""