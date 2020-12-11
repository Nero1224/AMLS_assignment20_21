# Gender Classification based on MLP
# import necessary API
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import dlib_feature_extract_b1 as ex
import matplotlib.pyplot as plt
from keras.utils import np_utils


def get_tr_te_set(num_tr, num_vali, num_te, n):
    """
    This function will automatically load the images
    inside dataset with given train and test set number

    :param num_tr: number of train set
    :param num_te: number of test set
    :return: train set and test set
    """
    print("Extraction begin")
    features, labels = ex.extract_features_labels(n)
    print("Extraction end")
    features = np.array(features)
    labels = np_utils.to_categorical(labels, 2)

    features_tr = features[:num_tr]
    features_vali = features[num_tr:(num_tr + num_vali)]
    features_te = features[(num_tr + num_vali): (num_tr + num_vali + num_te)]
    labels_tr = labels[:num_tr]
    labels_vali = labels[num_tr:(num_tr + num_vali)]
    labels_te = labels[(num_tr + num_vali): (num_tr + num_vali + num_te)]

    features_tr = features_tr.reshape(num_tr, 68 * 2)
    features_vali = features_vali.reshape(num_vali, 68 * 2)
    features_te = features_te.reshape(num_te, 68 * 2)
    labels_tr = list(zip(*labels_tr))[0]
    labels_vali = list(zip(*labels_vali))[0]
    labels_te = list(zip(*labels_te))[0]

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

    print("Train acc:{}; Validation acc:{}".format(score_tr, score_vali))

    return accuracy_score(labels_tr, model.predict(features_tr)),\
           accuracy_score(labels_vali, model.predict(features_vali))


features_tr, features_vali, features_te, labels_tr, labels_vali, labels_te = get_tr_te_set(3000, 500, 500, 5000)

acc_trs = []
acc_valis = []
print("SVM_LINEAR training begin")
for c in np.linspace(0.1, 1.0, 100):
    print(c)
    accs = svm(features_tr, features_vali, labels_tr, labels_vali, 'linear', c, None, None)
    acc_trs.append(accs[0])
    acc_valis.append(accs[1])
print("SVM_LINEAR training end")

acc_trs = np.array(acc_trs)
acc_valis = np.array(acc_valis)
"""
model = SVC(kernel='linear', C=np.where(acc_valis==np.max(acc_valis)))
model.fit(features_tr, labels_tr)
acc_te = accuracy_score(labels_te, model.predict(features_te))

print("Best validation accuracy:{}".format(np.amax(acc_valis)))
print("Optimal c value:{}".format(np.where(acc_valis==np.max(acc_valis))))
print("Test accuracy:{}".format(acc_te))
"""
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


acc_trs = []
acc_valis = []
print("SVM_RBF training begin")
for g in np.linspace(0.00001, 0.0024, 100):
    print(g)
    accs = svm(features_tr, features_vali, labels_tr, labels_vali, 'rbf', 1, g, None)
    acc_trs.append(accs[0])
    acc_valis.append(accs[1])
print("SVM_RBF training begin")

acc_trs = np.array(acc_trs)
acc_tes = np.array(acc_valis)
"""
model = SVC(kernel='rbf', gamma=, C=1)
model.fit(features_tr, labels_tr)
acc_te = accuracy_score(labels_te, model.predict(features_te))

print("Best validation accuracy:{}".format(np.amax(acc_valis)))
print("Optimal g value:{}".format(np.where(acc_valis==np.max(acc_valis))))
print("Test accuracy:{}".format(acc_te))
"""
g = np.linspace(0.00001, 0.0024, 100)
fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(c, acc_trs, 'ro', label='Training acc')
ax.plot(c, acc_tes, 'ro', color='b', label='Validation acc')
ax.set_title('Accuracy with different gamma', fontsize=22)
ax.set_xlabel(r'g_value', fontsize=22)
ax.set_ylabel(r'Accuracy', fontsize=22)
ax.tick_params(labelsize=22)
ax.legend(fontsize=24)
plt.show()


acc_trs = []
acc_valis = []
print("SVM_POLY training begin")
for p in np.linspace(1, 10, 10):
    print(p)
    accs = svm(features_tr, features_vali, labels_tr, labels_vali, 'poly', 1, None, p)
    acc_trs.append(accs[0])
    acc_valis.append(accs[1])
print("SVM_POLY training begin")

acc_trs = np.array(acc_trs)
acc_valis = np.array(acc_valis)
p = np.linspace(1, 10, 10)
"""
model = SVC(kernel='poly', C=np.where(acc_valis==np.max(acc_valis)))
model.fit(features_tr, labels_tr)
acc_te = accuracy_score(labels_te, model.predict(features_te))

print("Best validation accuracy:{}".format(np.amax(acc_valis)))
print("Optimal n value:{}".format(np.where(acc_valis==np.max(acc_valis))))
print("Test accuracy:{}".format(acc_te))
"""
fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(p, acc_trs, 'ro', label='Training acc')
ax.plot(p, acc_valis, 'ro', color='b', label='Validation acc')
ax.set_title('Accuracy with different poly', fontsize=22)
ax.set_xlabel(r'p_value', fontsize=22)
ax.set_ylabel(r'Accuracy', fontsize=22)
ax.tick_params(labelsize=22)
ax.legend(fontsize=24)
plt.show()
