# Gender Classification based on MLP
# import necessary API
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import dlib_feature_extract_b1 as ex
import matplotlib.pyplot as plt
from keras.utils import np_utils
import os
import shutil

base_path = os.path.dirname(os.getcwd())
img_path = os.path.join(base_path, r"Datasets\cartoon_set\img")
jpg_path = os.path.join(base_path, r"Datasets\cartoon_set\img_jpg")
if os.path.exists(jpg_path): pass
else:
    os.makedirs(jpg_path)
    for img in os.listdir(img_path):
        shutil.copyfile(os.path.join(img_path, r"%s" %img), os.path.join(jpg_path, "%s.jpg" %img.split(".")[0]))

def get_tr_te_set(num_tr, num_te, num_vali,  n):
    """
    This function will automatically load the images
    inside dataset with given train and test set number

    :param num_tr: number of train set
    :param num_te: number of test set
    :return: train set and test set
    """
    features, labels = ex.extract_features_labels(n)
    features = np.array(features)
    labels = np_utils.to_categorical(labels, 5)

    features_tr = features[:num_tr]
    features_te = features[num_tr:(num_tr + num_te)]
    features_vali = features[(num_tr + num_te): (num_tr + num_te + num_vali)]
    labels_tr = labels[:num_tr]
    labels_te = labels[num_tr:(num_tr + num_te)]
    labels_vali = labels[(num_tr + num_te): (num_tr + num_te + num_vali)]

    features_tr = features_tr.reshape(num_tr, 17 * 2)
    features_te = features_te.reshape(num_te, 17 * 2)
    features_vali = features_vali.reshape(num_vali, 17 * 2)
    labels_tr = list(zip(*labels_tr))[0]
    labels_te = list(zip(*labels_te))[0]
    labels_vali = list(zip(*labels_vali))[0]

    return features_tr, features_te, features_vali, labels_tr, labels_te, labels_vali


def svm(features_tr, features_te, labels_tr, labels_te, kernel, c, gamma, degree):
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

    print(accuracy_score(labels_te, model.predict(features_te)))

    return accuracy_score(labels_tr, model.predict(features_tr)),\
           accuracy_score(labels_te, model.predict(features_te))


features_tr, features_te, features_vali, labels_tr, labels_te, labels_vali = get_tr_te_set(3000, 1000, 1000, 7500)

"""
acc_trs = []
acc_tes = []
for c in np.linspace(0.1, 1.0, 100):
    acc_tr, acc_te = svm(features_tr, features_te, labels_tr, labels_te, 'linear', c, None, None)
    acc_trs.append(acc_tr)
    acc_tes.append(acc_te)

acc_trs = np.array(acc_trs)
acc_tes = np.array(acc_tes)
c = np.linspace(0.1, 1.0, 100)

fig, ax = plt.subplots(1, 1, figsize=(10,6))

ax.plot(c, acc_trs, 'ro', label='Training acc')
ax.plot(c, acc_tes, 'ro', color='b', label='Testing acc')
ax.set_title('Accuracy with different c', fontsize=22)
ax.set_xlabel(r'c_value', fontsize=22)
ax.set_ylabel(r'Accuracy', fontsize=22)
ax.tick_params(labelsize=22)
ax.legend(fontsize=24)
plt.show()


acc_trs = []
acc_tes = []
for g in np.linspace(0.00001, 0.0024, 100):
    acc_tr, acc_te = svm(features_tr, features_te, labels_tr, labels_te, 'rbf', 1, g, None)
    acc_trs.append(acc_tr)
    acc_tes.append(acc_te)

acc_trs = np.array(acc_trs)
acc_tes = np.array(acc_tes)
g = np.linspace(0.00001, 0.0024, 100)

fig, ax = plt.subplots(1, 1, figsize=(10,6))

ax.plot(c, acc_trs, 'ro', label='Training acc')
ax.plot(c, acc_tes, 'ro', color='b', label='Testing acc')
ax.set_title('Accuracy with different g', fontsize=22)
ax.set_xlabel(r'g_value', fontsize=22)
ax.set_ylabel(r'Accuracy', fontsize=22)
ax.tick_params(labelsize=22)
ax.legend(fontsize=24)
plt.show()

"""
acc_trs = []
acc_tes = []
for p in np.linspace(1, 10, 10):
    acc_tr, acc_te = svm(features_tr, features_te, labels_tr, labels_te, 'poly', 1, None, p)
    acc_trs.append(acc_tr)
    acc_tes.append(acc_te)

acc_trs = np.array(acc_trs)
acc_tes = np.array(acc_tes)
p = np.linspace(1, 30, 30)

fig, ax = plt.subplots(1, 1, figsize=(10,6))

ax.plot(p, acc_trs, 'ro', label='Training acc')
ax.plot(p, acc_tes, 'ro', color='b', label='Testing acc')
ax.set_title('Accuracy with different p', fontsize=22)
ax.set_xlabel(r'p_value', fontsize=22)
ax.set_ylabel(r'Accuracy', fontsize=22)
ax.tick_params(labelsize=22)
ax.legend(fontsize=24)
plt.show()
