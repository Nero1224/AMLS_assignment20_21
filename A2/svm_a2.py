# Gender Classification based on MLP
# import necessary API
import numpy as np
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
import dlib_feature_extract_a2 as ex
import dlib_feature_extract_a2_test as ex_te
import matplotlib.pyplot as plt


def get_tr_te_set():
    """

    :return:
    """
    print("Extraction begin")
    features, labels = ex.extract_features_labels()
    features_te, labels_te = ex_te.extract_features_labels()
    print("Extraction end")

    features = np.array(features)
    features_te = np.array(features_te)

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
n_lin = 100
n_rbf = 100
n_poly = 7

acc_trs = []
acc_valis = []
best_c = 0.1
print("SVM_LINEAR training begin")
for c in np.linspace(0.1, 1.0, n_lin):
    print(c)
    accs = svm(features_tr, features_vali, labels_tr, labels_vali, 'linear', c, None, None)
    if len(acc_valis) != 0:
        if accs[1] > max(acc_valis): best_c = c
    acc_trs.append(accs[0])
    acc_valis.append(accs[1])
print("SVM_LINEAR Test begin")
model = SVC(kernel='linear', C=best_c)
model.fit(features_tr, labels_tr)
acc_te = accuracy_score(labels_te, model.predict(features_te))

print("Best validation accuracy:{}".format(np.amax(acc_valis)))
print("Optimal c value:{}".format(best_c))
print("Test accuracy:{}".format(acc_te))

acc_trs = np.array(acc_trs)
acc_valis = np.array(acc_valis)

c = np.linspace(0.1, 1.0, n_lin)
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
best_g = 0.00001
print("SVM_RBF training begin")
for g in np.linspace(0.00001, 0.0024, n_rbf):
    print(g)
    accs = svm(features_tr, features_vali, labels_tr, labels_vali, 'rbf', 1, g, None)
    if len(acc_valis) != 0:
        if accs[1] > max(acc_valis): best_g = g
    acc_trs.append(accs[0])
    acc_valis.append(accs[1])
print("SVM_RBF training begin")

acc_trs = np.array(acc_trs)
acc_tes = np.array(acc_valis)

model = SVC(kernel='rbf', gamma=best_g, C=1)
model.fit(features_tr, labels_tr)
acc_te = accuracy_score(labels_te, model.predict(features_te))

print("Best validation accuracy:{}".format(np.amax(acc_valis)))
print("Optimal g value:{}".format(best_g))
print("Test accuracy:{}".format(acc_te))

g = np.linspace(0.00001, 0.0024, n_rbf)
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
best_d = 1
print("SVM_POLY training begin")
for d in np.linspace(1, 7, n_poly):
    print(d)
    accs = svm(features_tr, features_vali, labels_tr, labels_vali, 'poly', 1, None, d)
    if len(acc_valis) != 0:
        if accs[1] > max(acc_valis): best_d = d
    acc_trs.append(accs[0])
    acc_valis.append(accs[1])
print("SVM_POLY training begin")

acc_trs = np.array(acc_trs)
acc_valis = np.array(acc_valis)

model = SVC(kernel='poly', degree=best_d, C=1)
model.fit(features_tr, labels_tr)
acc_te = accuracy_score(labels_te, model.predict(features_te))

print("Best validation accuracy:{}".format(np.amax(acc_valis)))
print("Optimal n value:{}".format(best_d))
print("Test accuracy:{}".format(acc_te))

d = np.linspace(1, 7, n_poly)
fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(d, acc_trs, 'ro', label='Training acc')
ax.plot(d, acc_valis, 'ro', color='b', label='Validation acc')
ax.set_title('Accuracy with different degree', fontsize=22)
ax.set_xlabel(r'd_value', fontsize=22)
ax.set_ylabel(r'Accuracy', fontsize=22)
ax.tick_params(labelsize=22)
ax.legend(fontsize=24)
plt.show()