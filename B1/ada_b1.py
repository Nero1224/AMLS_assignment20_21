# Gender Classification based on MLP
# import necessary API
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import dlib_feature_extract_b1 as ex
import dlib_feature_extract_b1_test as ex_te
import matplotlib.pyplot as plt
from keras.utils import np_utils


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
    #labels = np_utils.to_categorical(labels, 5)
    features_te = np.array(features_te)
    #labels_te = np_utils.to_categorical(labels_te, 5)

    features_tr = features[:features_te.shape[0]*3]
    features_vali = features[features_te.shape[0]*3:features_te.shape[0]*4]
    labels_tr = labels[:features_te.shape[0]*3]
    labels_vali = labels[features_te.shape[0]*3:features_te.shape[0]*4]

    features_tr = features_tr.reshape(features_te.shape[0]*3, 17 * 2)
    features_vali = features_vali.reshape(features_te.shape[0], 17 * 2)
    features_te = features_te.reshape(features_te.shape[0], 17 * 2)

    return features_tr, features_vali, features_te, labels_tr, labels_vali, labels_te


def ada_boost(features_tr, features_vali, labels_tr, labels_vali, n):
    """
    This function will train and output knn model based on given train, test set
    :param features:
    :param labels:
    :param num_tr:
    :param num_te:
    :param k:
    :return:
    """
    model = AdaBoostClassifier(n_estimators=n)
    model.fit(features_tr, labels_tr, sample_weight=None)

    score_tr = accuracy_score(labels_tr, model.predict(features_tr))
    score_vali = accuracy_score(labels_vali, model.predict(features_vali))

    print("Training acc:{}; Validation acc:{}".format(round(score_tr,3), round(score_vali,3)))

    return score_tr, score_vali


acc_trs = []
acc_valis = []
ada_parameters = {''}
features_tr, features_vali, features_te, labels_tr, labels_vali, labels_te = get_tr_te_set()
print("Training begin")


for n in range(400):
    print(n)
    accs = ada_boost(features_tr, features_vali, labels_tr, labels_vali, n+1)
    acc_trs.append(accs[0])
    acc_valis.append(accs[1])

model = AdaBoostClassifier(n_estimators=acc_valis.index(max(acc_valis))+1)
model.fit(features_tr, labels_tr, sample_weight=None)
acc_te = accuracy_score(labels_te, model.predict(features_te))

print("Best validation accuracy:{}".format(max(acc_valis)))
print("Optimal n value:{}".format(acc_valis.index(max(acc_valis))+1))
print("Test accuracy:{}".format(acc_te))

acc_trs = np.array(acc_trs)
acc_valis = np.array(acc_valis)

n_value = range(1, len(acc_trs)+1)
fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(n_value, acc_trs, 'ro', label='Training acc')
ax.plot(n_value, acc_valis, 'ro', color='b', label='Validation acc')
ax.set_title('Accuracy with different n', fontsize=22)
ax.set_xlabel(r'n_value', fontsize=22)
ax.set_ylabel(r'Accuracy', fontsize=22)
ax.tick_params(labelsize=22)
ax.legend(fontsize=24)
plt.show()
