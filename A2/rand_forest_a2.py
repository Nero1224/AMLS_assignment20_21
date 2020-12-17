import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import dlib_feature_extract_a2 as ex
import dlib_feature_extract_a2_test as ex_te
from keras.utils import np_utils


def get_tr_te_set():
    """
    This function will extract dlib feature and labels for all CelebA images.
    It also provided prepared training, validation and test set with ratio 6:2:2
    :return:
    features_tr:    an array containing flatten 68 landmarks features for training
    features_vali:  an array containing flatten 68 landmarks features for validation
    features_te:    an array containing flatten 68 landmarks features for test
    labels_tr:      an list containing labels for training set
    labels_vali:    an list containing labels for validation set
    labels_te:      anlist containing labels for test set
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


def rand_forest(features_tr, features_vali, labels_tr, labels_vali, features_n_estimators):
    model = RandomForestClassifier(n_estimators=features_n_estimators)
    model.fit(features_tr, labels_tr)

    score_tr = accuracy_score(labels_tr, model.predict(features_tr))
    score_vali = accuracy_score(labels_vali, model.predict(features_vali))

    print("Training acc:{}; Validation acc:{}".format(score_tr, score_vali))

    return score_tr, score_vali


acc_trs = []
acc_valis = []
features_tr, features_vali, features_te, labels_tr, labels_vali, labels_te = get_tr_te_set()
print("Random_Forest Training begin")
for n in range(100):
    print(n)
    accs = rand_forest(features_tr, features_vali, labels_tr, labels_vali, n+1)
    acc_trs.append(accs[0])
    acc_valis.append(accs[1])
print("Random_Forest Test begin")
model = RandomForestClassifier(acc_valis.index(max(acc_valis))+1)
model.fit(features_tr, labels_tr)
acc_te = accuracy_score(labels_te, model.predict(features_te))

print("Best validation accuracy:{}".format(np.amax(acc_valis)))
print("Optimal n value:{}".format(acc_valis.index(max(acc_valis))+1))
print("Test accuracy:{}".format(acc_te))

acc_trs = np.array(acc_trs)
acc_valis = np.array(acc_valis)

n_value = range(1, len(acc_trs)+1)
fig, ax = plt.subplots(1,1, figsize=(10,6))
ax.plot(n_value, acc_trs, 'ro', label='Training acc')
ax.plot(n_value, acc_valis, 'ro', color='b', label='Validation acc')
ax.set_title('Accuracy with different n', fontsize=22)
ax.set_xlabel(r'n_value', fontsize=22)
ax.set_ylabel(r'Accuracy', fontsize=22)
ax.tick_params(labelsize=22)
ax.legend(fontsize=24)
plt.show()

