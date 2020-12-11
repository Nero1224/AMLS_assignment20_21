import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import dlib_feature_extract_a2 as ex
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


def rand_forest(features_tr, features_vali, labels_tr, labels_vali, features_n_estimators):
    model = RandomForestClassifier(n_estimators=features_n_estimators)
    model.fit(features_tr, labels_tr)

    score_tr = accuracy_score(labels_tr, model.predict(features_tr))
    score_vali = accuracy_score(labels_vali, model.predict(features_vali))

    print("Training acc:{}; Validation acc:{}".format(score_tr, score_vali))

    return score_tr, score_vali


acc_trs = []
acc_valis = []
features_tr, features_vali, features_te, labels_tr, labels_vali, labels_te = get_tr_te_set(3000, 500, 500, 5000)
print("Training begin")
for n in range(100):
    print(n)
    accs = rand_forest(features_tr, features_vali, labels_tr, labels_vali, n+1)
    acc_trs.append(accs[0])
    acc_valis.append(accs[1])

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

