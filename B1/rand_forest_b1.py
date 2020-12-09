import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import dlib_feature_extract_b1 as ex
from keras.utils import np_utils


def get_tr_te_set(num_tr, num_te, num_vali, n):
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


def rand_forest(features_tr, features_te, labels_tr, labels_te, features_n_estimators):
    model = RandomForestClassifier(n_estimators=features_n_estimators)
    model.fit(features_tr, labels_tr)
    score = accuracy_score(labels_te, model.predict(features_te))

    print(score)
    return score


acc = []
features_tr, features_te, features_vali, labels_tr, labels_te, labels_vali = get_tr_te_set(3000, 1000, 1000, 7500)
for n in range(200): acc.append(rand_forest(features_tr, features_te, labels_tr, labels_te, n+1))

acc = np.array(acc)
n_value = range(1, len(acc)+1)

fig, ax = plt.subplots(1,1, figsize=(10,6))

ax.plot(n_value, acc, 'ro', label='Training acc')
ax.set_title('Accuracy with different k', fontsize=22)
ax.set_xlabel(r'n_value', fontsize=22)
ax.set_ylabel(r'Accuracy', fontsize=22)
ax.tick_params(labelsize=22)
ax.legend(fontsize=24)
plt.show()
