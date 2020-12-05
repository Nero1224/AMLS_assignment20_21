# Gender Classification based on MLP
# import necessary API
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import dlib_feature_extract_b1 as ex
import matplotlib.pyplot as plt
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

    features_tr = features_tr.reshape(num_tr, 68 * 2)
    features_te = features_te.reshape(num_te, 68 * 2)
    features_vali = features_vali.reshape(num_vali, 68 * 2)
    labels_tr = list(zip(*labels_tr))[0]
    labels_te = list(zip(*labels_te))[0]
    labels_vali = list(zip(*labels_vali))[0]

    return features_tr, features_te, features_vali, labels_tr, labels_te, labels_vali


def knn(features_tr, features_te, labels_tr, labels_te, k):
    """
    This function will train and output knn model based on given train, test set
    :param features:
    :param labels:
    :param num_tr:
    :param num_te:
    :param k:
    :return:
    """
    model = KNeighborsClassifier(k)
    model.fit(features_tr, labels_tr)

    score = metrics.accuracy_score(labels_te, model.predict(features_te))
    print(score)
    return score


acc = []
features_tr, features_te, features_vali, labels_tr, labels_te, labels_vali = get_tr_te_set(600, 200, 200, 1500)
for k in range(200): acc.append(knn(features_tr, features_te, labels_tr, labels_te, k+1))
#knn(features_tr, features_te, labels_tr, labels_te, 100)

acc = np.array(acc)
k_value = range(1, len(acc)+1)

fig, ax = plt.subplots(1,1, figsize=(10,6))

ax.plot(k_value, acc, 'ro', label='Training acc')
#ax.plot(epochs, val_acc, 'b', label='Validation acc')
ax.set_title('Accuracy with different k', fontsize=22)
ax.set_xlabel(r'k_value', fontsize=22)
ax.set_ylabel(r'Accuracy', fontsize=22)
ax.tick_params(labelsize=22)
ax.legend(fontsize=24)
plt.show()

