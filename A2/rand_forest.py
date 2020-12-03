import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import dlib_facefeature_extract as ex


def get_tr_te_set(num_tr, num_te):
    """
    This function will automatically load the images
    inside dataset with given train and test set number

    :param num_tr: number of train set
    :param num_te: number of test set
    :return: train set and test set
    """
    features, labels = ex.extract_features_labels()
    features = np.array(features)
    labels = np.array([labels, -(labels - 1)]).T

    features_tr = features[:num_tr]
    features_te = features[num_tr:(num_tr + num_te)]
    labels_tr = labels[:num_tr]
    labels_te = labels[num_tr:(num_tr + num_te)]

    features_tr = features_tr.reshape(num_tr, 68 * 2)
    features_te = features_te.reshape(num_te, 68 * 2)
    labels_tr = list(zip(*labels_tr))[0]
    labels_te = list(zip(*labels_te))[0]

    return features_tr, features_te, labels_tr, labels_te


def rand_forest(features_tr, features_te, labels_tr, labels_te, features_n_estimators):
    model = RandomForestClassifier(n_estimators=features_n_estimators)
    model.fit(features_tr, labels_tr)
    score = accuracy_score(labels_te, model.predict(features_te))

    print(score)


features_tr, features_te, labels_tr, labels_te = get_tr_te_set(400, 100)
rand_forest(features_tr, features_te, labels_tr, labels_te, 100)
