# Gender Classification based on MLP
# import necessary API
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import dlib_feature_extract_a1 as ex

"""
def gpu_manage():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    config = tf.compat.v1.ConfigProto()
    config.gpu_option.per_process_gpu_memory_fraction = 0.1
    config.gpu_option.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
"""


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


features_tr, features_te, labels_tr, labels_te = get_tr_te_set(400, 100)
for k in range(100): knn(features_tr, features_te, labels_tr, labels_te, k+1)
#knn(features_tr, features_te, labels_tr, labels_te, 100)


