# Gender Classification based on MLP
# import necessary API
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
import dlib_feature_extract_b1 as ex
import dlib_feature_extract_b1_test as ex_te
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


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
    features_te = np.array(features_te)

    features = features.reshape(features.shape[0], 17 * 2)
    features_te = features_te.reshape(features_te.shape[0], 17 * 2)

    return features, features_te, labels, labels_te


features, features_te, labels, labels_te = get_tr_te_set()

n = 66
train_sizes, train_score, vali_score = learning_curve(AdaBoostClassifier(n_estimators=n),
                                                   features, labels, cv=5, scoring=None,
                                                   train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
train_score_mean = np.mean(train_score, 1)
vali_score_mean = np.mean(vali_score, 1)

plt.plot(train_sizes,train_score_mean,'o-',color='r',label='Trainning')
plt.plot(train_sizes,vali_score_mean,'o-',color='b',label='cross_validation')
plt.xlabel('Training examples')
plt.ylabel('Scores')
plt.legend()
plt.show()

vali_score_mean = list(vali_score_mean)
print("Best training size:",train_sizes[vali_score_mean.index(max(vali_score_mean))])
print("Best cross-validation score: ", max(np.array(vali_score_mean)))
print("Corresponding training score: ", train_score_mean[vali_score_mean.index(max(vali_score_mean))])


model = AdaBoostClassifier(n_estimators=n)
model.fit(features[:train_sizes[vali_score_mean.index(max(vali_score_mean))]],
          labels[:train_sizes[vali_score_mean.index(max(vali_score_mean))]])

score_te = accuracy_score(labels_te, model.predict(features_te))
print("Test accuracy:", score_te)
