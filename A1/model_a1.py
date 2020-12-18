import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
import joblib
import dlib_feature_extract_a1 as ex_a1
import dlib_feature_extract_a1_test as ex_te_a1


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
    features, labels = ex_a1.extract_features_labels()
    features_te, labels_te = ex_te_a1.extract_features_labels()
    print("Extraction end")

    features = np.array(features)
    features_te = np.array(features_te)

    features = features.reshape(features.shape[0], 68 * 2)
    features_te = features_te.reshape(features_te.shape[0], 68 * 2)

    return features, features_te, labels, labels_te


def rf_training(features, features_te, labels, labels_te):
    n = 66
    train_sizes, train_score, vali_score = learning_curve(RandomForestClassifier(n),
                                                          features, labels, cv=5, scoring=None,
                                                          random_state=3,
                                                          train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                                                       1.0])
    train_score_mean = np.mean(train_score, 1)
    vali_score_mean = np.mean(vali_score, 1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(train_sizes, train_score_mean, 'o-', color='r', label='Training')
    ax.plot(train_sizes, vali_score_mean, 'o-', color='b', label='cross_validation')
    ax.set_title('Learning curve for RF(n_estimators=66)', fontsize=22)
    ax.set_xlabel('Training examples', fontsize=22)
    ax.set_ylabel('Scores', fontsize=22)
    ax.tick_params(labelsize=22)
    ax.legend(fontsize=24, loc='best')
    plt.show()

    vali_score_mean = list(vali_score_mean)
    print("Best training size:", train_sizes[vali_score_mean.index(max(vali_score_mean))])
    print("Best cross-validation score: ", max(np.array(vali_score_mean)))
    print("Corresponding training score: ", train_score_mean[vali_score_mean.index(max(vali_score_mean))])

    model_a1 = RandomForestClassifier(n)
    model_a1.fit(features[:train_sizes[vali_score_mean.index(max(vali_score_mean))]],
                 labels[:train_sizes[vali_score_mean.index(max(vali_score_mean))]])

    joblib.dump(model_a1, 'model_a1.pkl')

    score_te = accuracy_score(labels_te, model_a1.predict(features_te))
    print("Test accuracy:", score_te)

    return train_score_mean[vali_score_mean.index(max(vali_score_mean))], score_te