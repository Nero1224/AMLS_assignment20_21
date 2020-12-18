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
    This function will extract dlib feature and gender labels for initial CelebA dataset (5000) for training and
    additional CelebA dataset (1000) for testing.
    :return:
    features:       an np array containing flatten 68 landmarks features for training
    features_te:    an np array containing flatten 68 landmarks features for test
    labels:         an list containing labels for training set
    labels_te:      an list containing labels for test set
    """
    # extracting dlib features and gender labels from training set and test set
    print("Extraction begin")
    features, labels = ex_a1.extract_features_labels()
    features_te, labels_te = ex_te_a1.extract_features_labels()
    print("Extraction end")

    # transfer to numpy array to do matrix operation
    features = np.array(features)
    features_te = np.array(features_te)

    # flatten data for training
    features = features.reshape(features.shape[0], 68 * 2)
    features_te = features_te.reshape(features_te.shape[0], 68 * 2)

    # return features and labels for training and testing
    return features, features_te, labels, labels_te


def rf_training(features, features_te, labels, labels_te):
    """
    This function will train a new RF model based learning curve. It will plot the learning curve,
    print training, cross-validation and test accuracy and save the model.
    :param features:       an np array containing flatten 68 landmarks features for training
    :param features_te:    an np array containing flatten 68 landmarks features for test
    :param labels:         an list containing labels for training set
    :param labels_te:      an list containing labels for test set
    :return:
    train_score_mean[vali_score_mean.index(max(vali_score_mean))]   training accuracy under hyperparameter giving best validation accuracy
    score_te                                                        testing accuracy under current model

    """
    # assign reasonable hyperparameter value
    n = 66
    # generating learning curve
    train_sizes, train_score, vali_score = learning_curve(RandomForestClassifier(n),
                                                          features, labels, cv=5, scoring=None,
                                                          random_state=3,
                                                          train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                                                       1.0])
    # calculate mean accuracies
    train_score_mean = np.mean(train_score, 1)
    vali_score_mean = np.mean(vali_score, 1)

    # plot learning curve
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(train_sizes, train_score_mean, 'o-', color='r', label='Training')
    ax.plot(train_sizes, vali_score_mean, 'o-', color='b', label='cross_validation')
    ax.set_title('Learning curve for RF(n_estimators=66)', fontsize=22)
    ax.set_xlabel('Training examples', fontsize=22)
    ax.set_ylabel('Scores', fontsize=22)
    ax.tick_params(labelsize=22)
    ax.legend(fontsize=24, loc='best')
    plt.show()

    # give best training size and corresponding performance
    vali_score_mean = list(vali_score_mean)
    print("Best training size:", train_sizes[vali_score_mean.index(max(vali_score_mean))])
    print("Best cross-validation score: ", max(np.array(vali_score_mean)))
    print("Corresponding training score: ", train_score_mean[vali_score_mean.index(max(vali_score_mean))])

    # re-train model
    model_a1 = RandomForestClassifier(n)
    model_a1.fit(features[:train_sizes[vali_score_mean.index(max(vali_score_mean))]],
                 labels[:train_sizes[vali_score_mean.index(max(vali_score_mean))]])

    # save model
    joblib.dump(model_a1, 'model_a1.pkl')

    # give final test accuracy
    score_te = accuracy_score(labels_te, model_a1.predict(features_te))
    print("Test accuracy:", score_te)

    return train_score_mean[vali_score_mean.index(max(vali_score_mean))], score_te