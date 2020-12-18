# Gender Classification based on MLP
# import necessary API
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve
import dlib_feature_extract_b1 as ex_b1
import dlib_feature_extract_b1_test as ex_te_b1
import matplotlib.pyplot as plt
import joblib


def get_tr_te_set():
    """
    This function will extract dlib feature and face shape labels for initial cartoon dataset (10000) for training and
    additional cartoon dataset (2500) for testing.
    :return:
    features:       an np array containing flatten 17 landmarks features for training
    features_te:    an np array containing flatten 17 landmarks features for test
    labels:         an list containing labels for training set
    labels_te:      an list containing labels for test set
    """
    # extracting dlib features and gender labels from training set and test set
    print("Extraction begin")
    features, labels = ex_b1.extract_features_labels()
    features_te, labels_te = ex_te_b1.extract_features_labels()
    print("Extraction end")

    # transfer to numpy array to do matrix operation
    features = np.array(features)
    features_te = np.array(features_te)

    # flatten data for training
    features = features.reshape(features.shape[0], 17 * 2)
    features_te = features_te.reshape(features_te.shape[0], 17 * 2)

    # return features and labels for training and testing
    return features, features_te, labels, labels_te


def svm_training(features, features_te, labels, labels_te):
    """
    This function will train a new SVM model based learning curve. It will plot the learning curve,
    print training, cross-validation and test accuracy and save the model.
    :param features:       an np array containing flatten 17 landmarks features for training
    :param features_te:    an np array containing flatten 17 landmarks features for test
    :param labels:         an list containing labels for training set
    :param labels_te:      an list containing labels for test set
    :return:
    train_score_mean[vali_score_mean.index(max(vali_score_mean))]   training accuracy under hyperparameter giving best validation accuracy
    score_te
    """
    # assign reasonable hyperparameter value
    c = 1
    g = 0.002
    # generating learning curve
    train_sizes, train_score, vali_score = learning_curve(SVC(kernel='rbf', C=c, gamma=g),
                                                          features, labels, cv=5, scoring=None,
                                                          train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                                                                       1.0])
    # calculate mean accuracies
    train_score_mean = np.mean(train_score, 1)
    vali_score_mean = np.mean(vali_score, 1)

    # plot learning curve
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(train_sizes, train_score_mean, 'o-', color='r', label='Trainning')
    ax.plot(train_sizes, vali_score_mean, 'o-', color='b', label='cross_validation')
    ax.set_title('Learning curve for SVM(kernel= rbf, C=1, gamma=0.002)', fontsize=22)
    ax.set_xlabel('Training examples', fontsize=22)
    ax.set_ylabel('Scores', fontsize=22)
    ax.tick_params(labelsize=22)
    ax.legend(fontsize=24)
    plt.show()

    # give best training size and corresponding performance
    vali_score_mean = list(vali_score_mean)
    print("Best training size:", train_sizes[vali_score_mean.index(max(vali_score_mean))])
    print("Best cross-validation score: ", max(np.array(vali_score_mean)))
    print("Corresponding training score: ", train_score_mean[vali_score_mean.index(max(vali_score_mean))])

    model_b1 = SVC(kernel='rbf', C=c, gamma=g)
    model_b1.fit(features[:train_sizes[vali_score_mean.index(max(vali_score_mean))]],
                 labels[:train_sizes[vali_score_mean.index(max(vali_score_mean))]])

    # save model
    joblib.dump(model_b1, 'model_b1.pkl')

    # give final test accuracy
    score_te = accuracy_score(labels_te, model_b1.predict(features_te))
    print("Test accuracy:", score_te)

    return train_score_mean[vali_score_mean.index(max(vali_score_mean))], score_te


