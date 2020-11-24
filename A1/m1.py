import matplotlib.pyplot as plt
import numpy as np
from pandas import *
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import cv2
import string

X = np.zeros(shape=(5000, 178*218*3))
for n in range(5000):
    image = cv2.imread("../Datasets/celeba/img/%s.jpg" %n, cv2.IMREAD_COLOR)
    # cv2.imshow('image', image)
    # cv2.waitKey(1000)
    image = image.reshape(1, 178 * 218 * 3)
    X[n] = image

Y = np.zeros(shape=(5000, 1))
data = pandas.read_csv("../Datasets/celeba/labels.csv", header=None)
for n in range(5000):
    middle = np.array(data.loc[n+1]).tolist()
    Y[n] = int(str(middle[0]).split('\t')[-2])
print(X)
print(Y)

test_size = 0.3

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)
print(X_train)
print(Y_train)
"""
print(Y)
cv2.imshow('image', X)
cv2.waitKey(0)



def KNNClassifier(X_train, Y_train, X_test, k):
    Create KNN object with a K coefficient
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, Y_train) # Fit KNN model
    y_pred = neigh.predict(X_test)
    return y_pred


Y_pred = KNNClassifier(X_train, Y_train, X_test, 1)
score = metrics.accuracy_score(Y_test, Y_pred)
print(score)
"""