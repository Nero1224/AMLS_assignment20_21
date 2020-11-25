import matplotlib.pyplot as plt
import numpy as np
from pandas import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import cv2

images = []
labels = []
test_size =0.4

data = pandas.read_csv("../Datasets/celeba/labels.csv", header=None)
for n in range(10):
    image = cv2.imread("../Datasets/celeba/img/%s.jpg" %n, cv2.IMREAD_COLOR)
    label = np.array(data.loc[n+1]).tolist()
    label = int(str(label[0]).split('\t')[-2])
    images.append(image)
    labels.append(label)

Y = LabelEncoder().fit_transform(labels)
Train_idx, Test_idx = train_test_split(range(len(Y)), test_size=test_size, random_state=1, stratify=Y)
Y_train = Y[Train_idx]
Y_test = Y[Test_idx]


def color_hist(image):
    """This function will extract frequency of three color channels"""
    # add a mask may be able to improve performance
    hist = cv2.calcHist([image], [0,1,2], None, [8]*3, [0,256]*3)
    return hist.ravel()


X = np.row_stack([color_hist(image) for image in images])
X_train = X[Train_idx, :]
X_test = X[Test_idx, :]

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

print("Random Forest Test Accuracy:", accuracy_score(Y_test, Y_pred))





