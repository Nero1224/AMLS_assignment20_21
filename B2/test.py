import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
import dlib_feature_extract_b2 as ex
"""
for n in range(1):
    img = cv2.imread("../Datasets/cartoon_set/img_jpg/8.jpg", cv2.IMREAD_COLOR)
    cv2.imshow('img', img)
    cv2.imshow('mask_img', img[250:280, 170:330])

    color = ('b', 'g', 'r')
    for k, c in enumerate(color):
        hist_full = cv2.calcHist([img], [k], None, [64], [0, 256])
        plt.plot(hist_full, color=c)
        plt.xlim([0, 64])
    plt.show()

    for k, c in enumerate(color):
        hist_mask = cv2.calcHist([img[250:280, 170:330]], [k], None, [64], [0, 256])
        plt.plot(hist_mask, color=c)
        plt.xlim([0, 64])
    plt.show()
    k = cv2.waitKey()

"""
features, labels = ex.extract_features_labels(5, mask_sw='on')
print(features.shape)
cv2.imshow('img', features[2])
k = cv2.waitKey()
