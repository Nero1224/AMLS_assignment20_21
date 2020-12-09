import cv2
import numpy as np
import matplotlib.pyplot as plt

for n in range(1):
    #img = cv2.imread("../Datasets/cartoon_set/img_jpg/%s.jpg" % n, cv2.IMREAD_COLOR)
    #img_grey = cv2.imread("../Datasets/cartoon_set/img_jpg/%s.jpg" % n, 0)
    img = cv2.imread("../Datasets/cartoon_set/img_jpg/3.jpg", cv2.IMREAD_COLOR)
    img_grey = cv2.imread("../Datasets/cartoon_set/img_jpg/3.jpg", 0)

    mask = np.zeros(np.array(img).shape[:2], np.uint8)
    mask[250:280, 170:330] = 255
    mask_img = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow('img', img)
    #cv2.imshow('img_grey', img_grey)
    #cv2.imshow('mask', mask)
    cv2.imshow('mask_img', mask_img)

    color = ('b', 'g', 'r')
    for k, c in enumerate(color):
        hist_full = cv2.calcHist([img], [k], None, [256], [0, 256])
        #hist_mask = cv2.calcHist([img], [k], mask, [256], [0, 256])
        plt.plot(hist_full, color=c)
        #plt.subplot(122), plt.plot(hist_mask, color=c)
        plt.xlim([0, 256])
    plt.show()

    for k, c in enumerate(color):
        hist_mask = cv2.calcHist([img], [k], mask, [256], [0, 256])
        plt.plot(hist_mask, color=c)
        plt.xlim([0, 256])
    plt.show()
    k = cv2.waitKey()
