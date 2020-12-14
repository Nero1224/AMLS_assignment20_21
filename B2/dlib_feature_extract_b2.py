import os
import numpy as np
from keras.preprocessing import image
import cv2
import dlib

# PATH TO ALL IMAGES
global basedir, image_paths, target_size
basedir = os.path.join(os.path.dirname(os.getcwd()), r'Datasets\cartoon_set')
images_dir = os.path.join(basedir, 'img')
labels_filename = 'labels.csv'


def extract_features_labels(n, mask_sw):
    """
    This funtion extracts the landmarks features for all images in the folder 'dataset/celeba'.
    It also extracts the shape label for each image.
    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        shape_labels:      an array containing the shape label (male=0 and female=1) for each image in
                            which a face was detected
    """
    image_paths = [os.path.join(images_dir, l) for l in os.listdir(images_dir)]
    labels_file = open(os.path.join(basedir, labels_filename), 'r')
    lines = labels_file.readlines()

    shape_labels = {line.split('\t')[0] : int(line.split('\t')[-3]) for line in lines[1:]}

    if os.path.isdir(images_dir):
        all_features = []
        all_labels = []
        for img_path in image_paths[:n]:
            file_name = img_path.split('.')[0].split('\\')[-1]
            # load image
            if mask_sw == 'on':
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                #mask = np.zeros(np.array(img).shape[:2], np.uint8)
                #mask[250:280, 170:330] = 1
                #mask_img = cv2.bitwise_and(img, img, mask=mask)

                all_features.append(img[250:280, 170:330])
                all_labels.append(shape_labels[file_name])
            else:
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)

                all_features.append(img)
                all_labels.append(shape_labels[file_name])


    landmark_features = np.array(all_features)
    return landmark_features, all_labels
