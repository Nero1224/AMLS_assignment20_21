import os
import numpy as np
import cv2
import shutil
import platform

# PATH TO ALL IMAGES
global basedir, image_paths, target_size

if platform.system().lower() == 'windows':
    print("Windows")
    basedir = os.path.join(os.path.dirname(os.getcwd()), r'Datasets\cartoon_set_test')
else:
    print("Linux")
    basedir = os.path.join(os.path.dirname(os.getcwd()), r'Datasets/cartoon_set_test')

img_path = os.path.join(basedir, r"img")
jpg_path = os.path.join(basedir, r"img_jpg")
if os.path.exists(jpg_path): pass
else:
    os.makedirs(jpg_path)
    for img in os.listdir(img_path):
        shutil.copyfile(os.path.join(img_path, r"%s" %img), os.path.join(jpg_path, "%s.jpg" %img.split(".")[0]))

images_dir = jpg_path
labels_filename = 'labels.csv'


def extract_features_labels(mask_sw):
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
        for img_path in image_paths:
            file_name = img_path.split('.')[0].split('\\')[-1]
            # load image
            if mask_sw == 'on':
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                all_features.append(img[250:280, 170:330])
                all_labels.append(shape_labels[file_name])
            else:
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)

                all_features.append(img)
                all_labels.append(shape_labels[file_name])

    landmark_features = np.array(all_features)
    return landmark_features, all_labels
