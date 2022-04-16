import sys

import numpy as np
from scipy import ndimage
from skimage import io, morphology
import pydicom
import os
import cv2


def transform_to_hu(medical_image, image):
    intercept = medical_image.RescaleIntercept
    slope = medical_image.RescaleSlope

    hu_image = image * slope + intercept

    return hu_image


def window_image(image, window_center, window_width):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    window_image = image.copy()
    window_image[window_image < img_min] = img_min
    window_image[window_image > img_max] = img_max

    return window_image





def remove_noise(file_path, x, y):
    medical_image = pydicom.read_file(file_path)
    image = medical_image.pixel_array

    hu_image = transform_to_hu(medical_image, image)
    # brain_image = window_image(hu_image, 65, 130)  # liver windowing
    brain_image = window_image(hu_image, x, y)  # liver windowing
    # brain_image = window_image(hu_image, 60, 400)  # bone windowing
    # cv2.imwrite("reza2.jpg", brain_image)
    segmentation = morphology.dilation(brain_image, np.ones((1, 1)))
    labels, label_nb = ndimage.label(segmentation)

    label_count = np.bincount(labels.ravel().astype(np.int))
    label_count[0] = 0

    mask = labels == label_count.argmax()

    mask = morphology.dilation(mask, np.ones((1, 1)))
    mask = ndimage.morphology.binary_fill_holes(mask)
    mask = morphology.dilation(mask, np.ones((3, 3)))
    masked_image = mask * brain_image

    return masked_image


def dicom_to_changeable_hu(source_dir, output_dir , hu , win):
  #  create_dir(output_dir)
    list_of_dicom_files = os.listdir(source_dir)
    # print(len(list_of_dicom_files))
    for file in list_of_dicom_files:
        print(source_dir + "/" + file)
        images = remove_noise(os.path.join(source_dir, file), hu, win)
        cv2.imwrite(os.path.join(output_dir, file) + '.jpg', images)
    print("HU finished !!!")




source_dir_path=sys.argv[1]
output_dir_path=sys.argv[2]

dicom_to_changeable_hu(source_dir_path, output_dir_path , 80 , 512)

