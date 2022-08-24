import cv2
import os
from tensorflow import keras
import numpy as np

# Defining additional functions
from divide_image import divide_image
from apply_UNET_mask import apply_UNET_mask
from recreate_UNET_image import recreate_UNET_image
from classify_contours_CNN import classify_contours

"""
    This example:
    1) Reads images from a pre-defined folder;
    2) Apply a trained UNET model to generate binary image;
    3) Identify the contours within the U-NET;
    4) Classify the contours as valid and invalid drops with the CNN model;
"""

# Defining the U-Net model and parameters
UNET_model_file = "../Models/UNET_best_model_large.h5"
UNET_window_size     = 1536 // 4 # sliding window size
UNET_stride_division = 8         # sliding window overlap level
UNET_bin_value       = 200

# Defining the CNN model and parameters
CNN_model_file = "../Models/CNN_best_model_32_32.h5"
CNN_input_image_size = 32 # size of the CNN image

# Defining the image folder
image_files_folder = "../Input"

# Loading the image files
image_files = []
for _file in os.listdir(image_files_folder):
    if _file.endswith("jpg"):
        image_file = os.path.join(image_files_folder, _file)
        image_files.append(image_file)

# Loading the U-Net and CNN models
UNET_model = keras.models.load_model(UNET_model_file, compile=False)
CNN_model  = keras.models.load_model(CNN_model_file, compile=False)

# Looping over the images
for image_file in image_files:

    # Loading the image
    image_rgb = cv2.imread(image_file, 1)

    # Dividing the full image into small sub-images - Sliding window approach
    subdivided_images  = divide_image(UNET_window_size, image_rgb, UNET_stride_division)

    # Applying the U-Net mask model to the sub-images
    subdivided_image_UNET = apply_UNET_mask(subdivided_images, UNET_model)

    # Recreating a full U-Net image from small U-Net  sub-images
    UNET_image = recreate_UNET_image(subdivided_image_UNET, UNET_window_size, image_rgb, UNET_stride_division)

    # Creating a binary image from the U-Net file
    UNET_image_bin = UNET_image.copy()
    UNET_image_bin[UNET_image_bin >= UNET_bin_value] = 255
    UNET_image_bin[UNET_image_bin < UNET_bin_value]  = 0

    # Finding contours in the binary image
    contours, retr = cv2.findContours(UNET_image_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Classifying the labels with the CNN model
    contour_labels = classify_contours(CNN_input_image_size, image_rgb, contours, CNN_model)

    # Drawing a 'drawed contour' image
    image_rgb_overlap = image_rgb.copy()
    for k, cnt in enumerate(contours):
        if contour_labels[k]:
            color = (0,255,0) # if valid, draw green contours
        else:
            color = (0,0,255) # if invalid, draw red contours
        cv2.drawContours(image_rgb_overlap, [cnt], -1, color, 2)

    # Saving the U-Net image to a png file
    basename = 'output_%s' % image_file.split('/')[-1]
    UNET_image_rgb = cv2.cvtColor(UNET_image, cv2.COLOR_GRAY2BGR)
    image_output = np.hstack((image_rgb, UNET_image_rgb, image_rgb_overlap))
    cv2.imwrite(basename, image_output)


