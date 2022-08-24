import os
import cv2
import numpy as np
import tensorflow as tf
import time


def divide_image(window_size, image_rgb, stride_division=2):

    # Only a single image channel
    raw_img = image_rgb[:, :, 0]

    # Number of image subdivisions
    N_I = raw_img.shape[0] // window_size + 1
    N_J = raw_img.shape[1] // window_size + 1

    # Sliding window passes start position
    scan_I = np.arange(0, raw_img.shape[0] - window_size + 1, window_size // stride_division)
    scan_J = np.arange(0, raw_img.shape[1] - window_size + 1, window_size // stride_division)

    # Matrix where the subdivided image is stored
    raw_img_subdivided = np.zeros((scan_I.shape[0] * scan_J.shape[0], window_size, window_size, 1), dtype=np.float64)

    # Applying the sliding window
    index = 0
    for i in scan_I:
        for j in scan_J:
            i_start = i
            i_end   = i + window_size
            j_start = j
            j_end   = j + window_size


            # Storing the sliding window image in a given pass
            # and storing the image into the matrix
            window_image = raw_img[i_start:i_end, j_start:j_end]
            raw_img_subdivided[index, :, :, 0] = window_image

            index += 1

    return raw_img_subdivided
