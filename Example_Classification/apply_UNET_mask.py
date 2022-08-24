import os
import cv2
import numpy as np
import tensorflow as tf
import time


def apply_UNET_mask(subdivided_image, model):

    # Appling the U-Net in the different sub-images
    subdivided_image = subdivided_image.astype(np.float64) / 255.0
    subdivided_UNET_image = model.predict(subdivided_image)
    subdivided_UNET_image *= 255
    subdivided_UNET_image = subdivided_UNET_image.astype(np.uint8)


    return subdivided_UNET_image
