import os
import cv2
import numpy as np
import tensorflow as tf

def classify_contours(img_input_size, image_rgb, contours, model_CNN, DIAM_1_THRESH=8, CNN_THRESH=0.5):

    # We only need the first channel
    img_raw = image_rgb[:, :, 0]

    # Total number of contours
    N_contours = len(contours)
    img_keras = np.zeros((N_contours,img_input_size,img_input_size,1), dtype=np.uint8)

    # Creating the droplet flags
    small_droplet_flags = np.zeros(N_contours, dtype=bool)
    small_droplet_flags[...] = False

    # Looping over all the contours
    for k, cnt in enumerate(contours):

        # Extracting the countour bounding box image
        bbox = cv2.boundingRect(cnt)
        y_1 = bbox[0]
        y_2 = bbox[0] + bbox[2]
        x_1 = bbox[1]
        x_2 = bbox[1] + bbox[3]
        bbox_img_raw  = img_raw[x_1:x_2, y_1:y_2]

        # Copying the bounding box
        cnt_bbox = cnt.copy()
        cnt_bbox[:,0,0] -= y_1
        cnt_bbox[:,0,1] -= x_1

        # If we have a small contour, it 99% of chance it is valid
        if max(bbox[2], bbox[3]) < DIAM_1_THRESH:
            small_droplet_flags[k] = True
            continue

        # Finding the contour image foreground and background
        img_bbox_contour_foreground = np.zeros_like(bbox_img_raw)
        cv2.drawContours(img_bbox_contour_foreground, [cnt_bbox], -1, (255), -1)
        img_bbox_contour_background = cv2.bitwise_not(img_bbox_contour_foreground)

        img_bbox_foreground = cv2.bitwise_and(img_bbox_contour_foreground, bbox_img_raw)
        img_bbox_background = cv2.bitwise_and(img_bbox_contour_background, bbox_img_raw)

        img_bbox_contour_background_value = img_bbox_background.astype(float)
        img_bbox_contour_background_value[img_bbox_contour_foreground == 255] = np.nan
        avg_background_intensity = np.nanmean(img_bbox_contour_background_value)

        # In case it fails...
        if np.isnan(avg_background_intensity):
            # continue
            avg_background_intensity = 125

        img_bbox_background_artificial      = np.zeros_like(bbox_img_raw)
        img_bbox_background_artificial[...] = int(avg_background_intensity)
        img_bbox_background_artificial = cv2.bitwise_and(img_bbox_contour_background, img_bbox_background_artificial)

        ### Here you get the 4th option (img preproc + edges) as detailed in the paper
        img_class_CNN = img_bbox_background_artificial + img_bbox_foreground
        cv2.drawContours(img_class_CNN, [cnt_bbox], -1, (255), 1)
        img_class_CNN = cv2.resize(img_class_CNN, (img_input_size,img_input_size))
        img_keras[k,:,:,0] = img_class_CNN

    img_keras = img_keras.astype(np.float64)

    # Applying the CNN model in the contour images
    labels = model_CNN.predict(img_keras)
    droplet_flags = np.where(labels > CNN_THRESH, True, False)

    # Overring the CNN classificatio for small droplets
    droplet_flags[small_droplet_flags] = True

    # cv2.namedWindow('test', cv2.WINDOW_GUI_NORMAL)
    # cv2.imshow('test',img_bbox_contour_background)
    # cv2.waitKey(0)


    return droplet_flags
