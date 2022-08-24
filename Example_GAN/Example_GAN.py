import os
import numpy as np
import cv2
import keras
import matplotlib.pyplot as plt
import random
import string

from return_random_drop_diameters import return_random_drop_diameters

"""
   This example generates a sythetic dispersed oil-water two-phase flow dispersion image with
   a brigh bacgkround from a pre-trained GAN model. It is possible to modify the number of oil
   drops and the its DSD (Log-Normal Distribution). Two images are generated, one is the real image
   and the second is a segmented(or mask) image. Those image pairs are used in the UNET-Transfer Learning example.
"""

# Defining GAN parameters
GAN_image_size = (1536//2, 1536//2) # GAN image size
px_by_mm = (1382.0 / 111.4)       # px/mm relation
GAN_latent_dim = 100              # GAN latent dimension
GAN_input_size = 24               # GAN image size

# Defining DSD/drop parameters
N_classes = 18                                 # Number of drop DSD classes
d_classes = np.linspace(0.1,2.0,num=N_classes) # minimum and maximum drop diameters [mm]
N_drops = 200                                  # Number of drops
mu_DSD = 0.0                                   # mu DSD parameter
sigma_DSD = 1.0                                # sigma DSD parameter

# Log-Normal distribution
X_DSD = np.linspace(1.0e-6,6.0,num=200)
Y_DSD_1 = (np.log(X_DSD) - mu_DSD)**2.0
Y_DSD_2 = Y_DSD_1 / (2.0 * (sigma_DSD**2.0))
Y_DSD_3 = np.exp(-Y_DSD_2)
Y_DSD_4 = (X_DSD * sigma_DSD * np.sqrt(2.0 * np.pi))
Y_DSD = Y_DSD_3 / Y_DSD_4

# Returning the drop diameters
drop_diams = return_random_drop_diameters(np.c_[X_DSD, Y_DSD], N_drops)

## Uncomment block below to check the random drop diameters and the
## prescribed DSD
#plt.close()
#plt.plot(X_DSD, Y_DSD, color='red')
#plt.hist(drop_diams, bins=20, density=True)
#plt.show()


# Loading the generators. The 'naive' approach here
# is to use a different GAN for each drop class
d_diam = d_classes[1] - d_classes[0]
GAN_models = []
for d_class in range(N_classes-1):
    GAN_model = keras.models.load_model("../Models/GAN_models/class_%02d.h5" % d_class, compile=False)
    GAN_models.append(GAN_model)

# Creating the background image
background_img = np.zeros(GAN_image_size, dtype=np.uint8)
# Create a mask/segmented image pair
mask_img       = np.zeros(GAN_image_size, dtype=np.uint8)

# Plain background
#background_img[...] = 200

# Noisy background
noise = np.random.normal(255./2,255./10, GAN_image_size)
noise = noise.astype(np.uint8)
noise = cv2.blur(noise, (3,3))
background_img = noise

# Calculating the DSD class for the drop diameters
d_class_gens = (drop_diams - np.min(d_classes)) / d_diam
d_class_gens = d_class_gens.astype(int)
# Large drops goes to the last DSD class
d_class_gens[d_class_gens > (N_classes-1)] = (N_classes-1)

# Looping over the drops
for k in range(N_drops):
    # Retrieving GAN model
    dsd_class = d_class_gens[k]
    GAN_model = GAN_models[d_class]

    # Generating GAN image - naive implementation
    GAN_input_noise  = np.random.randn(1,GAN_latent_dim)
    GAN_image = GAN_model.predict(GAN_input_noise)
    GAN_image *= 127.5
    GAN_image += 127.5
    GAN_image = GAN_image.astype(np.uint8)
    GAN_image = GAN_image[0,:,:,0]

    # Dividing the image into foreground and background
    _, GAN_image_otsu = cv2.threshold(GAN_image.copy(),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)

    # Calculating drop diameter based on the projected area - Projection based area
    GAN_image_otsu_area = np.where(GAN_image_otsu > 100 , 1, 0)
    GAN_image_otsu_area = np.sum(GAN_image_otsu_area)
    d_esf_pix = np.sqrt(4.0 * float(GAN_image_otsu_area) / np.pi)

    # Resizing the GAN image-based drop to its real (drop_diams) diameter
    try:
        d_esf_pix_expected = drop_diams[k] * px_by_mm
        resizing_corr_factor = float(d_esf_pix_expected) / float(d_esf_pix)
        resize_shape = int(resizing_corr_factor * GAN_input_size)
        GAN_image = cv2.resize(GAN_image, (resize_shape, resize_shape))
    except:
        print('WARNING!')
        continue

    # Defining a random position for the GAN-based drop
    c_x_rand = np.random.randint(resize_shape + 1, GAN_image_size[0] - resize_shape - 1)
    c_y_rand = np.random.randint(resize_shape + 1, GAN_image_size[1] - resize_shape - 1)

    # Drawing the drop on top of the bacgkround image and mask image
    _, GAN_image_otsu = cv2.threshold(GAN_image.copy(),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU+cv2.THRESH_BINARY_INV)
    i_start = c_x_rand; i_end = (c_x_rand+resize_shape)
    j_start = c_y_rand; j_end = (c_y_rand+resize_shape)

    sub_img = background_img[i_start:i_end, j_start:j_end]
    sub_img[GAN_image_otsu == 255] = GAN_image[GAN_image_otsu==255]
    background_img[i_start:i_end, j_start:j_end] = sub_img

    sub_mask_img = mask_img[i_start:i_end, j_start:j_end]
    sub_mask_img[GAN_image_otsu == 255] = 255
    mask_img[i_start:i_end, j_start:j_end] = sub_mask_img

# Create a random name to store the image
random_file_name = (''.join(random.choices(string.ascii_lowercase, k=5)))

# Drawing the final images
cv2.imwrite('./GAN_DATASET/full_raw_imgs/%s.jpg'  % random_file_name, background_img)
cv2.imwrite('./GAN_DATASET/full_mask_imgs/%s.jpg' % random_file_name, mask_img)
