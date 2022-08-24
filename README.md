# dispersed-flow-classifier-and-generator

The following code repository contains example scripts on how to use and test a few of the different methodologies described in our Particle Tracking Velocimetry (PTV) algorithm for the analysis of oil drops behaviour in two-phase oil-water dispersions. 

# Example_Classification
This folder contains an example script code used to identify oil drops in two-phase flow dispersed images from https://doi.org/10.1016/j.expthermflusci.2019.03.009.
It follows the U-NET + CNN approach described in our work.

# Example_GAN
This folder contains a sample script showing how to produce synthetic images from the GAN methodology described in our work.
It can be used to generate a synthetic image dataset in our sample U-Net Transfer learning tool.

# Example_UNET_TransferLearning
This folder contains a sample script to: 1) Setup and manipulate the image data, for instance, generated from "Example_GAN" folder and 2) Run our Transfer-Learning tool based on our best-ranked U-Net on new images.

# Input
Input images used in the present examples. Images extracted from the experimental setup from https://doi.org/10.1016/j.expthermflusci.2019.03.009.

# Models
Best-ranked U-NET, CNN and GAN models trained in our work.

# Reference
The related article can be found at XXXXXXXXXXXXXXXXX
