#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
import os
import time
from sklearn.utils import shuffle
from IPython.display import clear_output
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import csv
import pickle
from cganfilter.models.video_filter import DeepNoisyBayesianFilter
from cganfilter.models.particle_filter import  ParticleFilterRect
from spo_dataset.spo_generator import get_video, get_dataset_from_video, get_dataset_from_image, generate_image, get_dataset_rotating_objects
import scipy.io
from common import train_relax, train_likelihood, train_predictor, train_update, normalize_image, cm_error, img_desc, mass_error
import skvideo.io
import matplotlib
from skimage import data
import cv2
import spo_dataset
from scipy.stats import wasserstein_distance
# ---- Aditional functions ---- #

def get_histogram(img):
  '''
  Get the histogram of an image. For an 8-bit, grayscale image, the
  histogram will be a 256 unit vector in which the nth value indicates
  the percent of the pixels in the image with the given darkness level.
  The histogram's values sum to 1.
  '''
  h, w = img.shape
  hist = [0.0] * 256
  for i in range(h):
    for j in range(w):
      hist[img[i, j]] += 1
  return np.array(hist) / (h * w) 

def add_border(img, border_size = 1, intense = 255):
    img_size = img.shape
    bigger_img = np.ones((img_size[0]+border_size*2, img_size[1]+border_size*2))*intense
    bigger_img[border_size:(border_size+img_size[0]) , border_size:(border_size+img_size[1])] = img
    return bigger_img





# ---- initialize parameters ---- #
hist = 4     
img_shape = (128,128) 
noise_rate = 0.2
n = 600
n_test = 300 
n_train = n -  n_test


# ---- Get the dataset ---- #
var = 0.1
x, z = get_dataset_rotating_objects(image_shape = img_shape, n = n, var = var)

        
x_test = x[:n_train]
z_test = z[:n_train]
x_train = x[n_train:]
z_train = z[n_train:] 

# ---- initialize particle filter ---- #

pf = ParticleFilterRect(Np = 2000,
                        var = var,
                        img_shape = img_shape,
                        beta = 60)

# ---- Test and viualize ---- #
x_old = x_test[:hist].copy()   
frames = []
obs_frames = []
state_frames = []
pf_frames = []
direct_frames = []
for t in range(0+hist,n_test-1):   
    z_new = z_test[t].copy() 
    z_new_test = z_test[t].copy()
    x_new = x_test[t].copy() 
    x_hat_pf = pf.step(z_new)
    plt.subplot(1,2,1)
    plt.imshow(x_hat_pf)
    plt.subplot(1,2,2)
    plt.imshow(x_new)
    plt.show()
    # cv2.imshow("real",normalize_image(x_new))
    # cv2.imshow("pf",normalize_image(x_hat_pf))
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break
    obs_frames.append(add_border(normalize_image(z_new)))       
    state_frames.append(add_border(normalize_image(x_new)))  
    pf_frames.append(add_border(normalize_image(x_hat_pf)))  
# cv2.destroyAllWindows()

# ---- Saves multiple samples as an image ---- #
idxs = np.arange(15,55,4, dtype = np.int16)
obs_img = np.concatenate(tuple(np.array(obs_frames)[idxs]),axis=1)
state_img = np.concatenate(tuple(np.array(state_frames)[idxs]),axis=1)
pf_img = np.concatenate(tuple(np.array(pf_frames)[idxs]),axis=1)
full_img = np.concatenate(( obs_img,state_img, df_img), axis = 0).astype(np.uint8)
matplotlib.image.imsave('samples_pf.png', full_img, cmap='gray')

# ---- Saves a video ---- #  
outputdata = np.array(frames).astype(np.uint8)    
skvideo.io.vwrite("samples.mp4", frames) 

# ---- Save Weights ---- #
df.save_weights('model_weights')

 
    

    
    
    
    
    

        
