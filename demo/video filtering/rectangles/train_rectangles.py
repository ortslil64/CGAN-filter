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
from spo_dataset.spo_generator import get_dataset_rotating_objects
import scipy.io
from cganfilter.common.common import train_relax, train_likelihood, train_predictor, train_update, normalize_image, cm_error, img_desc, mass_error
import skvideo.io
import matplotlib
from skimage import data
import cv2
import spo_dataset


gpus = tf.config.experimental.list_physical_devices('GPU')

# ---- Aditional functions ---- #

def add_border(img, border_size = 1, intense = 255):
    img_size = img.shape
    bigger_img = np.ones((img_size[0]+border_size*2, img_size[1]+border_size*2))*intense
    bigger_img[border_size:(border_size+img_size[0]) , border_size:(border_size+img_size[1])] = img
    return bigger_img





# ---- initialize parameters ---- #
hist = 4     
img_shape = (128,128) 
n = 1000
n_test = 300 
n_train = n -  n_test
# ---- Get the dataset ---- #

x, z = get_dataset_rotating_objects(image_shape = img_shape, n = n, var = 0.5, Ber = True, partial=True)
        
x_test = np.array(x[:n_test])
z_test = np.array(z[:n_test])
x_train = np.array(x[n_test:])
z_train = np.array(z[n_test:] )


# ---- Initialize ---- #
tf.keras.backend.clear_session()
df = DeepNoisyBayesianFilter(hist,img_shape)

# ---- Train ---- #
train_likelihood(df, x_train, z_train, epochs = 130) #100
train_predictor(df,x_train,epochs = 20, min_img = 2) #5
train_predictor(df,x_train,epochs = 30, min_img = 20) #10
train_update(df,x_train,z_train,epochs = 10, min_img = 2) #10
train_relax(df, x_train, z_train, epochs = 5,  min_img = 10) # 10
train_relax(df, x_train, z_train, epochs = 50,  min_img = 50) # 10
train_relax(df, x_train, z_train, epochs = 200,  min_img = None) # 10




# ---- Initialize testing arrays ---- #
cm_err_df = []  
mass_err_df = []  
img_err_df = []  
 
cm_err_direct = []  
mass_err_direct = [] 
img_err_direct = []

frames = []
obs_frames = []
state_frames = []
df_frames = []
direct_frames = []

# ---- Test and viualize ---- #
x_old = x_test[0:hist,...].copy()   

for t in range(0+hist,n_test-1):   
    z_new = z_test[t].copy() 
    z_new_test = z_test[t].copy()
    x_new = x_test[t].copy() 
    #x_hat_pf = pf.step(z_new)
    x_hat_df = df.predict_mean(x_old, z_new)
    x_hat_df = x_hat_df[:,:,0]
    #x_hat_df[x_hat_df<0.5] = 0
    x_hat_df_like = df.estimate(z_new_test)
    x_hat_df_like = x_hat_df_like[0,:,:,0]  
    #x_hat_df_like[x_hat_df_like<0.5] = 0
    x_old[:-1,:,:] = x_old[1:,:,:]
    x_old[-1,:,:] = x_hat_df   
    obs_frames.append(add_border(normalize_image(z_new)))       
    state_frames.append(add_border(normalize_image(x_new)))  
    df_frames.append(add_border(normalize_image(x_hat_df)))  
    direct_frames.append(add_border(normalize_image(x_hat_df_like)))  
    frame1 = np.concatenate((normalize_image(x_new),normalize_image(z_new)),axis = 1)
    frame2 = np.concatenate((normalize_image(x_hat_df),normalize_image(x_hat_df_like) ),axis = 1)
    frame = np.concatenate((frame1,frame2),axis = 0)
    frames.append(frame)

# ---- Saves multiple samples as an image ---- #
idxs = np.arange(0,200,5, dtype = np.int16)
obs_img = np.concatenate(tuple(np.array(obs_frames)[idxs]),axis=1)
state_img = np.concatenate(tuple(np.array(state_frames)[idxs]),axis=1)
df_img = np.concatenate(tuple(np.array(df_frames)[idxs]),axis=1)
direct_img = np.concatenate(tuple(np.array(direct_frames)[idxs]),axis=1)
full_img = np.concatenate(( obs_img,state_img, df_img, direct_img), axis = 0).astype(np.uint8)
matplotlib.image.imsave('t.png', full_img, cmap='gray')

# ---- Saves a video ---- #  
outputdata = np.array(frames).astype(np.uint8)    
skvideo.io.vwrite("samples.mp4", frames) 

# ---- Save Weights ---- #
df.save_weights('model_weights_rectangles_very_noisy')

 
    

    
    
    
    
    

        
