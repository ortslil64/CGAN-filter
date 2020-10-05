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
from cganfilter.models.particle_filter import  ParticleFilter
from spo_dataset.spo_generator import get_video, get_dataset_from_video, get_dataset_from_image, generate_image
import scipy.io
from common import train_relax, train_likelihood, train_predictor, train_update, normalize_image, cm_error, img_desc, mass_error
import skvideo.io
import matplotlib
from skimage import data
import cv2
import spo_dataset



# ---- Aditional functions ---- #

def add_border(img, border_size = 1, intense = 255):
    img_size = img.shape
    bigger_img = np.ones((img_size[0]+border_size*2, img_size[1]+border_size*2))*intense
    bigger_img[border_size:(border_size+img_size[0]) , border_size:(border_size+img_size[1])] = img
    return bigger_img

def generate_dataset(img_shape, n = 100,video_path = None, image_path = None, image_type = None,  output_type = "images", output_folder = "dataset/images/dots/",  partial = False, mask = None):
    frames = []
    if mask is not None:
        mask = cv2.imread(mask,0)
        mask = cv2.resize(mask, img_shape,interpolation = cv2.INTER_AREA)
    if video_path is not None:
        images, _ = get_video(video_path, n)
        x,z = get_dataset_from_video(images, n)
    elif image_path is not None:
        image = cv2.imread(image_path,0)
        image = cv2.resize(image, img_shape,interpolation = cv2.INTER_AREA)
        x,z = get_dataset_from_image(image/255, n, radius = [20],  partial = partial, mask = mask, pose = [[10,10]], n_circles = 1, v = [[1,2]])
    elif image_type == "dots":
        image = generate_image(0.01)
        x,z = get_dataset_from_image(image, n, radius = [20],  partial = partial, mask = mask, pose = [[10,10]], n_circles = 1, v = [[1,2]])
    elif image_type == "checkers":
        image = np.array(data.checkerboard()).astype(np.float64)
        image = cv2.resize(image, img_shape,interpolation = cv2.INTER_AREA)
        x,z = get_dataset_from_image(image/255, n, radius = [20],  partial = partial, mask = mask, pose = [[10,10]], n_circles = 1, v = [[1,2]])
    
    return x, z




# ---- initialize parameters ---- #
hist = 4     
img_shape = (128,128) 
noise_rate = 0.2
n = 600
n_test = 300 
n_train = n -  n_test
# ---- Get the dataset ---- #

x, z = generate_dataset(img_shape, n = n,
                        image_type = "checkers")
x_test = x[:n_train]
z_test = z[:n_train]
x_train = x[n_train:]
z_train = z[n_train:] 


# ---- Initialize ---- #
tf.keras.backend.clear_session()
df = DeepNoisyBayesianFilter(hist,img_shape)

# ---- Train ---- #
 
train_likelihood(df, x_train, z_train, epochs = 130) #100
train_predictor(df,x_train,epochs = 10, min_img = 2) #5
train_predictor(df,x_train,epochs = 15, min_img = 20) #10
train_update(df,x_train,z_train,epochs = 10, min_img = 2) #10
train_relax(df, x_train, z_train, epochs = 5,  min_img = 10) # 10
train_relax(df, x_train, z_train, epochs = 50,  min_img = 50) # 10
train_relax(df, x_train, z_train, epochs = 50,  min_img = None) # 10



# ---- Test and viualize ---- #
x_old = x_test[:hist,...].copy()   
frames = []
obs_frames = []
state_frames = []
df_frames = []
direct_frames = []
for t in range(0+hist,n_test-1):   
    z_new = z_test[t].copy() 
    z_new_test = z_test[t].copy()
    x_new = x_test[t].copy() 
    #x_hat_pf = pf.step(z_new)
    x_hat_df = df.predict_mean(x_old, z_new)
    x_hat_df = x_hat_df[:,:,0]
    x_hat_df_like = df.estimate(z_new_test)
    x_hat_df_like = x_hat_df_like[0,:,:,0]    
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
idxs = np.arange(15,200,4, dtype = np.int16)
obs_img = np.concatenate(tuple(np.array(obs_frames)[idxs]),axis=1)
state_img = np.concatenate(tuple(np.array(state_frames)[idxs]),axis=1)
df_img = np.concatenate(tuple(np.array(df_frames)[idxs]),axis=1)
direct_img = np.concatenate(tuple(np.array(direct_frames)[idxs]),axis=1)
full_img = np.concatenate(( obs_img,state_img, df_img, direct_img), axis = 0).astype(np.uint8)
matplotlib.image.imsave('samples.png', full_img, cmap='gray')

# ---- Saves a video ---- #  
outputdata = np.array(frames).astype(np.uint8)    
skvideo.io.vwrite("samples.mp4", frames) 

# ---- Save Weights ---- #
df.save_weights('model_weights_inverse_chekkers')

 
    

    
    
    
    
    

        
