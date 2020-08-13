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
from spo_dataset.spo_generator import get_video, get_dataset_from_video, get_dataset_from_image, generate_image
import scipy.io
from train_deepbayesianfilter_video import train_relax, train_likelihood, train_predictor, train_update, normalize_image, cm_error, img_desc, mass_error
import skvideo.io
import matplotlib
from skimage import data
import cv2

def add_border(img, border_size = 1, intense = 255):
    img_size = img.shape
    bigger_img = np.ones((img_size[0]+border_size*2, img_size[1]+border_size*2))*intense
    bigger_img[border_size:(border_size+img_size[0]) , border_size:(border_size+img_size[1])] = img
    return bigger_img
    
         
err_df = []
err_direct = []
hist = 4     
img_shape = (128,128) 

noise_rate = 0.2



n = 1000
n_test = 700 
n_train = n -  n_test
image = cv2.imread('source_image/tree.jpg',0)
mask = cv2.imread('source_image/tree_masked.jpg',0)
image = cv2.resize(image, img_shape,interpolation = cv2.INTER_AREA)/255.0
mask = cv2.resize(mask, img_shape ,interpolation = cv2.INTER_AREA)
x, z = get_dataset(image,n_train,n_circles = 1, radius = [25], v = [[1,1]], pose = [[30,0]])
# video_path = 'source_video/illusion.mp4'
# images, _ = get_video(video_path, n)
# z,x = get_dataset(images, n, n_circles = 2, radius = [14, 16])
# image = data.checkerboard()/255.0
# image = cv2.resize(image, (128,128),interpolation = cv2.INTER_AREA)
# x, z = get_dataset(image,n,n_circles = 1, radius = [13], v = [[1,1]], pose = [[10,0]], partial = True)
x_train = x
z_train = z

tf.keras.backend.clear_session()

df = DeepNoisyBayesianFilter(hist,img_shape)
 
train_likelihood(df, x_train, z_train, epochs = 130) #100
train_predictor(df,x_train,epochs = 10, min_img = 2) #5
train_predictor(df,x_train,epochs = 15, min_img = 20) #10
train_update(df,x_train,z_train,epochs = 10, min_img = 2) #10
train_relax(df, x_train, z_train, epochs = 5,  min_img = 10) # 10
train_relax(df, x_train, z_train, epochs = 50,  min_img = 50) # 10
train_relax(df, x_train, z_train, epochs = 50,  min_img = None) # 10

#df.save_weights(".weights"+str(noise_rate))

# df.load_weights(".weights"+str(noise_rate))
x, z = get_dataset(image,n_test,n_circles = 1, radius = [25], v = [[1,1]], pose = [[30,0]], mask = mask)
x_test= x
z_test= z


x_old = x_test[:hist,...].copy() 


     
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
for t in range(0+hist,n_test-1):   
    z_new = z_test[t].copy() 
    z_new_test = z_test[t].copy()
    x_new = x_test[t].copy() 
    x_hat_df = df.predict_mean(x_old, z_new)
    x_hat_df = x_hat_df[:,:,0]
    x_hat_df_like = df.estimate(z_new_test)
    x_hat_df_like = x_hat_df_like[0,:,:,0]    

    cm_err_df.append(cm_error(x_new, x_hat_df)) 
    mass_err_df.append(mass_error(x_new, x_hat_df))  
    img_err_df.append(img_desc(x_new, x_hat_df)) 
    
    cm_err_direct.append(cm_error(x_new, x_hat_df_like)) 
    mass_err_direct.append(mass_error(x_new, x_hat_df_like))  
    img_err_direct.append(img_desc(x_new, x_hat_df_like))
                 
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
    
idxs = np.arange(0,140,10, dtype = np.int16)
obs_img = np.concatenate(tuple(np.array(obs_frames)[idxs]),axis=1)
state_img = np.concatenate(tuple(np.array(state_frames)[idxs]),axis=1)
df_img = np.concatenate(tuple(np.array(df_frames)[idxs]),axis=1)
direct_img = np.concatenate(tuple(np.array(direct_frames)[idxs]),axis=1)
full_img = np.concatenate(( obs_img,state_img, df_img, direct_img), axis = 0)
matplotlib.image.imsave('partial_observation.png', full_img, cmap='gray')
  
outputdata = np.array(frames).astype(np.uint8)    
skvideo.io.vwrite("partial_observation_output.mp4", frames) 

scipy.io.savemat('partial_observation_data.mat', mdict={'cm_err_df': np.array(cm_err_df),
                                       'cm_err_direct': np.array(cm_err_direct),
                                       'img_err_df': np.array(img_err_df),
                                       'img_err_direct': np.array(img_err_direct),
                                       'mass_err_df': np.array(mass_err_df),
                                       'mass_err_direct': np.array(mass_err_direct)})


 
    

    
    
    
    
    

        
