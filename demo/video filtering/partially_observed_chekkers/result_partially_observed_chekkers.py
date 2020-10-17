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
from cganfilter.models.particle_filter import  ParticleFilter_deep
from spo_dataset.spo_generator import get_video, get_dataset_from_video, get_dataset_from_image, generate_image
import scipy.io
from cganfilter.common.common import train_relax, train_likelihood, train_predictor, train_update, normalize_image, cm_error, img_desc, mass_error
import skvideo.io
import matplotlib
from skimage import data
import cv2
import spo_dataset
# ---- Aditional functions ---- #

def KL_img(img1, img2, bins = 100):
    H1,_ = np.histogram(img1.ravel(),bins,(0,1)) 
    H2,_ = np.histogram(img2.ravel(),bins,(0,1)) 
    H1 = H1/np.sum(H1)
    H2 = H2/np.sum(H1)
    H1 = H1 + 0.001
    H1 = H1/np.sum(H1)
    H2 = H2 + 0.001
    H2 = H2/np.sum(H1)
    kl = -np.sum(H1 * (np.log(H1) - np.log(H2)))
    return kl

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
        x,z = get_dataset_from_image(image/255, n, radius = [18],  partial = partial, mask = mask, pose = [[20,10]], n_circles = 1, v = [[1,2]])
    elif image_type == "dots":
        image = generate_image(0.01)
        x,z = get_dataset_from_image(image, n, radius = [18],  partial = partial, mask = mask, pose = [[20,10]], n_circles = 1, v = [[1,2]])
    elif image_type == "checkers":
        image = np.array(data.checkerboard()).astype(np.float64)
        image = cv2.resize(image, img_shape,interpolation = cv2.INTER_AREA)
        x,z = get_dataset_from_image(image/255, n, radius = [18],  partial = partial, mask = mask, pose = [[20,10]], n_circles = 1, v = [[1,2]])
    
    return x, z





# ---- initialize parameters ---- #
hist = 4     
img_shape = (128,128) 
noise_rate = 0.2
n = 600
n_test = 300 
n_train = n -  n_test
# ---- Get the dataset ---- #

x, z = generate_dataset(img_shape = img_shape,
                        n = n,
                        image_type = "checkers",
                        partial=True)
        
x_test = x[:n_train]
z_test = z[:n_train]
x_train = x[n_train:]
z_train = z[n_train:] 


# ---- Initialize ---- #
tf.keras.backend.clear_session()
df = DeepNoisyBayesianFilter(hist,img_shape)


# ---- Load weights ---- #
df.load_weights('model_weights_partially_observed_chekkers')




# ---- initialize particle filter ---- #
ref_img = np.array(data.checkerboard()).astype(np.float64)
ref_img = cv2.resize(ref_img, img_shape,interpolation = cv2.INTER_AREA)

pf = ParticleFilter_deep(Np = 10000,
                        No = 1,
                        ref_img = ref_img,
                        radiuses = [18],
                        initial_pose = [[20,10]],
                        beta = 30,
                        likelihood=df)

# ---- Test and viualize ---- #
x_old = x_test[:hist,...].copy()   
frames = []
obs_frames = []
state_frames = []
pf_frames = []
direct_frames = []
df_frames = []

cm_err_df = []  
mass_err_df = []  
img_err_df = []  
img_kl_df = []

cm_err_pf = []  
mass_err_pf = []  
img_err_pf = []  
img_kl_pf = []
 
cm_err_direct = []  
mass_err_direct = [] 
img_err_direct = []
img_kl_direct = []

for t in range(0+hist,n_test-1):   
    z_new = z_test[t].copy() 
    z_new_test = z_test[t].copy()
    x_new = x_test[t].copy() 
    x_hat_pf = pf.step(z_new)
    x_hat_df = df.predict_mean(x_old, z_new)
    x_hat_df = x_hat_df[:,:,0]
    x_hat_df[x_hat_df<0.5] = 0
    x_hat_df_like = df.estimate(z_new_test)
    x_hat_df_like = x_hat_df_like[0,:,:,0] 
    x_hat_df_like[x_hat_df_like<0.5] = 0
    x_old[:-1,:,:] = x_old[1:,:,:]
    x_old[-1,:,:] = x_hat_df 
    
    cm_err_df.append(cm_error(x_new, x_hat_df)) 
    mass_err_df.append(mass_error(x_new, x_hat_df))  
    img_err_df.append(img_desc(x_new, x_hat_df)) 
    img_kl_df.append(KL_img(x_new, x_hat_df))
    
    cm_err_pf.append(cm_error(x_new, x_hat_pf)) 
    mass_err_pf.append(mass_error(x_new, x_hat_pf))  
    img_err_pf.append(img_desc(x_new, x_hat_pf)) 
    img_kl_pf.append(KL_img(x_new, x_hat_pf))
    
    cm_err_direct.append(cm_error(x_new, x_hat_df_like)) 
    mass_err_direct.append(mass_error(x_new, x_hat_df_like))  
    img_err_direct.append(img_desc(x_new, x_hat_df_like))
    img_kl_direct.append(KL_img(x_new, x_hat_df_like))
    
    pf_frames.append(add_border(normalize_image(x_hat_pf)))  
    obs_frames.append(add_border(normalize_image(z_new)))       
    state_frames.append(add_border(normalize_image(x_new)))  
    df_frames.append(add_border(normalize_image(x_hat_df)))  
    direct_frames.append(add_border(normalize_image(x_hat_df_like)))  
    
    frame1 = np.concatenate((normalize_image(x_new),normalize_image(z_new), np.zeros_like(z_new)),axis = 1)
    frame2 = np.concatenate((normalize_image(x_hat_df),normalize_image(x_hat_df_like) ,normalize_image(x_hat_pf) ),axis = 1)
    frame = np.concatenate((frame1,frame2),axis = 0)
    frames.append(frame)
    
    plt.subplot(1,2,1)
    plt.imshow(x_hat_pf)
    plt.subplot(1,2,2)
    plt.imshow(x_new)
    plt.show()
    
    
# ---- Saves multiple samples as an image ---- #
idxs = np.arange(0,99,4, dtype = np.int16)
obs_img = np.concatenate(tuple(np.array(obs_frames)[idxs]),axis=1)
state_img = np.concatenate(tuple(np.array(state_frames)[idxs]),axis=1)
pf_img = np.concatenate(tuple(np.array(pf_frames)[idxs]),axis=1)
df_img = np.concatenate(tuple(np.array(df_frames)[idxs]),axis=1)
direct_img = np.concatenate(tuple(np.array(direct_frames)[idxs]),axis=1)
full_img = np.concatenate(( obs_img,state_img, df_img,direct_img, pf_img ), axis = 0).astype(np.uint8)
matplotlib.image.imsave('samples3.png', full_img, cmap='gray')

# ---- Saves a video ---- #  
outputdata = np.array(frames).astype(np.uint8)    
skvideo.io.vwrite("samples.mp4", frames) 

# ---- Visualize errors ---- #
plt.figure(1)
plt.plot(cm_err_df, c='blue')
plt.plot(cm_err_pf, c='red')
plt.plot(cm_err_direct, c='green')
plt.show()

plt.figure(2)
plt.plot(mass_err_df, c='blue')
plt.plot(mass_err_pf, c='red')
plt.plot(mass_err_direct, c='green')
plt.show()

plt.figure(3)
plt.plot(img_err_df, c='blue')
plt.plot(img_err_pf, c='red')
plt.plot(img_err_direct, c='green')
plt.show()

plt.figure(4)
plt.plot(img_kl_df, c='blue')
plt.plot(img_kl_pf, c='red')
plt.plot(img_kl_direct, c='green')
plt.show()

# ---- Save error statistics ---- #

scipy.io.savemat('partial_observation_tree_data.mat', mdict={'cm_err_df': np.array(cm_err_df),
                                                             'cm_err_pf': np.array(cm_err_pf),
                                                             'cm_err_direct': np.array(cm_err_direct),
                                                             'img_err_df': np.array(img_err_df),
                                                             'img_err_pf': np.array(img_err_pf),
                                                             'img_err_direct': np.array(img_err_direct),
                                                             'mass_err_df': np.array(mass_err_df),
                                                             'mass_err_pf': np.array(mass_err_pf),
                                                             'mass_err_direct': np.array(mass_err_direct)})



 
    

    
    
    
    
    

        
