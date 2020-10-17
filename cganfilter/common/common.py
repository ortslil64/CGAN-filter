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
import cv2
from skimage import data
import skvideo.io

def KL_img(img1, img2, bins = 100):
    H1,_ = np.histogram(img1.ravel(),bins,(0,1)) 
    H2,_ = np.histogram(img2.ravel(),bins,(0,1)) 
    H1 = H1/np.sum(H1)
    H2 = H2/np.sum(H1)
    D = np.sum(H1*H2)
    return D

def img_desc(img1, img2):
    img_shape = img1.shape
    img1[img1>0.5] = 1
    img1[img1<=0.5] = 0
    img2[img2>0.5] = 1
    img2[img2<=0.5] = 0
    e = np.sum(np.logical_xor(img1, img2))/np.prod(img_shape)
    return e
        
def cm_error(img1, img2):
    img1[img1>0.5] = 1
    img1[img1<=0.5] = 0
    img2[img2>0.5] = 1
    img2[img2<=0.5] = 0
    img1_M = cv2.moments(img1)
    img1_cX = int(img1_M["m10"]/(img1_M["m00"]+1))  
    img1_cY = int(img1_M["m01"]/(img1_M["m00"]+1))
    
    
    img2_M = cv2.moments(img2)
    img2_cX = int(img2_M["m10"] / (img2_M["m00"]+1))
    img2_cY = int(img2_M["m01"] / (img2_M["m00"]+1))
    mse = (img2_cX - img1_cX)**2 + (img2_cY - img1_cY)**2
    return mse

def mass_error(img1, img2):
    img1[img1>0.5] = 1
    img1[img1<=0.5] = 0
    img2[img2>0.5] = 1
    img2[img2<=0.5] = 0
    img1_M = cv2.moments(img1)
    img1_cX = int(img1_M["m10"])  
    img1_cY = int(img1_M["m01"])
    
    
    img2_M = cv2.moments(img2)
    img2_cX = int(img2_M["m10"] )
    img2_cY = int(img2_M["m01"] )
    mse = (img2_cX - img1_cX)**2 + (img2_cY - img1_cY)**2
    return mse

def similarity(img1, img2):
    #img1[img1>0.5] = 1
    #img1[img1<0] = 0
    #img2[img2>0.5] = 1
    #img2[img2<0] = 0
    img1 = normalize_image(img1)/255.0
    img2 = normalize_image(img2)/255.0
    d = img1-img2
    dsqr = np.power(d,2)
    return np.sum(dsqr)
    
    
    



def normalize_image(x):
    cmax = np.max(x)
    cmin = np.min(x)
    cscale = cmax - cmin
    x_out = (x * 1.0 - cmin) / cscale 
    x_out = np.cast[np.float32](x_out)
    return x_out*255.0

def train_update(model, x_train, z_train,epochs = 10, min_img = 2):
    hist = model.hist
    n_train = len(x_train)
    for epoch in tqdm(range(epochs)):           
        for t in range(min_img+hist,n_train-1,min_img):
        # ----- Train on the current history ---- #
            for itr in range(t-min_img, t):
                z_new = z_train[itr].copy()
                x_new = x_train[itr].copy()
                x_old = x_train[itr-hist:itr].copy()
                model.train_updator(x_old,x_new,z_new)







def train_predictor(model, x_train, epochs = 5, min_img = 2,  x_test = None, relax = False):
    n_train = len(x_train)
    hist = model.hist
    if relax == False:
        for epoch in tqdm(range(epochs)):
            for t in range(min_img+hist,n_train-1):
                x_new = x_train[t].copy()
                x_old = x_train[t-hist:t].copy()
                model.train_predictor(x_old, x_new)
            if x_test is not None:
                idx = np.random.choice(len(x_test)-hist) + hist
                x_new_test = x_test[idx].copy()
                x_old_test = x_test[idx-hist:idx].copy()
                x_hat = model.propogate(x_old_test)
                x_hat = x_hat[0,:,:,0]
                fig, ((ax1, ax2)) = plt.subplots(1, 2)
                ax1.imshow(x_hat)
                ax1.set_title("x_hat predicted")
                ax2.imshow(x_new_test)
                ax2.set_title("ground truth (x)")
                fig.show()
                plt.show()
    else:
        n_train = len(x_train)
        for epoch in tqdm(range(epochs)):
            for t in range(min_img+hist,n_train-1,min_img):
            # ----- Train on the current history ---- #
                x_old = x_train[t-min_img-hist:t-min_img].copy()
                for itr in range(t-min_img, t):
                    x_new = x_train[itr].copy()
                    model.train_predictor(x_old, x_new)
                    x_hat_df_pre = df.propogate(x_old)
                    x_hat_df_pre = x_hat_df_pre[0,:,:,0]
                    x_old[:-1,:,:] = x_old[1:,:,:]
                    x_old[-1,:,:] = x_hat_df_pre
            if x_test is not None:
                idx = np.random.choice(len(x_test)-hist) + hist
                x_new_test = x_test[idx].copy()
                x_old_test = x_test[idx-hist:idx].copy()
                x_hat = model.propogate(x_old_test)
                x_hat = x_hat[0,:,:,0]
                fig, ((ax1, ax2)) = plt.subplots(1, 2)
                ax1.imshow(x_hat)
                ax1.set_title("x_hat predicted")
                ax2.imshow(x_new_test)
                ax2.set_title("ground truth (x)")
                fig.show()
                plt.show()

        
            

            
def train_likelihood(model, x_train, z_train,epochs = 120,  x_test = None, z_test = None):
    n_train = len(x_train)
    for epoch in tqdm(range(epochs)):
        for t in range(n_train):
            z_new = z_train[t].copy()
            x_new = x_train[t].copy()
            model.train_likelihood(z_new, x_new)
        if x_test is not None and z_test is not None:
            idx = np.random.choice(len(x_test)) 
            z_new_test = z_test[idx].copy()
            x_new_test = x_test[idx].copy()
            x_hat_df_like = df.estimate(z_new_test)
            x_hat_df_like = x_hat_df_like[0,:,:,0]
            fig, ((ax1, ax2)) = plt.subplots(1, 2)
            ax1.imshow(x_hat_df_like)
            ax1.set_title("x_hat update")
            ax2.imshow(x_new_test)
            ax2.set_title("ground truth (x)")
            fig.show()
            plt.show()
            
            


def train_relax(model, x_train, z_train, epochs = 120,  min_img = None):
    hist = model.hist
    n_train = len(x_train)
    for epoch in tqdm(range(epochs)):
        if min_img is None:
             x_old = x_train[:hist].copy()
             for t in range(hist,n_train-1):
            # ----- Train on the current history ---- #
                z_new = z_train[t].copy()
                x_new = x_train[t].copy()
                model.train_predictor(x_old, x_new)
                model.train_updator(x_old,x_new,z_new)
                x_hat_df = model.predict_mean(x_old, z_new)
                x_hat_df = x_hat_df[:,:,0]
                x_old[:-1,:,:] = x_old[1:,:,:]
                x_old[-1,:,:] = x_hat_df
        else:        
            for t in range(min_img+hist,n_train-1,min_img):
            # ----- Train on the current history ---- #
                i = np.random.randint(t-25,t+25)
                if i-min_img-hist < 0: i = 0
                x_old = x_train[t-min_img-hist:t-min_img].copy()
                for itr in range(t-min_img, t):
                    z_new = z_train[itr].copy()
                    x_new = x_train[itr].copy()
                    
                    model.train_predictor(x_old, x_new)
                    model.train_updator(x_old,x_new,z_new)
                    x_hat_df = model.predict_mean(x_old, z_new)
                    x_hat_df = x_hat_df[:,:,0]
                    x_old[:-1,:,:] = x_old[1:,:,:]
                    x_old[-1,:,:] = x_hat_df
                    


 