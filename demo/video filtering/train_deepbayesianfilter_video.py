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
from deepbayesianfilter.models.video_filter import DeepNoisyBayesianFilter
from deepbayesianfilter.dataset_generator.video_generator import get_dataset, generate_image
import cv2
from skimage import data
import skvideo.io

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
                    


            
            
if __name__ == '__main__':
         
    
    
    n = 600 
     
    noise_rate = 0.1
    image = generate_image(noise_rate)
    x, z = get_dataset(image,n)
    
    x_train = x[:400]
    x_test= x[400:]
    z_train = z[:400]
    z_test= z[400:]
    
    
    hist = 3     
    img_shape = (128,128)
    #tf.keras.backend.clear_session()
    
    df = DeepNoisyBayesianFilter(hist,img_shape)
    
    err_df = []
    err_kf = []
    MSE_df = []
    MSE_kf = []
    training_ep = 0
    max_training = 5000
    min_training = 2000
    
    
    train_likelihood(df, x_train, z_train, epochs = 120)
    train_predictor(df,x_train,epochs = 5, min_imgs = 2)
    train_predictor(df,x_train,epochs = 10, min_imgs = 20)
    train_update(df,x_train,z_train,epochs = 10, min_img = 2)
    train_relax(df, x_train, z_train, epochs = 10,  min_img = 30)
    train_relax(df, x_train, z_train, epochs = 100,  min_img = None)
            
    df.save_weights(".weights"+str(noise_rate))
     
    
    
    
    # df.load_weights(".weights"+str(noise_rate))
    
    err_mc_df = []
    err_mc_pred = []
    err_mc_direct = []
    for ii in range(30):
    
        x_old = x_test[ii:ii+hist,...].copy()   
        x_old_pred = x_test[ii:ii+hist,...].copy() 
        err_df = []
        err_direct = []   
        err_pred = []  
        frames = []
        n_test = 200 
        for t in range(0+hist,n_test-1):   
            z_new = z_test[t].copy() 
            z_new_test = z_test[t].copy()
            x_new = x_test[t].copy() 
            x_hat_df = df.predict_mean(x_old, z_new)
            x_hat_df = x_hat_df[:,:,0]
            x_hat_df_pre = df.propogate(x_old_pred)
            x_hat_df_pre = x_hat_df_pre[0,:,:,0]
            
            x_hat_df_like = df.estimate(z_new_test)
            x_hat_df_like = x_hat_df_like[0,:,:,0]
            
        
            
            cm_err_de = cm_error(x_new,x_hat_df)
            cm_err_de_pred = cm_error(x_new,x_hat_df_pre)
            cm_err_direct = cm_error(x_new,x_hat_df_like)
        
            
            x_old[:-1,:,:] = x_old[1:,:,:]
            x_old[-1,:,:] = x_hat_df
            
            x_old_pred[:-1,:,:] = x_old_pred[1:,:,:]
            x_old_pred[-1,:,:] = x_hat_df_pre
            
            err_df.append(cm_err_de)
            err_direct.append(cm_err_direct)
            err_pred.append(cm_err_de_pred)
            
            frame1 = np.concatenate((normalize_image(x_new),normalize_image(z_new),np.zeros((128,128))),axis = 1)
            frame2 = np.concatenate((normalize_image(x_hat_df),normalize_image(x_hat_df_like),normalize_image(x_hat_df_pre)),axis = 1)
            frame = np.concatenate((frame1,frame2),axis = 0)
            frames.append(frame)
            # print("DF predicted:" + str(img_desc(x_hat_df_pre,x_new)))
            # print("DF update:" + str(img_desc(x_hat_df,x_new)))
            # print("DF likelihood:" + str(img_desc(x_hat_df_like,x_new)))
        err_mc_df.append(np.mean(err_df))
        err_mc_pred.append(np.mean(err_pred))
        err_mc_direct.append(np.mean(err_direct))
        
    x_idxs = np.arange(len(err_mc_df))   
    plt.plot(x_idxs,err_mc_df, color='blue', linewidth=1.5, label = "DF")
    plt.plot(x_idxs,err_mc_pred, color='red', linewidth=1.5 , label = "prediction") 
    plt.plot(x_idxs,err_mc_direct, color='black', linewidth=1.5, label = "likelihood") 
    plt.xlabel("initial noise")
    plt.ylabel("avarage MSE")
    plt.legend()
        
    
    
    
    x_old = x_test[1:1+hist,...].copy()   
    x_old_pred = x_test[1:1+hist,...].copy() 
    err_df = []
    err_direct = []   
    err_pred = []  
    frames = []
    n_test = 200 
    for t in range(0+hist,n_test-1):   
        z_new = z_test[t].copy() 
        z_new_test = z_test[t].copy()
        x_new = x_test[t].copy() 
        x_hat_df = df.predict_mean(x_old, z_new)
        x_hat_df = x_hat_df[:,:,0]
        x_hat_df_pre = df.propogate(x_old_pred)
        x_hat_df_pre = x_hat_df_pre[0,:,:,0]
        
        x_hat_df_like = df.estimate(z_new_test)
        x_hat_df_like = x_hat_df_like[0,:,:,0]
        
    
        
        cm_err_de = img_desc(x_new,x_hat_df)
        cm_err_de_pred = img_desc(x_new,x_hat_df_pre)
        cm_err_direct = img_desc(x_new,x_hat_df_like)
    
        
        x_old[:-1,:,:] = x_old[1:,:,:]
        x_old[-1,:,:] = x_hat_df
        
        x_old_pred[:-1,:,:] = x_old_pred[1:,:,:]
        x_old_pred[-1,:,:] = x_hat_df_pre
        
        err_df.append(cm_err_de)
        err_direct.append(cm_err_direct)
        err_pred.append(cm_err_de_pred)
        
        frame1 = np.concatenate((normalize_image(x_new),normalize_image(z_new),np.zeros((128,128))),axis = 1)
        frame2 = np.concatenate((normalize_image(x_hat_df),normalize_image(x_hat_df_like),normalize_image(x_hat_df_pre)),axis = 1)
        frame = np.concatenate((frame1,frame2),axis = 0)
        frames.append(frame)
    
    outputdata = np.array(frames).astype(np.uint8)    
    skvideo.io.vwrite(str(noise_rate)+".mp4", frames) 
    x_idxs = np.arange(len(err_df))    
    
        
    plt.plot(x_idxs,err_df, color='blue', linewidth=1.5, label = "DF")
    plt.plot(x_idxs,err_pred, color='red', linewidth=1.5 , label = "prediction") 
    plt.plot(x_idxs,err_direct, color='black', linewidth=1.5, label = "likelihood") 
    plt.xlabel("t")
    plt.ylabel("MSE")
    plt.xlim(0,195)
    #plt.ylim(-1,300)
    #plt.legend()
        
        
        
        
    

        
