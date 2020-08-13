from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from cganfilter.models.time_serias_filter import DeepNoisyBayesianFilter, DeepNoisyBayesianFilterLinearPredictor, TCNBayesianFilter, DeepNoisyBayesianFilter1D
from time_series_generator import random_walk
from cganfilter.legacy_filters.kalman_filter import Kalman_smoother
import tensorflow as tf
import scipy.io
TF_ENABLE_GPU_GARBAGE_COLLECTION=False
        
def smooth(x):
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i-99)
        y[i] = float(x[start:(i+1)].sum())/(i-start+1)
    return y

# ---- Initialization ---- #
df_samples = []
kf_samples = []
real_samples = []
obs_samples = []
idxs_samples = []
err_df = []
err_kf = []
MSE_df = []
MSE_kf = []
u = None
# ---- Parameters ---- #
plot_flag = True
measure_gap = 100
img_shape = (64,64)
min_img = 20
skips = img_shape[0]*img_shape[1]//2
n_crop = 100
min_training =  2000
mc_runs = 100

# ---- Monte-Carlo Simulation ---- # 
for mc_run in range(mc_runs):
    training_ep = 0
    tf.keras.backend.clear_session()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.set_memory_growth(gpus[1], True)
    with tf.device('/gpu:0'):
        df = DeepNoisyBayesianFilter(sh = img_shape[0], H = 0.5, loss_type = "l1")
    
    df_samples.append([])
    kf_samples.append([])
    real_samples.append([])
    obs_samples.append([])
    idxs_samples.append([])
    err_df.append([])
    
    err_kf.append([])
    MSE_df.append([])
    MSE_kf.append([])   
    # ---- Sample simulation ---- #             
    x, z, idxs = random_walk(steps = 1200000,random_measurments = False, measure_time = measure_gap)
    for t in range(min_img*img_shape[0]*img_shape[1], len(x), img_shape[0]*img_shape[1]//2):
        # ----- Train on the current history ---- #
        for itr in range(t-min_img*img_shape[0]*img_shape[1], t-2*img_shape[0]*img_shape[1], skips):
            ii = itr + np.random.randint(-skips//2, skips//2)
            if ii < 0: ii = 0
            z_old = z[ii:(ii+img_shape[0]*img_shape[1])]
            x_old = x[ii:(ii+img_shape[0]*img_shape[1])]
            z_new = z[(ii+img_shape[0]*img_shape[1]):(ii+2*img_shape[0]*img_shape[1])]
            x_new = x[(ii+img_shape[0]*img_shape[1]):(ii+2*img_shape[0]*img_shape[1])]
            if training_ep < min_training:
                with tf.device('/gpu:0'):
                    df.train_predictor(x_old, x_new)
                    df.train_noise_predictor(z_new)
            if  training_ep > min_training:
                with tf.device('/gpu:0'):
                    df.train_updator(x_old,x_new,z_new)    
            training_ep += 1
            
        # ----- Use model to smooth ---- #
        if training_ep > min_training:
            z_new = z[(itr+img_shape[0]*img_shape[1]):(itr+2*img_shape[0]*img_shape[1])]  
            x_new = x[(itr+img_shape[0]*img_shape[1]):(itr+2*img_shape[0]*img_shape[1])] 
            idxs_new = idxs[(itr+img_shape[0]*img_shape[1]):(itr+2*img_shape[0]*img_shape[1])]
            if u is not None:
                u_new = u[(itr+img_shape[0]*img_shape[1]):(itr+2*img_shape[0]*img_shape[1])]
            else:
                u_new = None
            x_old = x[(itr):(itr+img_shape[0]*img_shape[1])]             
            with tf.device('/gpu:0'):
                x_hat_df = df.predict_mean(x_old, z_new)
           
            
         
            x_hat_kf, p_hat_kf = Kalman_smoother(z_new, idxs_new, u = u_new ,x0 = x_new[0], A = 1.0, R = 0.01, Q = 0.0001, hist = 200)
           
            
            df_samples[mc_run].append(x_hat_df[n_crop:-n_crop])
            idxs_samples[mc_run].append(idxs_new[n_crop:-n_crop])

            kf_samples[mc_run].append(x_hat_kf[n_crop:-n_crop])
            real_samples[mc_run].append(x_new[n_crop:-n_crop])
            obs_samples[mc_run].append(z_new[n_crop:-n_crop])            
           
            err_kf[mc_run].append(np.mean((x_new - x_hat_kf)[n_crop:-n_crop]))
            err_df[mc_run].append(np.mean((x_new - x_hat_df)[n_crop:-n_crop]))
            MSE_df[mc_run].append(np.mean(np.power(np.abs(x_new - x_hat_df),2)[n_crop:-n_crop]))
           
            MSE_kf[mc_run].append(np.mean(np.power(np.abs(x_new - x_hat_kf),2)[n_crop:-n_crop]))
            print("MC run:" + str(mc_run)+"Epoch: "+str(training_ep)+", time step: "+str(t))
            print("____ err ____")
            print("KF: "+'{:.3f}'.format(err_kf[mc_run][-1]))
            print("DF: "+'{:.3f}'.format(err_df[mc_run][-1]))
          
            print("____ MSE ____")
            print("KF: "+'{:.3f}'.format(MSE_kf[mc_run][-1]))
            print("DF: "+'{:.3f}'.format(MSE_df[mc_run][-1]))
           
            
            if len(err_df[mc_run]) % 100 == 0 and len(err_df[mc_run]) > 10 and plot_flag == True:
                plt.subplot(2,1,1)
                plt.plot(smooth(np.array(err_df[mc_run])),'blue')
               
                plt.plot(smooth(np.array(err_kf[mc_run])),'black')

                plt.xlabel("epoch")
                plt.ylabel("bias")
                plt.xscale("log")
                plt.subplot(2,1,2)
                plt.plot(smooth(np.array(MSE_df[mc_run])),'blue')
             
                plt.plot(smooth(np.array(MSE_kf[mc_run])),'black')
                plt.yscale("log")
                plt.xscale("log")
                plt.xlabel("epoch")
                plt.ylabel("MSE")
                plt.show()
        
       
     
 

