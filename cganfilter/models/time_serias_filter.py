#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from tensorflow.keras import regularizers
from cganfilter.components.deepfilter import DeepFilter, smooth, normalize, unnormalize, preprocess_data, preprocess_Bayesian_data, smooth_var, gen_sample
from cganfilter.components.tcn_predictor import TCNPredictor
from cganfilter.components.deepfilter import DeepFilter1D
    
    
class DeepNoisyBayesianFilter():
    def __init__(self, H = None, loss_type = "l1", sh = 128):
        self.Predictor = DeepFilter(input_shape = (sh,sh,1), output_shape = (sh,sh,1), n0_filter_zise = 4 , lr = 1e-4,  max_filters = 256, n0_filters = 32)
        self.NoisePredictor = DeepFilter(input_shape = (sh,sh,1), output_shape = (sh,sh,1), lr = 1e-4, max_filters = 64, n0_filter_zise = 2, n0_filters = 32)
        self.Updator = DeepFilter(input_shape = (sh,sh,2), output_shape = (sh,sh,1), n0_filter_zise = 4, lr = 2e-5, max_filters = 512, n0_filters = 64)
        self.H = H
        self.loss_type = loss_type
        self.p_len = 64
    def train_predictor(self, x_old, x_new):
        cmin_x = np.min(x_old) - 10*np.std(x_old)
        cmax_x = np.max(x_old) + 10*np.std(x_old)
        x_old = normalize(x_old, cmin_x, cmax_x)
        x_new = normalize(x_new, cmin_x, cmax_x)
        x_new = preprocess_data(x_new)
        x_old = preprocess_data(x_old)
        self.Predictor.train_step(np.expand_dims(x_old,3), np.expand_dims(x_new,3), L = 50)
    def train_noise_predictor(self, z_new):
        #z_sigma = smooth_var(z_new)
        z_sigma = gen_sample(p_len = self.p_len, v_size = len(z_new))
        cmin_z_sigma = np.min(z_sigma) - 3*np.std(z_sigma)
        cmax_z_sigma = np.max(z_sigma) + 3*np.std(z_sigma)
        z_sigma_normalized = normalize(z_sigma, cmin_z_sigma, cmax_z_sigma)
        z_output = z_new - np.mean(z_new)
        cmin_z = np.min(z_output) - 3*np.std(z_output)
        cmax_z = np.max(z_output) + 3*np.std(z_output)
        z_output_normalized = normalize(z_output, cmin_z, cmax_z)
        z_sigma_normalized = preprocess_data(z_sigma_normalized)
        z_output_normalized = preprocess_data(z_output_normalized)
        self.NoisePredictor.train_step(np.expand_dims(z_sigma_normalized,3), np.expand_dims(z_output_normalized,3), L = 50, loss_type = "l2")
    def train_updator(self,x_old, x_new, z_new):
        cmin_z = np.min(z_new) - 3*np.std(z_new)
        cmax_z = np.max(z_new) + 3*np.std(z_new)
        if self.H is None:
            cmin_x = np.min(x_old) - 10*np.std(x_old)
            cmax_x = np.max(x_old) + 10*np.std(x_old)
        else:
            cmin_x = cmin_z/self.H
            cmax_x = cmax_z/self.H
        z_new = normalize(z_new, cmin_z, cmax_z)
        x_old = normalize(x_old, cmin_x, cmax_x)
        x_new = normalize(x_new, cmin_x, cmax_x)
        x_new = preprocess_data(x_new)
        x_old = preprocess_data(x_old)
        x_new_hat = self.Predictor.generator(np.expand_dims(x_old,3), training=False)[0].numpy()
        x_new_hat = np.reshape(x_new_hat, (-1))
        input_data_update = preprocess_Bayesian_data(x_new_hat, z_new)
        self.Updator.train_step(np.expand_dims(input_data_update,0), np.expand_dims(x_new,3), L = 50, loss_type = self.loss_type)
                
    def predict_var(self, x_old,z_new, Ns = 30):
        cmin_z = np.min(z_new) - 3*np.std(z_new)
        cmax_z = np.max(z_new) + 3*np.std(z_new)
        cmin_z_unb = np.min(z_new-np.mean(z_new)) - 3*np.std(z_new)
        cmax_z_unb = np.max(z_new-np.mean(z_new)) + 3*np.std(z_new)
        if self.H is None:
            cmin_x = np.min(x_old) - 10*np.std(x_old)
            cmax_x = np.max(x_old) + 10*np.std(x_old)
        else:
            cmin_x = cmin_z/self.H
            cmax_x = cmax_z/self.H
        x_old_normalized = normalize(x_old, cmin_x, cmax_x)
        x_old_normalized = preprocess_data(x_old_normalized)
        x_new_hat = self.Predictor.generator(np.expand_dims(x_old_normalized,3), training=False)[0].numpy()
        x_new_hat = np.reshape(x_new_hat, (-1))
        x_stack = []
        for ii in range(Ns):
            z_sigma = gen_sample(p_len = self.p_len, v_size = len(x_new_hat))
            #z_sigma = smooth_var(z_sigma)
            cmin_z_sigma = np.min(z_sigma) - 3*np.std(z_sigma)
            cmax_z_sigma = np.max(z_sigma) + 3*np.std(z_sigma)
            
            
            
            z_sigma_normalized = normalize(z_sigma, cmin_z_sigma, cmax_z_sigma)
            z_sigma_normalized = preprocess_data(z_sigma_normalized)
            noise_hat = self.NoisePredictor.generator(np.expand_dims(z_sigma_normalized,3), training=False)[0].numpy()
            noise_hat = np.reshape(noise_hat, (-1))
            noise_hat = unnormalize(noise_hat, cmin_z_unb, cmax_z_unb)
            noise_hat = smooth(noise_hat)
            z_new_noisy = z_new + noise_hat
            cmin_z_n = np.min(z_new_noisy) - 3*np.std(z_new_noisy)
            cmax_z_n = np.max(z_new_noisy) + 3*np.std(z_new_noisy)
            z_new_noisy_normalized = normalize(z_new_noisy, cmin_z_n, cmax_z_n)
            
            
            input_data_update = preprocess_Bayesian_data(x_new_hat, z_new_noisy_normalized)
            x_new_update = self.Updator.generator(np.expand_dims(input_data_update,0), training=False)[0].numpy()
            x_new_update = np.reshape(x_new_update, (-1))
            x_new_update = unnormalize(x_new_update, cmin_x, cmax_x)
            x_stack.append(x_new_update)
        x_stack = np.array(x_stack)
        var = np.var(x_stack, axis = 0)
        return var, x_stack
        
    def predict_mean(self, x_old,z_new):
        cmin_z = np.min(z_new) - 3*np.std(z_new)
        cmax_z = np.max(z_new) + 3*np.std(z_new)
        if self.H is None:
            cmin_x = np.min(x_old) - 10*np.std(x_old)
            cmax_x = np.max(x_old) + 10*np.std(x_old)
        else:
            cmin_x = cmin_z/self.H
            cmax_x = cmax_z/self.H
        z_new = normalize(z_new, cmin_z, cmax_z)
        x_old = normalize(x_old, cmin_x, cmax_x)
        x_old = preprocess_data(x_old)
        x_new_hat = self.Predictor.generator(np.expand_dims(x_old,3), training=False)[0].numpy()
        x_new_hat = np.reshape(x_new_hat, (-1))
        input_data_update = preprocess_Bayesian_data(x_new_hat, z_new)
        x_new_update = self.Updator.generator(np.expand_dims(input_data_update,0), training=False)[0].numpy()
        x_new_update = np.reshape(x_new_update, (-1))
        x_new_update = unnormalize(x_new_update, cmin_x, cmax_x)
        return x_new_update
    
    def prop(self, x_old, z_new):
        cmin_z = np.min(z_new) - 3*np.std(z_new)
        cmax_z = np.max(z_new) + 3*np.std(z_new)
        if self.H is None:
            cmin_x = np.min(x_old) - 10*np.std(x_old)
            cmax_x = np.max(x_old) + 10*np.std(x_old)
        else:
            cmin_x = cmin_z/self.H
            cmax_x = cmax_z/self.H
        x_old = normalize(x_old, cmin_x, cmax_x)
        x_old = preprocess_data(x_old)
        x_new_hat = self.Predictor.generator(np.expand_dims(x_old,3), training=False)[0].numpy()
        x_new_hat = np.reshape(x_new_hat, (-1))
        x_new_hat = unnormalize(x_new_hat, cmin_x, cmax_x)
        return x_new_hat
    
    def gen_noise(self, z_new, Ns = 30):
        cmin_z = np.min(z_new-np.mean(z_new)) - 3*np.std(z_new)
        cmax_z = np.max(z_new-np.mean(z_new)) + 3*np.std(z_new)
        z_new_noisy_stack = []
        for ii in range(Ns):
            z_sigma = gen_sample(p_len = self.p_len, v_size = len(z_new))
            cmin_z_sigma = np.min(z_sigma) - 3*np.std(z_sigma)
            cmax_z_sigma = np.max(z_sigma) + 3*np.std(z_sigma)
            z_sigma_normalized = normalize(z_sigma, cmin_z_sigma, cmax_z_sigma)
            z_sigma_normalized = preprocess_data(z_sigma_normalized)
            noise_hat = self.NoisePredictor.generator(np.expand_dims(z_sigma_normalized,3), training=False)[0].numpy()
            noise_hat = np.reshape(noise_hat, (-1))
            noise_hat = unnormalize(noise_hat, cmin_z, cmax_z)
            noise_hat = smooth(noise_hat)
            z_new_noisy = z_new + noise_hat
            z_new_noisy_stack.append(z_new_noisy)
        return z_new_noisy_stack
    
    
    
    
class TCNBayesianFilter():
    def __init__(self, H = None, loss_type = "l1", sh = 128):
        self.timesteps = sh**2
        self.Predictor = TCNPredictor(input_shape = self.timesteps, timesteps = self.timesteps, lr = 1e-4)
        self.NoisePredictor = DeepFilter(input_shape = (sh,sh,1), output_shape = (sh,sh,1), lr = 1e-4, max_filters = 64, n0_filter_zise = 2, n0_filters = 32)
        self.Updator = DeepFilter(input_shape = (sh,sh,2), output_shape = (sh,sh,1), n0_filter_zise = 4, lr = 1e-4, max_filters = 256, n0_filters = 32)
        self.H = H
        self.loss_type = loss_type
        
    def train_predictor(self, x_old, x_new):
        cmin_x = np.min(x_old) - 10*np.std(x_old)
        cmax_x = np.max(x_old) + 10*np.std(x_old)
        x_old = normalize(x_old, cmin_x, cmax_x)
        x_new = normalize(x_new, cmin_x, cmax_x)
        x_new = np.reshape(x_new,(1,self.timesteps,1))
        x_old = np.reshape(x_old,(1,self.timesteps,1))
        self.Predictor.train_step(x_old, x_new, self.loss_type)
        
    def train_noise_predictor(self, z_new):
        z_sigma = smooth_var(z_new)
        cmin_z_sigma = np.min(z_sigma) - 3*np.std(z_sigma)
        cmax_z_sigma = np.max(z_sigma) + 3*np.std(z_sigma)
        z_sigma_normalized = normalize(z_sigma, cmin_z_sigma, cmax_z_sigma)
        z_output = z_new - np.mean(z_new)
        cmin_z = np.min(z_output) - 3*np.std(z_output)
        cmax_z = np.max(z_output) + 3*np.std(z_output)
        z_output_normalized = normalize(z_output, cmin_z, cmax_z)
        z_sigma_normalized = preprocess_data(z_sigma_normalized)
        z_output_normalized = preprocess_data(z_output_normalized)
        self.NoisePredictor.train_step(np.expand_dims(z_sigma_normalized,3), np.expand_dims(z_output_normalized,3), L = 30)
    def train_updator(self,x_old, x_new, z_new):
        cmin_z = np.min(z_new) - 3*np.std(z_new)
        cmax_z = np.max(z_new) + 3*np.std(z_new)
        if self.H is None:
            cmin_x = np.min(x_old) - 10*np.std(x_old)
            cmax_x = np.max(x_old) + 10*np.std(x_old)
        else:
            cmin_x = cmin_z/self.H
            cmax_x = cmax_z/self.H
        z_new = normalize(z_new, cmin_z, cmax_z)
        x_new = normalize(x_new, cmin_x, cmax_x)
        x_new = preprocess_data(x_new)
        x_old = normalize(x_old, cmin_x, cmax_x)
        x_new_hat = self.Predictor.predict(x_old)
        input_data_update = preprocess_Bayesian_data(x_new_hat, z_new)
        self.Updator.train_step(np.expand_dims(input_data_update,0), np.expand_dims(x_new,3), L = 100, loss_type = self.loss_type)
        
    def predict_var(self, x_old,z_new):
        cmin_z = np.min(z_new-np.mean(z_new)) - 3*np.std(z_new)
        cmax_z = np.max(z_new-np.mean(z_new)) + 3*np.std(z_new)
        if self.H is None:
            cmin_x = np.min(x_old) - 10*np.std(x_old)
            cmax_x = np.max(x_old) + 10*np.std(x_old)
        else:
            cmin_x = cmin_z/self.H
            cmax_x = cmax_z/self.H
        x_old = normalize(x_old, cmin_x, cmax_x)
        x_new_hat = self.Predictor.predict(x_old)
        x_stack = []
        for ii in range(30):
            z_sigma = gen_sample(v_size = len(x_new_hat))
            #z_sigma = smooth_var(z_sigma)
            cmin_z_sigma = np.min(z_sigma) - 3*np.std(z_sigma)
            cmax_z_sigma = np.max(z_sigma) + 3*np.std(z_sigma)
            
            
            
            z_sigma_normalized = normalize(z_sigma, cmin_z_sigma, cmax_z_sigma)
            z_sigma_normalized = preprocess_data(z_sigma_normalized)
            noise_hat = self.NoisePredictor.generator(np.expand_dims(z_sigma_normalized,3), training=False)[0].numpy()
            noise_hat = np.reshape(noise_hat, (-1))
            noise_hat = unnormalize(noise_hat, cmin_z, cmax_z)
            noise_hat = smooth(noise_hat)
            z_new_noisy = z_new + noise_hat
            cmin_z_n = np.min(z_new_noisy) - 3*np.std(z_new_noisy)
            cmax_z_n = np.max(z_new_noisy) + 3*np.std(z_new_noisy)
            z_new_noisy_normalized = normalize(z_new_noisy, cmin_z_n, cmax_z_n)
            
            
            input_data_update = preprocess_Bayesian_data(x_new_hat, z_new_noisy_normalized)
            x_new_update = self.Updator.generator(np.expand_dims(input_data_update,0), training=False)[0].numpy()
            x_new_update = np.reshape(x_new_update, (-1))
            x_new_update = unnormalize(x_new_update, cmin_x, cmax_x)
            x_stack.append(x_new_update)
        x_stack = np.array(x_stack)
        var = np.var(x_stack, axis = 0)
        return var
        
    def predict_mean(self, x_old,z_new):
        cmin_z = np.min(z_new) - 3*np.std(z_new)
        cmax_z = np.max(z_new) + 3*np.std(z_new)
        if self.H is None:
            cmin_x = np.min(x_old) - 10*np.std(x_old)
            cmax_x = np.max(x_old) + 10*np.std(x_old)
        else:
            cmin_x = cmin_z/self.H
            cmax_x = cmax_z/self.H
        z_new = normalize(z_new, cmin_z, cmax_z)
        x_old = normalize(x_old, cmin_x, cmax_x)
        x_new_hat = self.Predictor.predict(x_old)
        input_data_update = preprocess_Bayesian_data(x_new_hat, z_new)
        x_new_update = self.Updator.generator(np.expand_dims(input_data_update,0), training=False)[0].numpy()
        x_new_update = np.reshape(x_new_update, (-1))
        x_new_update = unnormalize(x_new_update, cmin_x, cmax_x)
        return x_new_update
    
    
    def prop(self, x_old, z_new):
        cmin_z = np.min(z_new) - 3*np.std(z_new)
        cmax_z = np.max(z_new) + 3*np.std(z_new)
        if self.H is None:
            cmin_x = np.min(x_old) - 10*np.std(x_old)
            cmax_x = np.max(x_old) + 10*np.std(x_old)
        else:
            cmin_x = cmin_z/self.H
            cmax_x = cmax_z/self.H
        x_old = normalize(x_old, cmin_x, cmax_x)
        x_new_hat = self.Predictor.predict(x_old)
        x_new_hat = np.reshape(x_new_hat, (-1))
        x_new_hat = unnormalize(x_new_hat, cmin_x, cmax_x)
        return x_new_hat
    
    def gen_noise(self, z_new):
        cmin_z = np.min(z_new) - 3*np.std(z_new)
        cmax_z = np.max(z_new) + 3*np.std(z_new)
        z_sigma = gen_sample()
        cmin_z_sigma = np.min(z_sigma) - 3*np.std(z_sigma)
        cmax_z_sigma = np.max(z_sigma) + 3*np.std(z_sigma)
        z_sigma_normalized = normalize(z_sigma, cmin_z_sigma, cmax_z_sigma)
        z_sigma_normalized = preprocess_data(z_sigma_normalized)
        noise_hat = self.NoisePredictor.generator(np.expand_dims(z_sigma_normalized,3), training=False)[0].numpy()
        noise_hat = np.reshape(noise_hat, (-1))
        noise_hat = unnormalize(noise_hat, cmin_z, cmax_z)
        noise_hat = smooth(noise_hat)
        return noise_hat    
   
    
class DeepNoisyBayesianFilterLinearPredictor():
    def __init__(self, H = None):
        self.NoisePredictor = DeepFilter(input_shape = (128,128,1), output_shape = (128,128,1), lr = 2e-5, max_filters = 64, n0_filter_zise = 2)
        self.Updator = DeepFilter(input_shape = (128,128,2), output_shape = (128,128,1), n0_filter_zise = 4, lr = 1e-5, max_filters = 256)
        self.H = H
    
    def predict_linear(self, x_old):
        x_new_hat = np.ones_like(x_old)
        x_new_hat = x_new_hat*x_old[-1]
        return x_new_hat
    
    def train_noise_predictor(self, z_new):
        z_sigma = smooth_var(z_new)
        cmin_z_sigma = np.min(z_sigma) - 3*np.std(z_sigma)
        cmax_z_sigma = np.max(z_sigma) + 3*np.std(z_sigma)
        z_sigma_normalized = normalize(z_sigma, cmin_z_sigma, cmax_z_sigma)
        z_output = z_new - np.mean(z_new)
        cmin_z = np.min(z_output) - 3*np.std(z_output)
        cmax_z = np.max(z_output) + 3*np.std(z_output)
        z_output_normalized = normalize(z_output, cmin_z, cmax_z)
        z_sigma_normalized = preprocess_data(z_sigma_normalized)
        z_output_normalized = preprocess_data(z_output_normalized)
        self.NoisePredictor.train_step(np.expand_dims(z_sigma_normalized,3), np.expand_dims(z_output_normalized,3), L = 30)
    def train_updator(self,x_old, x_new, z_new):
        cmin_z = np.min(z_new) - 3*np.std(z_new)
        cmax_z = np.max(z_new) + 3*np.std(z_new)
        if self.H is None:
            cmin_x = np.min(x_old) - 10*np.std(x_old)
            cmax_x = np.max(x_old) + 10*np.std(x_old)
        else:
            cmin_x = cmin_z/self.H
            cmax_x = cmax_z/self.H
        z_new = normalize(z_new, cmin_z, cmax_z)
        x_old = normalize(x_old, cmin_x, cmax_x)
        x_new = normalize(x_new, cmin_x, cmax_x)
        x_new = preprocess_data(x_new)
        x_old = preprocess_data(x_old)
        
        x_new_hat = self.predict_linear(np.reshape(x_old,(-1)))
        input_data_update = preprocess_Bayesian_data(x_new_hat, z_new)
        self.Updator.train_step(np.expand_dims(input_data_update,0), np.expand_dims(x_new,3), L = 100)
        
    def predict_var(self, x_old,z_new):
        cmin_z = np.min(z_new-np.mean(z_new)) - 3*np.std(z_new)
        cmax_z = np.max(z_new-np.mean(z_new)) + 3*np.std(z_new)
        if self.H is None:
            cmin_x = np.min(x_old) - 10*np.std(x_old)
            cmax_x = np.max(x_old) + 10*np.std(x_old)
        else:
            cmin_x = cmin_z/self.H
            cmax_x = cmax_z/self.H
        x_old_normalized = normalize(x_old, cmin_x, cmax_x)
        x_old_normalized = preprocess_data(x_old_normalized)
        x_new_hat  = self.predict_linear(np.reshape(x_old_normalized,(-1)))
        x_stack = []
        for ii in range(30):
            z_sigma = gen_sample()
            #z_sigma = smooth_var(z_sigma)
            cmin_z_sigma = np.min(z_sigma) - 3*np.std(z_sigma)
            cmax_z_sigma = np.max(z_sigma) + 3*np.std(z_sigma)
            
            
            
            z_sigma_normalized = normalize(z_sigma, cmin_z_sigma, cmax_z_sigma)
            z_sigma_normalized = preprocess_data(z_sigma_normalized)
            noise_hat = self.NoisePredictor.generator(np.expand_dims(z_sigma_normalized,3), training=False)[0].numpy()
            noise_hat = np.reshape(noise_hat, (-1))
            noise_hat = unnormalize(noise_hat, cmin_z, cmax_z)
            noise_hat = smooth(noise_hat)
            z_new_noisy = z_new + noise_hat
            cmin_z_n = np.min(z_new_noisy) - 3*np.std(z_new_noisy)
            cmax_z_n = np.max(z_new_noisy) + 3*np.std(z_new_noisy)
            z_new_noisy_normalized = normalize(z_new_noisy, cmin_z_n, cmax_z_n)
            
            
            input_data_update = preprocess_Bayesian_data(x_new_hat, z_new_noisy_normalized)
            x_new_update = self.Updator.generator(np.expand_dims(input_data_update,0), training=False)[0].numpy()
            x_new_update = np.reshape(x_new_update, (-1))
            x_new_update = unnormalize(x_new_update, cmin_x, cmax_x)
            x_stack.append(x_new_update)
        x_stack = np.array(x_stack)
        var = np.var(x_stack, axis = 0)
        return var
        
    def predict_mean(self, x_old,z_new):
        cmin_z = np.min(z_new) - 3*np.std(z_new)
        cmax_z = np.max(z_new) + 3*np.std(z_new)
        if self.H is None:
            cmin_x = np.min(x_old) - 10*np.std(x_old)
            cmax_x = np.max(x_old) + 10*np.std(x_old)
        else:
            cmin_x = cmin_z/self.H
            cmax_x = cmax_z/self.H
        z_new = normalize(z_new, cmin_z, cmax_z)
        x_old = normalize(x_old, cmin_x, cmax_x)
        x_old = preprocess_data(x_old)
        x_new_hat = self.predict_linear(np.reshape(x_old,(-1)))
        input_data_update = preprocess_Bayesian_data(x_new_hat, z_new)
        x_new_update = self.Updator.generator(np.expand_dims(input_data_update,0), training=False)[0].numpy()
        x_new_update = np.reshape(x_new_update, (-1))
        x_new_update = unnormalize(x_new_update, cmin_x, cmax_x)
        return x_new_update
    

    def gen_noise(self, z_new):
        cmin_z = np.min(z_new) - 3*np.std(z_new)
        cmax_z = np.max(z_new) + 3*np.std(z_new)
        z_sigma = gen_sample()
        cmin_z_sigma = np.min(z_sigma) - 3*np.std(z_sigma)
        cmax_z_sigma = np.max(z_sigma) + 3*np.std(z_sigma)
        z_sigma_normalized = normalize(z_sigma, cmin_z_sigma, cmax_z_sigma)
        z_sigma_normalized = preprocess_data(z_sigma_normalized)
        noise_hat = self.NoisePredictor.generator(np.expand_dims(z_sigma_normalized,3), training=False)[0].numpy()
        noise_hat = np.reshape(noise_hat, (-1))
        noise_hat = unnormalize(noise_hat, cmin_z, cmax_z)
        noise_hat = smooth(noise_hat)
        return noise_hat
   
    


    
class DeepNoisyBayesianFilter1D():
    def __init__(self, H = None, loss_type = "l1", input_shape = (128,1), output_shape = (128,1)):
        self.Predictor = DeepFilter1D(input_shape = input_shape, output_shape = output_shape, n0_filter_zise = 4 , lr = 5e-5,  max_filters = 256)
        self.NoisePredictor = DeepFilter1D(input_shape = input_shape, output_shape = output_shape, lr = 2e-5, max_filters = 64)
        self.Updator = DeepFilter1D(input_shape = (output_shape[0],2*output_shape[1]), output_shape = output_shape, lr = 1e-5, max_filters = 256)
        self.H = H
        self.loss_type = loss_type
        self.input_shape = input_shape
        self.output_shape = output_shape
        
    def train_predictor(self, x_old, x_new):
        cmin_x = np.min(x_old) - 3*np.std(x_old)
        cmax_x = np.max(x_old) + 3*np.std(x_old)
        x_old = normalize(x_old, cmin_x, cmax_x)
        x_new = normalize(x_new, cmin_x, cmax_x)
        x_new = np.reshape(x_new, (1,self.output_shape[0], self.output_shape[1]))
        x_old = np.reshape(x_old, (1,self.input_shape[0], self.input_shape[1]))
        self.Predictor.train_step(x_old, x_new, L = 50)
        
    def train_noise_predictor(self, z_new):
        z_sigma = smooth_var(z_new)
        cmin_z_sigma = np.min(z_sigma) - 3*np.std(z_sigma)
        cmax_z_sigma = np.max(z_sigma) + 3*np.std(z_sigma)
        z_sigma_normalized = normalize(z_sigma, cmin_z_sigma, cmax_z_sigma)
        z_output = z_new - np.mean(z_new)
        cmin_z = np.min(z_output) - 3*np.std(z_output)
        cmax_z = np.max(z_output) + 3*np.std(z_output)
        z_output_normalized = normalize(z_output, cmin_z, cmax_z)
        z_sigma_normalized = np.reshape(z_sigma_normalized, (1,self.input_shape[0], self.input_shape[1]))
        z_output_normalized = np.reshape(z_output_normalized, (1,self.output_shape[0], self.output_shape[1]))
        self.NoisePredictor.train_step(z_sigma_normalized, z_output_normalized, L = 30)
        
    def train_updator(self,x_old, x_new, z_new):
        cmin_z = np.min(z_new) - 3*np.std(z_new)
        cmax_z = np.max(z_new) + 3*np.std(z_new)
        if self.H is None:
            cmin_x = np.min(x_old) - 10*np.std(x_old)
            cmax_x = np.max(x_old) + 10*np.std(x_old)
        else:
            cmin_x = cmin_z/self.H
            cmax_x = cmax_z/self.H
        z_new = normalize(z_new, cmin_z, cmax_z)
        x_old = normalize(x_old, cmin_x, cmax_x)
        x_new = normalize(x_new, cmin_x, cmax_x)
        x_new = np.reshape(x_new, (1,self.output_shape[0], self.output_shape[1]))
        x_old = np.reshape(x_old, (1,self.input_shape[0], self.input_shape[1]))
        x_new_hat = self.Predictor.generator.predict(x_old)
        x_new_hat = np.reshape(x_new_hat, (1,self.output_shape[0], self.output_shape[1]))
        z_new = np.reshape(z_new, (1,self.output_shape[0], self.output_shape[1])) 
        input_data_update = np.concatenate((x_new_hat, z_new), axis = 2)
        
        self.Updator.train_step(input_data_update, x_new, L = 100, loss_type = self.loss_type)
        
    def predict_var(self, x_old,z_new):
        cmin_z = np.min(z_new-np.mean(z_new)) - 3*np.std(z_new)
        cmax_z = np.max(z_new-np.mean(z_new)) + 3*np.std(z_new)
        if self.H is None:
            cmin_x = np.min(x_old) - 10*np.std(x_old)
            cmax_x = np.max(x_old) + 10*np.std(x_old)
        else:
            cmin_x = cmin_z/self.H
            cmax_x = cmax_z/self.H
        x_old_normalized = normalize(x_old, cmin_x, cmax_x)
        x_old_normalized = np.reshape(x_old_normalized, (1,self.input_shape[0], self.input_shape[1]))
        x_new_hat = self.Predictor.generator.predict(x_old_normalized)[0]
        x_new_hat = np.reshape(x_new_hat, (1,self.output_shape[0], self.output_shape[1]))
        x_stack = []
        for ii in range(30):
            z_sigma = gen_sample(self.output_shape[0])
            #z_sigma = smooth_var(z_sigma)
            cmin_z_sigma = np.min(z_sigma) - 3*np.std(z_sigma)
            cmax_z_sigma = np.max(z_sigma) + 3*np.std(z_sigma)
            
            
            
            z_sigma_normalized = normalize(z_sigma, cmin_z_sigma, cmax_z_sigma)
            z_sigma_normalized = np.reshape(z_sigma_normalized, (1,self.output_shape[0], self.output_shape[1]))
            noise_hat = self.NoisePredictor.generator.predict(z_sigma_normalized)[0]
            noise_hat = np.reshape(noise_hat, (-1))
            noise_hat = unnormalize(noise_hat, cmin_z, cmax_z)
            noise_hat = smooth(noise_hat)
            z_new_noisy = z_new + noise_hat
            cmin_z_n = np.min(z_new_noisy) - 3*np.std(z_new_noisy)
            cmax_z_n = np.max(z_new_noisy) + 3*np.std(z_new_noisy)
            z_new_noisy_normalized = normalize(z_new_noisy, cmin_z_n, cmax_z_n)
            z_new_noisy_normalized = np.reshape(z_new_noisy_normalized, (1,self.output_shape[0], self.output_shape[1]))
            input_data_update = np.concatenate((x_new_hat, z_new_noisy_normalized), axis = 2)
            x_new_update = self.Updator.generator.predict(input_data_update)[0]
            x_new_update = np.reshape(x_new_update, (-1))
            x_new_update = unnormalize(x_new_update, cmin_x, cmax_x)
            x_stack.append(x_new_update)
        x_stack = np.array(x_stack)
        var = np.var(x_stack, axis = 0)
        return var
        
    def predict_mean(self, x_old,z_new):
        cmin_z = np.min(z_new) - 3*np.std(z_new)
        cmax_z = np.max(z_new) + 3*np.std(z_new)
        if self.H is None:
            cmin_x = np.min(x_old) - 10*np.std(x_old)
            cmax_x = np.max(x_old) + 10*np.std(x_old)
        else:
            cmin_x = cmin_z/self.H
            cmax_x = cmax_z/self.H
        z_new = normalize(z_new, cmin_z, cmax_z)
        x_old = normalize(x_old, cmin_x, cmax_x)
        x_old = np.reshape(x_old, (1,self.input_shape[0], self.input_shape[1]))
        x_new_hat = self.Predictor.generator.predict(x_old)[0]
        x_new_hat = np.reshape(x_new_hat, (1,self.output_shape[0], self.output_shape[1]))
        z_new = np.reshape(z_new, (1,self.output_shape[0], self.output_shape[1]))
        input_data_update = np.concatenate((x_new_hat, z_new), axis = 2)
        x_new_update = self.Updator.generator.predict(input_data_update)
        x_new_update = np.reshape(x_new_update, (-1))
        x_new_update = unnormalize(x_new_update, cmin_x, cmax_x)
        return x_new_update
    
    def prop(self, x_old, z_new):
        cmin_z = np.min(z_new) - 3*np.std(z_new)
        cmax_z = np.max(z_new) + 3*np.std(z_new)
        if self.H is None:
            cmin_x = np.min(x_old) - 10*np.std(x_old)
            cmax_x = np.max(x_old) + 10*np.std(x_old)
        else:
            cmin_x = cmin_z/self.H
            cmax_x = cmax_z/self.H
        x_old = normalize(x_old, cmin_x, cmax_x)
        x_old = np.reshape(x_old, (1,self.input_shape[0], self.input_shape[1]))
        x_new_hat = self.Predictor.generator.predict(x_old)[0]
        x_new_hat = np.reshape(x_new_hat, (-1))
        x_new_hat = unnormalize(x_new_hat, cmin_x, cmax_x)
        return x_new_hat
    

        
       





  



