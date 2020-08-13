#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from tensorflow.keras import regularizers
from deepbayesianfilter.components.deepfilter import  DeepFilter, smooth, normalize, unnormalize, preprocess_data, preprocess_Bayesian_data, smooth_var, gen_sample
from deepbayesianfilter.components.variationaldeepfilter import VariationalDeepFilter

def sample(z, n):
    ns = len(z)
    step_sz = ns//n
    z_sample = []
    for ii in range(step_sz,ns+1,step_sz):
        mu = np.mean(z[ii-step_sz:ii])
        sigma = np.std(z[ii-step_sz:ii])
        z_sample.append(np.random.normal(mu,6*sigma,step_sz))
    return np.array(z_sample).reshape((-1))

# a=np.random.normal(0,1,4080)
# a = np.cumsum(a)
# b = sample(a, 50)

# plt.plot(b)
# plt.plot(a)
# plt.show()


class DeepVariationalBayesianFilter():
    def __init__(self, H = None, loss_type = "l1", sh = 128):
        self.Predictor = DeepFilter(input_shape = (sh,sh,1), output_shape = (sh,sh,1), n0_filter_zise = 4 , lr = 1e-4,  max_filters = 256, n0_filters = 32)
        self.NoisePredictor = DeepFilter(input_shape = (sh,sh,1), output_shape = (sh,sh,1), lr = 1e-4, max_filters = 64, n0_filter_zise = 2, n0_filters = 32)
        self.Updator = DeepFilter(input_shape = (sh,sh,2), output_shape = (sh,sh,1), n0_filter_zise = 4, lr = 2e-5, max_filters = 512, n0_filters = 64)
        self.H = H
        self.loss_type = loss_type
        self.p_len = 32
    def train_predictor(self, x_old, x_new):
        cmin_x = np.min(x_old) - 10*np.std(x_old)
        cmax_x = np.max(x_old) + 10*np.std(x_old)
        x_old = normalize(x_old, cmin_x, cmax_x)
        x_new = normalize(x_new, cmin_x, cmax_x)
        x_new = preprocess_data(x_new)
        x_old = preprocess_data(x_old)
        self.Predictor.train_step(np.expand_dims(x_old,3), np.expand_dims(x_new,3), L = 50)
    def train_noise_predictor(self, z_new):
        cmin_z = np.min(z_new) - 3*np.std(z_new)
        cmax_z = np.max(z_new) + 3*np.std(z_new)
        z_sample = sample(z_new,self.p_len)
        z_sample_normalized = normalize(z_sample, cmin_z, cmax_z)
        z_new_normalized = normalize(z_new, cmin_z, cmax_z)
        z_new_normalized = preprocess_data(z_new_normalized)
        z_sample_normalized = preprocess_data(z_sample_normalized)
        self.NoisePredictor.train_step(np.expand_dims(z_sample_normalized,3), np.expand_dims(z_new_normalized,3), L = 50, loss_type = "l2")
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
            z_sample = sample(z_new,self.p_len)
            z_sample_normalized = normalize(z_sample, cmin_z, cmax_z)
            z_sample_normalized = preprocess_data(z_sample_normalized)

            noise_hat = self.NoisePredictor.generator(np.expand_dims(z_sample_normalized,3), training=False)[0].numpy()
            noise_hat = np.reshape(noise_hat, (-1))
            
            
            
            input_data_update = preprocess_Bayesian_data(x_new_hat, z_sample_normalized)
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
        cmin_z = np.min(z_new) - 3*np.std(z_new)
        cmax_z = np.max(z_new) + 3*np.std(z_new)
        z_new_noisy_stack = []
        for ii in range(Ns):
            z_sample = sample(z_new,self.p_len)
            # z_sample_normalized = normalize(z_sample, cmin_z, cmax_z)
            # z_sample_normalized = preprocess_data(z_sample_normalized)
            # noise_hat = self.NoisePredictor.generator(np.expand_dims(z_sample_normalized,3), training=False)[0].numpy()
            # noise_hat = np.reshape(noise_hat, (-1))
            # noise_hat = unnormalize(noise_hat, cmin_z, cmax_z)
            z_new_noisy_stack.append(z_sample)
        return z_new_noisy_stack
    
    
    