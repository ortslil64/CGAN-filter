#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from cganfilter.components.deepfilter import DeepFilter, normalize, smooth_var, gen_sample, preprocess_data

def normalize_v(x):
    cmax = np.max(x)
    cmin = np.min(x)
    cscale = cmax - cmin
    x_out = (x * 1.0 - cmin) / cscale 
    x_out = np.cast[np.float32](x_out)
    return x_out

class DeepNoisyBayesianFilter():
    def __init__(self, hist = 4, image_shape = (128,128)):
        self.Predictor = DeepFilter(input_shape = (image_shape[0],image_shape[1],hist),
                                    output_shape = (image_shape[0],image_shape[1],1),
                                    lr = 1e-5,
                                    max_filters = 256)
        
        
        
        self.NoisePredictor = DeepFilter(input_shape = (image_shape[0],image_shape[1],4),
                                         output_shape = (image_shape[0],image_shape[1],1),
                                         lr = 1e-5,
                                         max_filters = 128)
        
        self.Updator = DeepFilter(input_shape = (image_shape[0],image_shape[1],2),
                                  output_shape = (image_shape[0],image_shape[1],1),
                                  lr = 1e-5,
                                  max_filters = 1024)
        
        self.Likelihood = DeepFilter(input_shape = (image_shape[0],image_shape[1],1),
                                  output_shape = (image_shape[0],image_shape[1],1),
                                  lr = 2e-6,
                                  max_filters = 1024)
        
        self.hist = hist
    def train_predictor(self, x_old, x_new):
        x_new = np.expand_dims(x_new,0)
        x_new = np.expand_dims(x_new,3)
        x_old = np.expand_dims(x_old,3)
        x_old = np.transpose(x_old,axes=(3,1,2,0))
        self.Predictor.train_step(x_old, x_new, L = 30)    
        
        
        
        
    def train_likelihood(self, z_new, x_new):
        x_new = np.expand_dims(x_new,0)
        x_new = np.expand_dims(x_new,3)
        z_new = np.expand_dims(z_new,0)
        z_new = np.expand_dims(z_new,3)
        self.Likelihood.train_step(z_new, x_new, L = 20)
        
    def train_noise_predictor(self, x_old, x_new):
        x_new = np.expand_dims(x_new,0)
        x_new = np.expand_dims(x_new,3)
        
        x_old = np.expand_dims(x_old,3)
        x_old = np.transpose(x_old,axes=(3,1,2,0))
        
        x_new_pred = self.Predictor.generator(x_old, training=False).numpy()
        x_new_pred = normalize_v(x_new_pred)
        err = x_new - x_new_pred
        err = normalize_v(err)
        var = np.var(x_old, axis = 3)
        var = np.expand_dims(var,3)
        var = normalize_v(var)
        input_data_update = np.concatenate((x_old, var), axis = 3)
        self.NoisePredictor.train_step(input_data_update, err, L = 30)
        
    def train_updator(self,x_old, x_new, z_new):
        x_old = np.expand_dims(x_old,3)
        x_old = np.transpose(x_old,axes=(3,1,2,0))
        x_new_pred = self.Predictor.generator(x_old, training=False).numpy()
        
        z_new = np.expand_dims(z_new,0)
        z_new = np.expand_dims(z_new,3)
        x_new_hat = self.Likelihood.generator(z_new, training=False).numpy()
        x_new_hat = normalize_v(x_new_hat)
        input_data_update = np.concatenate((x_new_hat, x_new_pred), axis = 3)
        x_new = np.expand_dims(x_new,0)
        x_new = np.expand_dims(x_new,3)
        self.Updator.train_step(input_data_update, x_new, L = 20)
        
    def predict_var(self, x_old,z_new):
        x_new_hat = self.predict_mean(x_old,z_new)
        x_stack = []
        for ii in range(50):
            z_sigma = gen_sample()
            z_sigma = preprocess_data(z_sigma)
            z_sigma = normalize_v(z_sigma)
            z_sigma = np.expand_dims(z_sigma,3)
            input_data_update = np.concatenate((x_old, z_sigma), axis = 3)
            noise_hat = self.NoisePredictor.generator(input_data_update, training=False)[0].numpy()

            x_new_hat_noisy = x_new_hat + noise_hat
            x_stack.append(x_new_hat_noisy)
        x_stack = np.array(x_stack)
        var = np.var(x_stack, axis = 0)
        return var
    
    def propogate(self, x_old):
        x_old = np.expand_dims(x_old,3)
        x_old = np.transpose(x_old,axes=(3,1,2,0))
        x_new_hat = self.Predictor.generator(x_old, training=False).numpy()
        return x_new_hat
    
    def estimate(self, z_new):
        z_new = np.expand_dims(z_new,0)
        z_new = np.expand_dims(z_new,3)
        x_new_hat = self.Likelihood.generator(z_new, training=False).numpy()
        x_new_hat = normalize_v(x_new_hat)
        return x_new_hat
        
    def predict_mean(self, x_old,z_new):
        x_new_pred = self.propogate(x_old)
        x_new_hat = self.estimate(z_new)
        z_new = np.expand_dims(z_new,0)
        z_new = np.expand_dims(z_new,3)
        x_new_mult = np.multiply(x_new_pred,z_new)
        #input_data_update = np.concatenate((z_new, x_new_pred), axis = 3)
        input_data_update = np.concatenate((x_new_hat, x_new_pred), axis = 3)
        #input_data_update = x_new_mult
        x_new_update = self.Updator.generator(input_data_update, training=False)[0].numpy()
        return x_new_update
    
    def save_weights(self, path):
        Predictor_path = path + '/predictor'
        # InvPredictor_path = path + '/invpredictor'
        Updator_path = path + '/updator'
        Likelihood_path = path + '/likelihood'
        self.Predictor.generator.save_weights(Predictor_path)
        # self.InvPredictor.generator.save_weights(InvPredictor_path)
        self.Updator.generator.save_weights(Updator_path)
        self.Likelihood.generator.save_weights(Likelihood_path)
    
    def load_weights(self, path):
        Predictor_path = path + '/predictor'
        # InvPredictor_path = path + '/invpredictor'
        Updator_path = path + '/updator'
        Likelihood_path = path + '/likelihood'
        self.Predictor.generator.load_weights(Predictor_path)
        # self.InvPredictor.generator.save_weights(InvPredictor_path)
        self.Updator.generator.load_weights(Updator_path)
        self.Likelihood.generator.load_weights(Likelihood_path)
        

    
