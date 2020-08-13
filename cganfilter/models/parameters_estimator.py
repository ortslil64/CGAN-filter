#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from tensorflow.keras import regularizers
from deepbayesianfilter.components.deepfilter  import DeepFilter, smooth, normalize, unnormalize, preprocess_data, preprocess_Bayesian_data, smooth_var, gen_sample
from deepbayesianfilter.components.tcn_predictor import TCNPredictor

    
    
class DeepParametersEstimator():
    def __init__(self, H = None, loss_type = "l1", input_shape = (128,128,1), output_shape = (128,128,1)):
        self.estimator = DeepFilter(input_shape = input_shape, output_shape = output_shape, n0_filter_zise = 4 , lr = 1e-5,  max_filters = 512)
        self.loss_type = loss_type
        self.input_shape = input_shape
        self.output_shape = output_shape
    def train(self, x_old, x_new):
        cmin_x = np.min(x_old) - 3*np.std(x_old)
        cmax_x = np.max(x_old) + 3*np.std(x_old)
        x_old = normalize(x_old, cmin_x, cmax_x)
        x_new = normalize(x_new, cmin_x, cmax_x)
        x_new = np.reshape(x_new, self.output_shape)
        x_old = np.reshape(x_old, self.input_shape)
        self.estimator.train_step(np.expand_dims(x_old,0), np.expand_dims(x_new,0), L = 100, loss_type = self.loss_type)
    
    
    def estimate(self, x_old):
        cmin_x = np.min(x_old) - 3*np.std(x_old)
        cmax_x = np.max(x_old) + 3*np.std(x_old)
        
        x_old = normalize(x_old, cmin_x, cmax_x)
        x_old = np.reshape(x_old, self.input_shape)
        x_new_hat = self.estimator.generator(np.expand_dims(x_old,0), training=False)[0].numpy()
        x_new_hat = np.reshape(x_new_hat, (-1))
        x_new_hat = unnormalize(x_new_hat, cmin_x, cmax_x)
        return x_new_hat
    
    

    
    
    
    
    