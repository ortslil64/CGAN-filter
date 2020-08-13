from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from tensorflow.keras.layers import Dense
from tensorflow.keras import Input, Model
import tensorflow as tf
from tcn import TCN



def smooth(x):
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i-99)
        y[i] = float(x[start:(i+1)].sum())/(i-start+1)
    return y
    

def normalize(x,cmin, cmax):
    cscale = cmax - cmin
    x_out = (x * 1.0 - cmin) / cscale 
    x_out = np.cast[np.float32](x_out)
    return x_out

def unnormalize(x,cmin, cmax):
    cscale = cmax - cmin
    x_out = x * cscale + cmin
    x_out = np.cast[np.float32](x_out)
    return x_out


def preprocess_data(x):
    d = int(np.sqrt(len(x)))
    img = np.reshape(x, (d,d))
    img = np.expand_dims(img,0)
    return img

def preprocess_Bayesian_data(x_old, z_new):
    d = int(np.sqrt(len(x_old)))
    x = np.reshape(x_old, (d,d))
    x = np.expand_dims(x,2)
    z = np.reshape(z_new, (d,d))
    z = np.expand_dims(z,2)
    return np.concatenate((z,x),axis = 2)

def smooth_var(z, var_len = 128*128//32):
    n = len(z)//var_len
    z_sigma = np.empty_like(z)
    for ii in range(n):
        sigma = np.var(z[(ii*var_len):((ii+1)*var_len)])
        z_sigma[(ii*var_len):((ii+1)*var_len)] = sigma
    z_sigma = z_sigma + np.random.normal(0, 0.1, size =  len(z))
    return z_sigma

def gen_sample(v_size = 128*128, p_len = 16):
    n = v_size//p_len
    z_sigma = np.empty(v_size)
    for ii in range(p_len):
        z_sigma[(ii*n):((ii+1)*n)] = np.random.rand()
    z_sigma = z_sigma + np.random.normal(0, 0.1, size = v_size)
    return z_sigma

def get_x_y(x, timesteps = 32):
    x_out = []
    x_pred_out = []
    n = len(x) - timesteps
    for t in range(0,n,timesteps):
        x_out.append(x[t:(t+timesteps)])
        x_pred_out.append(x[(t+timesteps):(t+2*timesteps)])
    return np.array(x_out, dtype = np.float32), np.array(x_pred_out, dtype = np.float32)

def arrange_data_to_predict(x, timesteps = 32):
    x_out = []
    n = (len(x)- timesteps)
    for t in range(0,n,timesteps):
        if len(x[t:(t+timesteps)]) != timesteps:
            break
        else:
            x_out.append(x[t:(t+timesteps)])
    return x_out



class TCNPredictor():
    def __init__(self, input_shape = 128**2, timesteps = 32, lr = 1e-4):
        self.input_shape = input_shape
        self.timesteps = timesteps
        batch_size = None
        input_dim = 1
        tcn_input = Input(batch_shape=(batch_size, timesteps, input_dim))
        tcn_output = TCN(return_sequences=True, use_batch_norm=False)(tcn_input)  # The TCN layers are here.
        tcn_output = Dense(1)(tcn_output)
        self.tcn_model = Model(inputs=[tcn_input], outputs=[tcn_output])
        self.optimizer = tf.keras.optimizers.Adam(lr)
        
        
    def train(self, x,y, epochs = 1):
        x_train = np.reshape(x,(1,self.timesteps,1))
        y_train = np.reshape(y,(1,self.timesteps,1))
        self.tcn_model.fit(x_train, y_train, epochs=epochs , verbose = 0)
        del x_train, y_train
    
    def train_step(self, x, y, loss_type = "l1"):
        with tf.GradientTape() as tape:
            y_hat = self.tcn_model([x], training=True)
            if loss_type == "l1":
                loss = tf.reduce_mean(tf.abs(y - y_hat))
            if loss_type == "l2":
                loss = tf.reduce_mean(tf.pow(y - y_hat,2))

        gradients = tape.gradient(loss, self.tcn_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.tcn_model.trainable_variables))
        
  
    def predict(self,x ):
        x_to_predict = np.reshape(x,(1,self.timesteps,1))
        x_predicted = self.tcn_model.predict(x_to_predict)
        return x_predicted.reshape((-1))
            
            
   

        
       





  



