from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
from deepbayesianfilter.components.components import downsample, upsample, downsample1d, upsample1d, discriminator_loss, generator_loss
from tensorflow.keras import regularizers
    

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
    z_sigma = z_sigma + np.random.normal(0, 0.05, size = v_size)
    return z_sigma

def preprocess_Bayesian_data_fft(x_old_real, x_old_imag, x_new_real, x_new_imag):
    d = int(np.sqrt(len(x_old_real)))
    x_old_real = np.reshape(x_old_real, (d,d))
    x_old_real = np.expand_dims(x_old_real,2)
    x_old_imag = np.reshape(x_old_imag, (d,d))
    x_old_imag = np.expand_dims(x_old_imag,2)
    x_new_real = np.reshape(x_new_real, (d,d))
    x_new_real = np.expand_dims(x_new_real,2)
    x_new_imag = np.reshape(x_new_imag, (d,d))
    x_new_imag = np.expand_dims(x_new_imag,2)
    
    return np.concatenate((x_old_real, x_old_imag, x_new_real, x_new_imag),axis = 2)

    


class DeepFilter():
    def __init__(self, input_shape, output_shape, lr = 2e-4, n0_filters = 64, max_filters = 512, n0_filter_zise = 4, apply_dropout = True):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.generator_optimizer = tf.keras.optimizers.Adam(lr,  beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(lr,  beta_1=0.5)
        # ---- compiling the generator ---- #
        down_stack = []
        up_stack = []
        down_steps = int(np.log2(self.input_shape[0]))
        up_steps = int(np.log2(self.output_shape[0]))
        for ii in range(down_steps):
            n_filters = n0_filters*(2**ii)
            if n_filters > max_filters:
                n_filters = max_filters
            if ii == 0:
                down_stack.append(downsample(n_filters, n0_filter_zise, apply_batchnorm=False, apply_dropout=False))
            else:
                down_stack.append(downsample(n_filters, 4, apply_batchnorm=False, apply_dropout=False))
        for ii in range(up_steps):
            n_filters = n0_filters*(2**ii)
            if n_filters > max_filters:
                n_filters = max_filters
            if ii == 0:
                up_stack.append(upsample(n_filters, 4, apply_batchnorm=False, apply_dropout=False))
            else:
                up_stack.append(upsample(n_filters, 4, apply_batchnorm=False, apply_dropout=apply_dropout))
            
        up_stack = reversed(up_stack)
        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(self.output_shape[2], n0_filter_zise, strides=2, padding='same', kernel_initializer=initializer,kernel_regularizer=regularizers.l2(0.001),  activation='tanh')
        concat = tf.keras.layers.Concatenate()
        generator_input = tf.keras.layers.Input(shape=self.input_shape)
        x = generator_input
        # Downsampling through the model
        skips = []
        for down in down_stack:
          x = down(x)
          skips.append(x)
        skips = reversed(skips[:-1])
        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
          x = up(x)
          x = concat([x, skip])
        x = last(x)
        self.generator = tf.keras.Model(inputs=generator_input, outputs=x)
        
        # ---- compiling the discriminator ---- #
        discriminator_inp = tf.keras.layers.Input(shape=self.input_shape, name='input_image')
        discriminator_tar = tf.keras.layers.Input(shape=self.output_shape, name='target_image')
        if down_steps == up_steps:
            x = tf.keras.layers.concatenate([discriminator_inp, discriminator_tar], axis=3)
        elif down_steps>up_steps:
            n_steps = down_steps - up_steps
            tar_x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(discriminator_tar)
            for ii in range(n_steps-1):
                tar_x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(tar_x)
            x = tf.keras.layers.concatenate([discriminator_inp, tar_x], axis=3)
        elif down_steps<up_steps:
            n_steps = up_steps - down_steps
            inp_x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(discriminator_inp)
            for ii in range(n_steps-1):
                inp_x = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(inp_x)
            x = tf.keras.layers.concatenate([inp_x, discriminator_tar], axis=3)
        x = downsample(64, 4, apply_batchnorm=False, apply_dropout=False)(x)  
        x = downsample(128, 4, apply_batchnorm=False, apply_dropout=apply_dropout)(x)
        #x = downsample(256, 4, apply_batchnorm=False, apply_dropout=apply_dropout)(x)
        x = tf.keras.layers.ZeroPadding2D()(x) # (bs, 10, 10, 256)
        x = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False, kernel_regularizer=regularizers.l2(0.001))(x)
        #x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.ZeroPadding2D()(x) 
        x = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer, kernel_regularizer=regularizers.l2(0.001))(x) 
        self.discriminator = tf.keras.Model(inputs=[discriminator_inp, discriminator_tar], outputs=x)
        
    def train_step(self, x, y, L = 30, loss_type = "l1"):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator([x], training=True)
            disc_real_output = self.discriminator([x, y], training=True)
            disc_generated_output = self.discriminator([x, gen_output], training=True)
            gen_loss = generator_loss(disc_generated_output, gen_output, y, LAMBDA = L, loss_type = loss_type)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
        generator_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,  self.discriminator.trainable_variables))
        
        


class DeepFilter1D():
    def __init__(self, input_shape, output_shape, lr = 2e-4, n0_filters = 32, max_filters = 512, n0_filter_zise = 4, apply_dropout = True):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.generator_optimizer = tf.keras.optimizers.Adam(lr)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(lr)
        # ---- compiling the generator ---- #
        down_stack = []
        up_stack = []
        down_steps = int(np.log2(self.input_shape[0]))
        up_steps = int(np.log2(self.output_shape[0]))
        for ii in range(down_steps):
            n_filters = n0_filters*(2**ii)
            if n_filters > max_filters:
                n_filters = max_filters
            if ii == 0:
                down_stack.append(downsample1d(n_filters, n0_filter_zise, apply_batchnorm=False, apply_dropout=False))
            else:
                down_stack.append(downsample1d(n_filters, 4, apply_batchnorm=False, apply_dropout=False))
        for ii in range(up_steps):
            n_filters = n0_filters*(2**ii)
            if n_filters > max_filters:
                n_filters = max_filters
            if ii == 0:
                up_stack.append(upsample1d(n_filters, 4, apply_batchnorm=False, apply_dropout=False))
            else:
                up_stack.append(upsample1d(n_filters, 4, apply_batchnorm=False, apply_dropout=apply_dropout))
            
        up_stack = reversed(up_stack)
        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv1DTranspose(self.output_shape[1], n0_filter_zise, strides=2, padding='same', kernel_initializer=initializer,kernel_regularizer=regularizers.l2(0.001),  activation='tanh')
        concat = tf.keras.layers.Concatenate()
        generator_input = tf.keras.layers.Input(shape=self.input_shape)
        x = generator_input
        # Downsampling through the model
        skips = []
        for down in down_stack:
          x = down(x)
          skips.append(x)
        skips = reversed(skips[:-1])
        # Upsampling and establishing the skip connections
        for up, skip in zip(up_stack, skips):
          x = up(x)
          x = concat([x, skip])
        x = last(x)
        self.generator = tf.keras.Model(inputs=generator_input, outputs=x)
        
        # ---- compiling the discriminator ---- #
        discriminator_inp = tf.keras.layers.Input(shape=self.input_shape, name='input_image')
        discriminator_tar = tf.keras.layers.Input(shape=self.output_shape, name='target_image')
        if down_steps == up_steps:
            x = tf.keras.layers.concatenate([discriminator_inp, discriminator_tar], axis=2)
        elif down_steps>up_steps:
            n_steps = down_steps - up_steps
            tar_x = tf.keras.layers.AveragePooling1D(pool_size=(2))(discriminator_tar)
            for ii in range(n_steps-1):
                tar_x = tf.keras.layers.AveragePooling1D(pool_size=(2))(tar_x)
            x = tf.keras.layers.concatenate([discriminator_inp, tar_x], axis=2)
        elif down_steps<up_steps:
            n_steps = up_steps - down_steps
            inp_x = tf.keras.layers.AveragePooling1D(pool_size=(2))(discriminator_inp)
            for ii in range(n_steps-1):
                inp_x = tf.keras.layers.AveragePooling1D(pool_size=(2))(inp_x)
            x = tf.keras.layers.concatenate([inp_x, discriminator_tar], axis=2)
        x = downsample1d(64, 4, apply_batchnorm=False, apply_dropout=False)(x)  
        x = downsample1d(128, 4, apply_batchnorm=False, apply_dropout=apply_dropout)(x)
        x = downsample1d(256, 4, apply_batchnorm=False, apply_dropout=apply_dropout)(x)
        x = tf.keras.layers.ZeroPadding1D()(x) # (bs, 10, 10, 256)
        x = tf.keras.layers.Conv1D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False, kernel_regularizer=regularizers.l2(0.001))(x)
        #x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.ZeroPadding1D()(x) 
        x = tf.keras.layers.Conv1D(1, 4, strides=1, kernel_initializer=initializer, kernel_regularizer=regularizers.l2(0.001))(x) 
        self.discriminator = tf.keras.Model(inputs=[discriminator_inp, discriminator_tar], outputs=x)
        
    def train_step(self, x, y, L = 30, loss_type = "l1"):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = self.generator([x], training=True)
            disc_real_output = self.discriminator([x, y], training=True)
            disc_generated_output = self.discriminator([x, gen_output], training=True)
            gen_loss = generator_loss(disc_generated_output, gen_output, y, LAMBDA = L, loss_type = loss_type)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
        generator_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,  self.discriminator.trainable_variables))
        
  
   

        
       





  



