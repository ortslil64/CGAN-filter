from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras import regularizers

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)






def downsample(filters, size, apply_batchnorm=True, strides=2, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same',
                             kernel_initializer=initializer, use_bias=False, 
                             kernel_regularizer=regularizers.l2(0.001)))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())
    
  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.4))  
  result.add(tf.keras.layers.LeakyReLU())

  return result




def upsample(filters, size, apply_dropout=False, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False,
                                    kernel_regularizer=regularizers.l2(0.001)))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.4))

  result.add(tf.keras.layers.ReLU())

  return result

def downsample1d(filters, size, apply_batchnorm=True, strides=2, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv1D(filters, size, strides=strides, padding='same',
                             kernel_initializer=initializer, use_bias=False, 
                             kernel_regularizer=regularizers.l2(0.001)))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())
    
  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.4))  
  result.add(tf.keras.layers.LeakyReLU())

  return result




def upsample1d(filters, size, apply_dropout=False, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv1DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False,
                                    kernel_regularizer=regularizers.l2(0.001)))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.4))

  result.add(tf.keras.layers.ReLU())

  return result

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

def generator_loss(disc_generated_output, gen_output, target, LAMBDA, loss_type = "l1"):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error
  if loss_type == "l1":
      l_loss = tf.reduce_mean(tf.abs(target - gen_output))
  if loss_type == "l2":
      l_loss = tf.reduce_mean(tf.pow(target - gen_output,2))
  if loss_type == "l0.5":
      l_loss = tf.pow(tf.reduce_sum(tf.pow(tf.abs(target - gen_output),0.5)),2)
      

  total_gen_loss = gan_loss + (LAMBDA * l_loss)

  return total_gen_loss

def sampling(args):
    z_mean = args
    batch = tf.keras.backend.shape(z_mean)[0]
    dim = tf.keras.backend.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = tf.keras.backend.random_normal(mean=0.0,stddev=1.0, shape=(batch, dim))
    return z_mean + tf.keras.backend.exp(0.5 ) * epsilon




