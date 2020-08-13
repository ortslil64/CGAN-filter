#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np






def random_walk(steps = 5000000, random_measurments = False, measure_time = 600):
    H = 0.5
    A = 1.0
    data_x = np.zeros(steps)
    data_y = np.zeros(steps)
    idxs = np.ones(steps)
    Q = 0.0001
    R = 0.01
    idxs[0] = 0
    for i in range(steps-1):
        data_x[i+1] = A*data_x[i] + np.random.normal(0,np.sqrt(Q))
        if (i%measure_time == 0 and random_measurments == False) or (np.random.binomial(1,0.003) == 1 and random_measurments == True):
            data_y[i+1] = H*data_x[i+1] + np.random.normal(0,np.sqrt(R))
        else:
            data_y[i+1] = data_y[i]
            idxs[i+1] = 0
    return data_x, data_y, idxs

def linear_process_with_input(steps = 5000000):
    H = 0.5
    A = 0.8
    data_x = np.zeros(steps)
    data_y = np.zeros(steps)
    data_u = np.zeros(steps)
    idxs = np.ones(steps)
    Q = 0.001
    R = 0.01
    for i in range(steps-1):
        data_u[i+1] = np.random.normal(0,0.0001) #0.2*np.sin(0.002*i)
        data_x[i+1] = A*data_x[i] + data_u[i+1] + np.random.normal(0,np.sqrt(Q))
        if i%100 == 0:
            data_y[i+1] = H*data_x[i+1] + np.random.normal(0,np.sqrt(R))
        else:
            data_y[i+1] = data_y[i]
            idxs[i+1] = 0
        

    return data_x, data_y, idxs, data_u

def non_linear_process(steps = 5000000):
    data_x = np.zeros(steps)
    data_y = np.zeros(steps)
    idxs = np.ones(steps)
    Q = 0.0001
    R = 0.1
    for i in range(steps-1):
        data_x[i+1] = data_x[i]+ np.exp(-data_x[i]**2) + np.random.normal(0.1,np.sqrt(Q))*np.random.randint(-1,2)
        if i%400 == 0:
            data_y[i+1] = data_x[i+1] + np.random.normal(0,np.sqrt(R))
        else:
            data_y[i+1] = data_y[i]
            idxs[i+1] = 0
    return data_x, data_y, idxs

def fly_process(steps = 5000000, random_measurments = False):
    data_x = np.zeros((steps,3))
    data_z = np.zeros((steps,2))
    idxs = np.ones(steps)
    v = 0.0002
    omega = 2*np.pi/(128*128)
    Q = 0.00001
    R = 0.0001
    data_x[0,0] = np.random.normal(5,1)
    data_x[0,1] = np.random.normal(5,1)
    data_x[0,2] = np.random.uniform(0,np.pi)
    for i in range(steps-1):
        data_x[i+1,0] = data_x[i,0] + v*np.cos(data_x[i,2]) +np.random.normal(0,np.sqrt(Q))
        data_x[i+1,1] = data_x[i,1] + v*np.sin(data_x[i,2]) +np.random.normal(0,np.sqrt(Q))
        data_x[i+1,2] = data_x[i,2] + omega +np.random.normal(0,0.01)
        if (i%5 == 0 and random_measurments == False) or (np.random.binomial(1,0.003) == 1 and random_measurments == True):
            data_z[i+1,0] = np.linalg.norm(data_x[i+1,0:2]) + np.random.normal(0,np.sqrt(R))
            data_z[i+1,1] = np.arctan2(data_x[i+1,1],data_x[i+1,0]) + np.random.normal(0,np.sqrt(R))
        else:
            data_z[i+1,:] = data_z[i,:]
            idxs[i+1] = 0
    return data_x, data_z, idxs


def non_linear_model(x, Q):
    return np.sqrt(np.sin(x)) + np.random.normal(0.0, np.sqrt(Q))

def get_dataset_SMC():
    data_x = np.zeros(5000000)
    data_y = np.zeros(5000000)
    idxs = np.ones(5000000)
    Q = 0.0001
    R = 0.01
    for i in range(5000000-1):
        data_x[i+1] = non_linear_model(data_x[i],Q) 
        if i%600 == 0:
            data_y[i+1] = measurment_model(data_x[i+1],R)
        else:
            data_y[i+1] = data_y[i]
            idxs[i+1] = 0
    return data_x, data_y, idxs

def measurment_model(x, R):
    z = np.cos(x) + np.random.normal(0.0, np.sqrt(R))
    return z

def measurment_likelihood(x, z, beta):
    x_hat = np.arccos(z)
    L = np.exp(-beta*np.power((x_hat-x),2))
    return L
    


