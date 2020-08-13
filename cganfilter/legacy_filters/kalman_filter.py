#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np



def Kalman_filter(z, idxs=None, A = 1.0 , H = 0.5, Q = 0.01, R = 0.1):
    x = np.zeros_like(z)
    p = np.zeros_like(z)
    if idxs is None:
        for i in range(len(x)-1):
            x[i+1] = A*x[i]
            p[i+1] = (A**2)*p[i] + Q
            y = z[i+1] - H*x[i+1]
            s = (H**2)*p[i+1]+R
            k = p[i+1]*H/s
            x[i+1] = x[i+1] + k*y
            p[i+1] = (1-k*H)*p[i+1]
    else:
        for i in range(len(x)-1):
            x[i+1] = A*x[i]
            p[i+1] = (A**2)*p[i] + Q
            if idxs[i] == 1:
                y = z[i+1] - H*x[i+1]
                s = (H**2)*p[i+1]+R
                k = p[i+1]*H/s
                x[i+1] = x[i+1] + k*y
                p[i+1] = (1-k*H)*p[i+1]
    
    return x,p

def Kalman_predictor(z,idxs,u = None, A = 0.8 , H = 0.5, Q = 0.01, R = 0.1):
    x = np.zeros_like(z)
    p = np.zeros_like(z)
    x_pred = np.zeros_like(z)
    p_pred = np.zeros_like(z)
    for i in range(len(x)-1):
        if u is None:
            x[i+1] = A*x[i]
        else:
            x[i+1] = A*x[i] + u[i]
        p[i+1] = (A**2)*p[i] + Q
        if idxs[i] == 1:
            y = z[i+1] - H*x[i+1]
            s = (H**2)*p[i+1]+R
            k = p[i+1]*H/s
            x[i+1] = x[i+1] + k*y
            p[i+1] = (1-k*H)*p[i+1]
    x_pred[0] = x[-1] 
    p_pred[0] = p[-1]       
    for i in range(len(x_pred)-1):
        if u is None:
            x_pred[i+1] = A*x_pred[i]
        else:
            x_pred[i+1] = A*x_pred[i] + u[i] 
        p_pred[i+1] = (A**2)*p_pred[i] + Q

    return x_pred,p_pred


def Kalman_smoother(z, idxs = None,u = None, x0= None, A = 1.0 , H = 0.5, Q = 0.0001, R = 0.01, hist = 100):
    x = np.zeros_like(z)
    p = 0.01*np.ones_like(z)
    if x0 is None:
        x[0] = z[0]/H
    else:
        x[0] = x0
    for i in range(len(x)-1):
        if u is None:
            x[i+1] = A*x[i]
        else:
            x[i+1] = A*x[i] + u[i]
        p[i+1] = (A**2)*p[i] + Q
        if idxs[i+1] == 1:
            if len(x) - i > hist:
                hist_temp = hist - 1
            else:
                hist_temp = len(x) - i - 1
            for j in range(hist_temp):
                if idxs[i+1+j] == 1:
                    y = z[i+1+j] - H*x[i+1]
                    s = (H**2)*p[i+1]+R+j*Q
                    k = p[i+1]*H/s
                    x[i+1] = x[i+1] + k*y
                    p[i+1] = (1-k*H)*p[i+1]
    
    return x,p

def Kalman_smoother_unknown_input(z, idxs):
    x = np.zeros((len(z),2))
    p = np.ones((len(z),2,2))
    A = np.array([[0.8,1],[0,1]])
    H = np.array([0.5,0])
    Q = np.diag([0.001,0.0])
    R = 0.01
    for i in range(len(x)-1):
        x[i+1] = A.dot(x[i]) 
        p[i+1] = A.dot(p[i].dot(A.T)) + Q
        if idxs[i] == 1:
            y = z[i+1] - H.dot(x[i+1].T)
            s = H.dot(p[i+1].dot(H.T)) + R 
            k = p[i+1].dot(H.T.dot(1/s))
            x[i+1] = x[i+1] + k.dot(y)
            p[i+1] = (np.eye(2)-k.dot(H)).dot(p[i+1])
    
    return x[:,0],p[:,0,0]


