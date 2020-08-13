#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np




def particle_filter_fly(z,x0, idxs=None,v = 0.0002 , omega = 2*np.pi/(128*128), Q = 0.00001, R = 0.0001, Np = 2000):
    steps = len(z)
    x = np.zeros((steps,3))
    p = np.zeros((steps,3,3))
    X = np.zeros((Np,3))
    X[:,0] = np.random.normal(x0[0],0.1,Np)
    X[:,1] = np.random.normal(x0[1],0.1,Np)
    X[:,2] = np.random.normal(x0[2],0.01,Np)
    W = np.ones(Np)/Np
    if idxs is None:
        for i in range(len(x)-1):
            # ---- prediction ---- #
            X[:,0] = X[:,0] + v*np.cos(X[:,2]) +np.random.normal(0,np.sqrt(Q),Np)
            X[:,1] = X[:,1] + v*np.sin(X[:,2]) +np.random.normal(0,np.sqrt(Q),Np)
            X[:,2] = X[:,2] + omega +np.random.normal(0,0.01,Np)
            # ---- Likelihood ---- #
            r_hat = np.linalg.norm(X[:,0:2], axis = 1)
            theta_hat = np.arctan2(X[:,1],X[:,0])
            r_res = np.power(r_hat - z[i+1,0],2)
            theta_res1 = np.abs(theta_hat - z[i+1,1])
            theta_res2 = np.abs(2*np.pi - theta_hat + z[i+1,1])
            theta_res = np.power(np.min([theta_res1,theta_res2],axis = 0),2)
            W = np.exp(-np.multiply(r_res, theta_res)/R)
            W = W/np.sum(W)
            print(W)
            # ---- Resample ---- #
            X_idxs = np.random.choice(Np,Np, p = W)
            X = X[X_idxs]
            # ---- Estimate ---- #
            x[i+1,...] = np.mean(X, axis = 0)
            p[i+1,...] = np.cov(X.T)
    else:
        for i in range(len(x)-1):
            # ---- prediction ---- #
            X[:,0] = X[:,0] + v*np.cos(X[:,2]) +np.random.normal(0,np.sqrt(Q),Np)
            X[:,1] = X[:,1] + v*np.sin(X[:,2]) +np.random.normal(0,np.sqrt(Q),Np)
            X[:,2] = X[:,2] + omega +np.random.normal(0,0.01,Np)
            if idxs[i+1] == 1:
                # ---- Likelihood ---- #
                r_hat = np.linalg.norm(X[:,0:2], axis = 1)
                theta_hat = np.arctan2(X[:,1],X[:,0])
                r_res = np.power(r_hat - z[i+1,0],2)
                theta_res1 = np.abs(theta_hat - z[i+1,1])
                theta_res2 = np.abs(2*np.pi - theta_hat + z[i+1,1])
                theta_res = np.power(np.min([theta_res1,theta_res2],axis = 0),2)
                W = np.exp(-0.1*np.multiply(r_res, theta_res)/R)
                W = W/np.sum(W)
                # ---- Resample ---- #
                X_idxs = np.random.choice(Np, Np,p= W)
                X = X[X_idxs]
            # ---- Estimate ---- #
            x[i+1,...] = np.mean(X, axis = 0)
            p[i+1,...] = np.cov(X.T)
    
    return x,p


def particle_filter_non_linear(z,x0, idxs ,  Q = 0.0001, R = 0.0001, Np = 1000):
    steps = len(z)
    x = np.zeros(steps)
    p = np.zeros(steps)
    X = np.random.normal(x0,0.01,Np)
    W = np.ones(Np)/Np
    for i in range(len(x)-1):
        # ---- prediction ---- #
        
        X = X +np.exp(-X**2) + np.random.normal(0.1,5*np.sqrt(Q))*np.random.randint(-1,2,Np)
        if idxs[i+1] == 1:
            # ---- Likelihood ---- #
            z_hat =  X
                             
            z_res = z_hat - z[i+1]
            W = np.exp(-10*np.power(z_res,2)) 
            
            W = W/np.sum(W)
            if np.sum(np.isnan(W)) > 0:
                W = np.ones(Np)/Np
            # ---- Resample ---- #
            X_idxs = np.random.choice(Np,Np, p = W)
            X = X[X_idxs]
        # ---- Estimate ---- #
        x[i+1,...] = np.mean(X, axis = 0)
        p[i+1,...] = np.cov(X.T)
    
    return x,p



    
    
    
