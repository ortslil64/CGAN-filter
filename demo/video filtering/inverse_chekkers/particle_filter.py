#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from spo_dataset.spo_generator import add_circle_mag
from sklearn.cluster import KMeans
import cv2
from scipy.stats import multivariate_normal

class ParticleFilter_inv():
    def __init__(self, Np, No, ref_img, radiuses, initial_pose, beta):
        self.Np = Np
        self.No = No
        self.radiuses = radiuses
        self.beta = beta
        self.width = ref_img.shape[0]
        self.height = ref_img.shape[1]
        self.ref_img = ref_img
        self.W = np.ones(Np)/Np
        self.X = np.zeros((Np, No, 4))
        self.initial_pose = initial_pose
        self.V = [[1,2], [2,1]]
        for jj in range(No):
            for ii in range(Np):
                if initial_pose is None:
                    self.X[ii, jj, 0] = np.random.randint(0, self.width)
                    self.X[ii, jj, 1] = np.random.randint(0, self.height)
                else:
                    self.X[ii, jj, 0] = initial_pose[jj][0] + np.random.uniform(-20, 20)
                    if self.X[ii, jj, 0] >= self.width:
                        self.X[ii, jj, 0] = self.width
                    self.X[ii, jj, 1] = initial_pose[jj][1] + np.random.uniform(-20, 20)
                    if self.X[ii, jj, 1] >= self.height:
                        self.X[ii, jj, 1] = self.height 
                self.X[ii, jj, 2] = self.V[jj][0] 
                self.X[ii, jj, 3] = self.V[jj][1] 
        
    def propogate(self):
        for jj in range(self.No):
            for ii in range(self.Np):
                self.X[ii, jj, 0] = self.X[ii, jj, 0] + self.X[ii, jj, 2] + np.random.uniform(-1, 1)
                if self.X[ii, jj, 0] >= self.width:
                    self.X[ii, jj, 2] = -self.X[ii, jj, 2]
                    self.X[ii, jj, 0] = self.width
                if self.X[ii, jj, 0] <= 0:
                    self.X[ii, jj, 2] = -self.X[ii, jj, 2]
                    self.X[ii, jj, 0] = 0
                
                self.X[ii, jj, 1] = self.X[ii, jj, 1] + self.X[ii, jj, 3] + np.random.uniform(-1, 1)
                if self.X[ii, jj, 1] >= self.height:
                    self.X[ii, jj, 3] = -self.X[ii, jj, 3]
                    self.X[ii, jj, 1] = self.height
                if self.X[ii, jj, 1] <= 0:
                    self.X[ii, jj, 3] = -self.X[ii, jj, 3]
                    self.X[ii, jj, 1] = 0
                


    
    def update(self, z):
        for ii in range(self.Np):
            y_hat =  np.zeros_like(self.ref_img)
            for jj in range(self.No):
                y_hat = cv2.circle(y_hat,(int(self.X[ii, jj, 0]), int(self.X[ii, jj, 1])),self.radiuses[jj],(255,255,255),-1) / 255.0
            
            L = np.exp(-self.beta * np.mean(np.power(y_hat -z,2)))

            self.W[ii] = self.W[ii] * L
        self.W = self.W/np.sum(self.W)
        
    def resample(self):
        Neff = 1/np.sum(self.W**2)
        if Neff < self.Np/3:
            print("Resampling...")
            indxes = np.random.choice(self.Np, self.Np, p = self.W)
            self.X = self.X[indxes]
            for jj in range(self.No):
                for ii in range(self.Np):
                    self.X[ii, jj, 0] = self.X[ii, jj, 0]  + np.random.uniform(-4, 4)
                    if self.X[ii, jj, 0] >= self.width:
                        self.X[ii, jj, 0] = self.width
                    if self.X[ii, jj, 0] <= 0:
                        self.X[ii, jj, 0] = 0
                    
                    self.X[ii, jj, 1] = self.X[ii, jj, 1] +  np.random.uniform(-4, 4)
                    if self.X[ii, jj, 1] >= self.height:
                        self.X[ii, jj, 1] = self.height
                    if self.X[ii, jj, 1] <= 0:
                        self.X[ii, jj, 1] = 0
            self.W = np.ones(self.Np)/self.Np
            return True
        else:
            return False
        
    def step(self, z):
        self.propogate()
        self.update(z)
        r = self.resample()
        if r: self.update(z)
        X_output = self.ref_img.copy()
        for jj in range(self.No):
            #idx = np.argmax(self.W)
            xy = self.X[:, jj, 0:2].T.dot(self.W)
            #xy = self.X[idx, jj, 0:2]
            X_output = add_circle_mag(X_output, (int(xy[0]), int(xy[1])), self.radiuses[jj])
        return X_output/255
    