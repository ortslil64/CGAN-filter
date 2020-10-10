#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from spo_dataset.spo_generator import add_circle_mag
from sklearn.cluster import KMeans
import cv2

def params2img(a,b,c,theta,image_shape):
    image = np.zeros(image_shape)
    for jj in range(2):
        pts = [[int(c[jj][0] + 0.5*a[jj]*np.cos(theta[jj]) - 0.5*b[jj]*np.sin(theta[jj])), int( c[jj][1] + 0.5*a[jj]*np.sin(theta[jj]) + 0.5*b[jj]*np.cos(theta[jj]))],
               [int(c[jj][0] - 0.5*a[jj]*np.cos(theta[jj]) - 0.5*b[jj]*np.sin(theta[jj])),int( c[jj][1] - 0.5*a[jj]*np.sin(theta[jj]) + 0.5*b[jj]*np.cos(theta[jj]))],
               [int(c[jj][0] - 0.5*a[jj]*np.cos(theta[jj]) + 0.5*b[jj]*np.sin(theta[jj])),int( c[jj][1] - 0.5*a[jj]*np.sin(theta[jj]) - 0.5*b[jj]*np.cos(theta[jj]))],
               [int(c[jj][0] + 0.5*a[jj]*np.cos(theta[jj]) + 0.5*b[jj]*np.sin(theta[jj])),int( c[jj][1] + 0.5*a[jj]*np.sin(theta[jj]) - 0.5*b[jj]*np.cos(theta[jj]))]]
        pts = np.array(pts)
        pts = pts.reshape((-1, 1, 2)) 
        color = (255) 
  
        # Line thickness of 8 px 
        thickness = 2
        isClosed = True
        image = cv2.polylines(image, [pts],  
                  isClosed, color,  
                  thickness) 
        image = cv2.fillPoly(image, [pts], 255)
        
    return image/255

def params2obs(a,b,c,theta,image_shape, var):
    image = np.zeros(image_shape)
    for jj in range(2):
        pts = [[int(c[jj][0] + 0.5*a[jj]*np.cos(theta[jj]) - 0.5*b[jj]*np.sin(theta[jj])), int( c[jj][1] + 0.5*a[jj]*np.sin(theta[jj]) + 0.5*b[jj]*np.cos(theta[jj]))],
               [int(c[jj][0] - 0.5*a[jj]*np.cos(theta[jj]) - 0.5*b[jj]*np.sin(theta[jj])),int( c[jj][1] - 0.5*a[jj]*np.sin(theta[jj]) + 0.5*b[jj]*np.cos(theta[jj]))],
               [int(c[jj][0] - 0.5*a[jj]*np.cos(theta[jj]) + 0.5*b[jj]*np.sin(theta[jj])),int( c[jj][1] - 0.5*a[jj]*np.sin(theta[jj]) - 0.5*b[jj]*np.cos(theta[jj]))],
               [int(c[jj][0] + 0.5*a[jj]*np.cos(theta[jj]) + 0.5*b[jj]*np.sin(theta[jj])),int( c[jj][1] + 0.5*a[jj]*np.sin(theta[jj]) - 0.5*b[jj]*np.cos(theta[jj]))]]
        pts = np.array(pts)
        pts = pts.reshape((-1, 1, 2)) 
        color = (255) 
  
        # Line thickness of 8 px 
        thickness = 2
        isClosed = True
        image = cv2.polylines(image, [pts],  
                  isClosed, color,  
                  thickness) 
        image = cv2.fillPoly(image, [pts], 255)
        
        noise = np.random.normal(0, var, image_shape)
        noise[noise < 0] = 0
        noise[noise > 255] = 255
        noisy_image = 0.5*image + noise
        noisy_image[noisy_image > 255] = 255
    return noisy_image/255

class ParticleFilter():
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
        for jj in range(No):
            for ii in range(Np):
                if initial_pose is None:
                    self.X[ii, jj, 0] = np.random.randint(0, self.width)
                    self.X[ii, jj, 1] = np.random.randint(0, self.height)
                else:
                    self.X[ii, jj, 0] = initial_pose[jj][0] + np.random.randint(-50, 50)
                    if self.X[ii, jj, 0] >= self.width:
                        self.X[ii, jj, 0] = self.width
                    self.X[ii, jj, 1] = initial_pose[jj][1] + np.random.randint(-50, 50)
                    if self.X[ii, jj, 1] >= self.height:
                        self.X[ii, jj, 1] = self.height 
                self.X[ii, jj, 2] = np.random.randint(-3, 4)
                self.X[ii, jj, 3] = np.random.randint(-3, 4)
        
    def propogate(self):
        for jj in range(self.No):
            for ii in range(self.Np):
                self.X[ii, jj, 0] = self.X[ii, jj, 0] + self.X[ii, jj, 2] + np.random.randint(-1, 2)
                if self.X[ii, jj, 0] >= self.width:
                    self.X[ii, jj, 2] = -self.X[ii, jj, 2]
                    self.X[ii, jj, 0] = self.width
                if self.X[ii, jj, 0] <= 0:
                    self.X[ii, jj, 2] = -self.X[ii, jj, 2]
                    self.X[ii, jj, 0] = 0
                
                self.X[ii, jj, 1] = self.X[ii, jj, 1] + self.X[ii, jj, 3] + np.random.randint(-1, 2)
                if self.X[ii, jj, 1] >= self.height:
                    self.X[ii, jj, 3] = -self.X[ii, jj, 3]
                    self.X[ii, jj, 1] = self.height
                if self.X[ii, jj, 1] <= 0:
                    self.X[ii, jj, 3] = -self.X[ii, jj, 3]
                    self.X[ii, jj, 1] = 0
                    
                self.X[ii, jj, 2] = self.X[ii, jj, 2] + np.random.randint(-1, 2)
                self.X[ii, jj, 3] = self.X[ii, jj, 3] + np.random.randint(-1, 2)
                if self.X[ii, jj, 2] > 4:
                    self.X[ii, jj, 2] = 4
                if self.X[ii, jj, 2] < -4:
                    self.X[ii, jj, 2] = -4
                if self.X[ii, jj, 3] > 4:
                    self.X[ii, jj, 3] = 4
                if self.X[ii, jj, 3] < -4:
                    self.X[ii, jj, 3] = -4
                

    def update(self, z):
        for ii in range(self.Np):
            ref_img_temp = self.ref_img.copy()
            #ref_img_temp = np.zeros_like(self.ref_img)
            for jj in range(self.No):
                ref_img_temp = add_circle_mag(ref_img_temp, self.X[ii, jj, 0:2], self.radiuses[jj])
                #ref_img_temp = cv2.circle(ref_img_temp,(int(self.X[ii, jj, 0]), int(self.X[ii, jj, 1])),self.radiuses[jj],(255,255,255),-1) 
                ref_img_temp = ref_img_temp.astype(np.float32)/255
            #L = np.mean(np.abs(z.astype(np.float32) * ref_img_temp))
            L = np.exp(-self.beta * np.mean(np.power(z.astype(np.float32) -ref_img_temp,2)))
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
                    self.X[ii, jj, 0] = self.X[ii, jj, 0]  + np.random.randint(-6, 7)
                    if self.X[ii, jj, 0] >= self.width:
                        self.X[ii, jj, 0] = self.width
                    if self.X[ii, jj, 0] <= 0:
                        self.X[ii, jj, 0] = 0
                    
                    self.X[ii, jj, 1] = self.X[ii, jj, 1] +  np.random.randint(-6, 7)
                    if self.X[ii, jj, 1] >= self.height:
                        self.X[ii, jj, 1] = self.height
                    if self.X[ii, jj, 1] <= 0:
                        self.X[ii, jj, 1] = 0
                        
                    self.X[ii, jj, 2] = self.X[ii, jj, 2] + np.random.randint(-1, 2)
                    self.X[ii, jj, 3] = self.X[ii, jj, 3] + np.random.randint(-1, 2)
            return True
        else:
            return False
        
    def step(self, z):
        self.propogate()
        self.update(z)
        r = self.resample()
        if r: self.update(z)
        X_output = np.zeros_like(self.ref_img)
        # X = []
        # for jj in range(self.No):
        #     X.append(self.X[:, jj, 0:2])
        # X = np.concatenate(X, axis = 0)
        # kmeans = KMeans(n_clusters=self.No, random_state=0).fit(X)
        for jj in range(self.No):
            #idx = np.argmax(self.W)
            xy = self.X[:, jj, 0:2].T.dot(self.W)
            #xy = self.X[idx, jj, 0:2]
            X_output = cv2.circle(X_output,(int(xy[0]), int(xy[1])),self.radiuses[jj],(255,255,255),-1) 
        return X_output
    
class ParticleFilterRect():
    def __init__(self, Np,var, img_shape, beta):
        self.a = [15,13]
        self.b = [10,40]
        self.c = [[0,10],[100,100]]
        self.theta = [0.5, 1.2]
        self.v = [[1,2],[3,2]]
        self.omega = [0.1, -0.05]
        self.Np = Np
        self.beta = beta
        self.width = img_shape[0]
        self.height = img_shape[1]
        self.image_shape = img_shape
        self.W = np.ones(Np)/Np
        self.X = np.zeros((Np, 2, 3))
        self.var = var
        for jj in range(2):
            for ii in range(Np):
                self.X[ii, jj, 0] = self.c[jj][0] + np.random.randint(-50, 50)
                if self.X[ii, jj, 0] >= self.width:
                    self.X[ii, jj, 0] = self.width
                self.X[ii, jj, 1] = self.c[jj][1] + np.random.randint(-50, 50)
                if self.X[ii, jj, 1] >= self.height:
                    self.X[ii, jj, 1] = self.height 
                self.X[ii, jj, 2] = self.theta[jj] + np.random.normal(0, 3)
        
    def propogate(self):
        for jj in range(2):
            for ii in range(self.Np):
                self.X[ii, jj, 0] = self.X[ii, jj, 0] + self.v[jj][0]*np.random.choice([-1,1]) + np.random.randint(-5, 6)
                if self.X[ii, jj, 0] >= self.width:
                    self.v[jj][0]  = -self.v[jj][0] 
                    self.X[ii, jj, 0] = self.width
                if self.X[ii, jj, 0] <= 0:
                    self.v[jj][0]  = -self.v[jj][0] 
                    self.X[ii, jj, 0] = 0
                
                self.X[ii, jj, 1] = self.X[ii, jj, 1] + self.v[jj][1]*np.random.choice([-1,1])  + np.random.randint(-5, 6)
                if self.X[ii, jj, 1] >= self.height:
                    self.v[jj][1] = -self.v[jj][1]
                    self.X[ii, jj, 1] = self.height
                if self.X[ii, jj, 1] <= 0:
                    self.v[jj][1] = -self.v[jj][1]
                    self.X[ii, jj, 1] = 0
                    
                self.X[ii, jj, 2] = self.X[ii, jj, 2] + self.omega[jj] + np.random.normal(0, 0.1)
                
                

    def update(self, z):
        for ii in range(self.Np):
            c = [[self.X[ii, 0, 0],self.X[ii, 0, 1]],
                 [self.X[ii, 1, 0],self.X[ii, 1, 1]]]
            theta = [self.X[ii, 0, 2],self.X[ii, 1, 2]]
            z_temp = params2obs(self.a, self.b, c, theta, self.image_shape, self.var)
            L = np.exp(-self.beta * np.mean(np.abs(z -z_temp)))
            self.W[ii] = self.W[ii] * L
        self.W = self.W/np.sum(self.W)
        
    def resample(self):
        Neff = 1/np.sum(self.W**2)
        if Neff < self.Np/3:
            print("Resampling...")
            indxes = np.random.choice(self.Np, self.Np, p = self.W)
            self.X = self.X[indxes]
            for jj in range(2):
                for ii in range(self.Np):
                    self.X[ii, jj, 0] = self.X[ii, jj, 0]  + np.random.randint(-6, 7)
                    if self.X[ii, jj, 0] >= self.width:
                        self.X[ii, jj, 0] = self.width
                    if self.X[ii, jj, 0] <= 0:
                        self.X[ii, jj, 0] = 0
                    
                    self.X[ii, jj, 1] = self.X[ii, jj, 1] +  np.random.randint(-6, 7)
                    if self.X[ii, jj, 1] >= self.height:
                        self.X[ii, jj, 1] = self.height
                    if self.X[ii, jj, 1] <= 0:
                        self.X[ii, jj, 1] = 0
                        
                    self.X[ii, jj, 2] = self.X[ii, jj, 2] + np.random.normal(0, 0.1)
            return True
        else:
            return False
        
    def step(self, z):
        self.propogate()
        self.update(z)
        r = self.resample()
        if r: self.update(z)
        idx = np.argmax(self.W)
        c = [[self.X[:, 0, 0].T.dot(self.W),self.X[:, 0, 1].T.dot(self.W)],
             [self.X[:, 1, 0].T.dot(self.W),self.X[:, 1, 1].T.dot(self.W)]]
        
        theta = [self.X[idx, 0, 2],self.X[idx, 1, 2]]
        X_output = params2img(self.a, self.b, c, theta, self.image_shape)
        
        return X_output
    
    
    
    
    