#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from spo_dataset.spo_generator import add_circle_mag
from sklearn.cluster import KMeans
import cv2
from scipy.stats import multivariate_normal

def KL_img(img1, img2, bins = 100):
    H1,_ = np.histogram(img1.ravel(),bins,(0,1)) 
    H2,_ = np.histogram(img2.ravel(),bins,(0,1)) 
    H1 = H1/np.sum(H1)
    H2 = H2/np.sum(H1)
    D = np.sum(H1*H2)
    return D

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

def likelihood_rect(a,b,c,theta,image):    
    z = np.argwhere(image > 0.5)
    
    R = np.array([[np.cos(theta[0]), -np.sin(theta[0])],
                  [np.sin(theta[0]), np.cos(theta[0])]])
    cov = (0.5**2)*np.array([[b[0]**2,0],
                             [0, a[0]**2]], dtype=np.float64)
    rot_cov = R.dot(cov.dot(R.T))
    Mu = np.array(c[0])
    L1 = multivariate_normal.pdf(z, mean=Mu, cov = rot_cov)
    
    R = np.array([[np.cos(theta[1]), -np.sin(theta[1])],
                  [np.sin(theta[1]), np.cos(theta[1])]])
    cov = (0.5**2)*np.array([[b[1]**2,0],
                             [0, a[1]**2]], dtype=np.float64)
    rot_cov = R.dot(cov.dot(R.T))
    Mu = np.array(c[1])
    L2 = multivariate_normal.pdf(z, mean=Mu, cov = rot_cov)
    
    L = np.sum(np.log(L1 + L2))
        
        
    return L


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
        self.V = [[1,2], [2,1]]
        for jj in range(No):
            for ii in range(Np):
                if initial_pose is None:
                    self.X[ii, jj, 0] = np.random.randint(0, self.width)
                    self.X[ii, jj, 1] = np.random.randint(0, self.height)
                else:
                    self.X[ii, jj, 0] = initial_pose[jj][0] + np.random.uniform(-2, 2)
                    if self.X[ii, jj, 0] >= self.width:
                        self.X[ii, jj, 0] = self.width
                    self.X[ii, jj, 1] = initial_pose[jj][1] + np.random.uniform(-2, 2)
                    if self.X[ii, jj, 1] >= self.height:
                        self.X[ii, jj, 1] = self.height 
                self.X[ii, jj, 2] = self.V[jj][0] 
                self.X[ii, jj, 3] = self.V[jj][1] 
        
    def propogate(self):
        for jj in range(self.No):
            for ii in range(self.Np):
                self.X[ii, jj, 0] = self.X[ii, jj, 0] + self.X[ii, jj, 2] + np.random.uniform(-2, 2)
                if self.X[ii, jj, 0] >= self.width:
                    self.X[ii, jj, 2] = -self.X[ii, jj, 2]
                    self.X[ii, jj, 0] = self.width
                if self.X[ii, jj, 0] <= 0:
                    self.X[ii, jj, 2] = -self.X[ii, jj, 2]
                    self.X[ii, jj, 0] = 0
                
                self.X[ii, jj, 1] = self.X[ii, jj, 1] + self.X[ii, jj, 3] + np.random.uniform(-2, 2)
                if self.X[ii, jj, 1] >= self.height:
                    self.X[ii, jj, 3] = -self.X[ii, jj, 3]
                    self.X[ii, jj, 1] = self.height
                if self.X[ii, jj, 1] <= 0:
                    self.X[ii, jj, 3] = -self.X[ii, jj, 3]
                    self.X[ii, jj, 1] = 0
                    
                
                

    def update(self, z):
        for ii in range(self.Np):
            ref_img_temp = self.ref_img.copy()
            #ref_img_temp = np.zeros_like(self.ref_img)
            for jj in range(self.No):
                ref_img_temp = add_circle_mag(ref_img_temp, self.X[ii, jj, 0:2], self.radiuses[jj])
                #s = cv2.circle(np.zeros_like(self.ref_img),(int(self.X[ii, jj, 0]), int(self.X[ii, jj, 1])),self.radiuses[jj],(255,255,255),-1) / 255.0
                ref_img_temp = ref_img_temp.astype(np.float64)/255
            #L = np.mean(np.abs(z.astype(np.float32) * ref_img_temp))
            L = np.exp(-self.beta * np.mean(np.power(z.astype(np.float64) -ref_img_temp,2)))
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
            idx = np.argmax(self.W)
            xy = self.X[:, jj, 0:2].T.dot(self.W)
            xy = self.X[idx, jj, 0:2]
            X_output = cv2.circle(X_output,(int(xy[0]), int(xy[1])),self.radiuses[jj],(255,255,255),-1) 
        return X_output
    
class ParticleFilterRect():
    def __init__(self, Np,var, img_shape, beta, Ber):
        size_en = 1
        self.Ber = Ber
        self.a = [50 + size_en,15 + size_en]
        self.b = [25 + size_en,25 + size_en]
        self.c = [[50,20],[100,100]]
        self.theta = [0.5, 1.2]
        self.v = [[1,2],[-2,-1]]
        self.omega = [0.05, -0.05]
        self.Np = Np
        self.beta = beta
        self.width = img_shape[0]
        self.height = img_shape[1]
        self.image_shape = img_shape
        self.W = np.ones(Np)/Np
        self.X = np.zeros((Np, 2, 5))
        self.var = var
        for jj in range(2):
            for ii in range(Np):
                self.X[ii, jj, 0] = self.c[jj][0] + np.random.uniform(-10, 10)
                if self.X[ii, jj, 0] >= self.width:
                    self.X[ii, jj, 0] = self.width
                self.X[ii, jj, 1] = self.c[jj][1] + np.random.uniform(-10, 10)
                if self.X[ii, jj, 1] >= self.height:
                    self.X[ii, jj, 1] = self.height 
                self.X[ii, jj, 2] = self.theta[jj] + np.random.normal(0, 0.01)
                self.X[ii, jj, 3] = self.v[jj][0] 
                self.X[ii, jj, 4] = self.v[jj][1] 
        
    def propogate(self):
        for jj in range(2):
            for ii in range(self.Np):
                self.X[ii, jj, 0] = self.X[ii, jj, 0] + self.X[ii, jj, 3] + np.random.uniform(-3, 3)
                if self.X[ii, jj, 0] >= self.width:
                    self.X[ii, jj, 3]  = -self.X[ii, jj, 3]
                    self.X[ii, jj, 0] = self.width
                if self.X[ii, jj, 0] <= 0:
                    self.X[ii, jj, 3]  = -self.X[ii, jj, 3]
                    self.X[ii, jj, 0] = 0
                
                self.X[ii, jj, 1] = self.X[ii, jj, 1] + self.X[ii, jj, 4]  +np.random.uniform(-3, 3)
                if self.X[ii, jj, 1] >= self.height:
                    self.X[ii, jj, 4] = -self.X[ii, jj, 4]
                    self.X[ii, jj, 1] = self.height
                if self.X[ii, jj, 1] <= 0:
                    self.X[ii, jj, 4] = -self.X[ii, jj, 4]
                    self.X[ii, jj, 1] = 0
                    
                self.X[ii, jj, 2] = self.X[ii, jj, 2] + self.omega[jj] + np.random.normal(0, 0.001)
                

    def update(self, z):
        z[z<0.5] = 0
        for ii in range(self.Np):
            c = [[self.X[ii, 0, 0],self.X[ii, 0, 1]],
                 [self.X[ii, 1, 0],self.X[ii, 1, 1]]]
            theta = [self.X[ii, 0, 2],self.X[ii, 1, 2]]
            #z_temp = params2obs(self.a, self.b, c, theta, self.image_shape, self.var)
            z_temp = params2img(self.a, self.b, c, theta, self.image_shape)
            #L = np.exp(-self.beta * np.mean(np.abs(z -z_temp)))
            #z_temp_inv = -1*z_temp.astype(np.float32) +1
            #L_inv = np.exp(-self.beta*np.mean(z * z_temp_inv))
            L =  np.log(1+self.beta*np.mean(z * z_temp))
            #L = likelihood_rect(self.a, self.b, c, theta, z)
            #L = np.exp(self.beta*KL_img(z, z_temp, 50))
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
                    self.X[ii, jj, 0] = self.X[ii, jj, 0]  + np.random.uniform(-1, 1)
                    if self.X[ii, jj, 0] >= self.width:
                        self.X[ii, jj, 0] = self.width
                    if self.X[ii, jj, 0] <= 0:
                        self.X[ii, jj, 0] = 0
                    
                    self.X[ii, jj, 1] = self.X[ii, jj, 1] +  np.random.uniform(-1, 1)
                    if self.X[ii, jj, 1] >= self.height:
                        self.X[ii, jj, 1] = self.height
                    if self.X[ii, jj, 1] <= 0:
                        self.X[ii, jj, 1] = 0
                        
                    self.X[ii, jj, 2] = self.X[ii, jj, 2] + np.random.normal(0, 0.001)

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
        # c = [[self.X[idx, 0, 0],self.X[idx, 0, 1]],
        #      [self.X[idx, 1, 0],self.X[idx, 1, 1]]]
        
        theta = [self.X[:, 0, 2].T.dot(self.W),self.X[:, 1, 2].T.dot(self.W)]
        #theta = [self.X[idx, 0, 2],self.X[idx, 1, 2]]
        X_output = params2img(self.a, self.b, c, theta, self.image_shape)
        
        return X_output
    
    
    
    
    
class ParticleFilter_deep():
    def __init__(self, Np, No, ref_img, radiuses, initial_pose, beta, likelihood):
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
        self.likelihood = likelihood
        for jj in range(No):
            for ii in range(Np):
                if initial_pose is None:
                    self.X[ii, jj, 0] = np.random.randint(0, self.width)
                    self.X[ii, jj, 1] = np.random.randint(0, self.height)
                else:
                    self.X[ii, jj, 0] = initial_pose[jj][0] + np.random.uniform(-2, 2)
                    if self.X[ii, jj, 0] >= self.width:
                        self.X[ii, jj, 0] = self.width
                    self.X[ii, jj, 1] = initial_pose[jj][1] + np.random.uniform(-2, 2)
                    if self.X[ii, jj, 1] >= self.height:
                        self.X[ii, jj, 1] = self.height 
                self.X[ii, jj, 2] = self.V[jj][0] 
                self.X[ii, jj, 3] = self.V[jj][1] 
        
    def propogate(self):
        for jj in range(self.No):
            for ii in range(self.Np):
                self.X[ii, jj, 0] = self.X[ii, jj, 0] + self.X[ii, jj, 2] + np.random.uniform(-3, 3)
                if self.X[ii, jj, 0] >= self.width:
                    self.X[ii, jj, 2] = -self.X[ii, jj, 2]
                    self.X[ii, jj, 0] = self.width
                if self.X[ii, jj, 0] <= 0:
                    self.X[ii, jj, 2] = -self.X[ii, jj, 2]
                    self.X[ii, jj, 0] = 0
                
                self.X[ii, jj, 1] = self.X[ii, jj, 1] + self.X[ii, jj, 3] + np.random.uniform(-3, 3)
                if self.X[ii, jj, 1] >= self.height:
                    self.X[ii, jj, 3] = -self.X[ii, jj, 3]
                    self.X[ii, jj, 1] = self.height
                if self.X[ii, jj, 1] <= 0:
                    self.X[ii, jj, 3] = -self.X[ii, jj, 3]
                    self.X[ii, jj, 1] = 0
                


    
    def update(self, z):
        x_hat_df_like = self.likelihood.estimate(z)
        x_hat_df_like = x_hat_df_like[0,:,:,0] 
        x_hat_df_like[x_hat_df_like<0.5] = 0
        for ii in range(self.Np):
            y_hat =  np.zeros_like(self.ref_img)
            for jj in range(self.No):
                y_hat = cv2.circle(y_hat,(int(self.X[ii, jj, 0]), int(self.X[ii, jj, 1])),self.radiuses[jj],(255,255,255),-1) / 255.0
            
            L = np.exp(-self.beta * np.mean(np.power(y_hat -x_hat_df_like,2)))

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


class ParticleFilter_inverse():
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
                    self.X[ii, jj, 0] = initial_pose[jj][0] + np.random.uniform(-2, 2)
                    if self.X[ii, jj, 0] >= self.width:
                        self.X[ii, jj, 0] = self.width
                    self.X[ii, jj, 1] = initial_pose[jj][1] + np.random.uniform(-2, 2)
                    if self.X[ii, jj, 1] >= self.height:
                        self.X[ii, jj, 1] = self.height 
                self.X[ii, jj, 2] = self.V[jj][0] 
                self.X[ii, jj, 3] = self.V[jj][1] 
        
    def propogate(self):
        for jj in range(self.No):
            for ii in range(self.Np):
                self.X[ii, jj, 0] = self.X[ii, jj, 0] + self.X[ii, jj, 2] + np.random.uniform(-2, 2)
                if self.X[ii, jj, 0] >= self.width:
                    self.X[ii, jj, 2] = -self.X[ii, jj, 2]
                    self.X[ii, jj, 0] = self.width
                if self.X[ii, jj, 0] <= 0:
                    self.X[ii, jj, 2] = -self.X[ii, jj, 2]
                    self.X[ii, jj, 0] = 0
                
                self.X[ii, jj, 1] = self.X[ii, jj, 1] + self.X[ii, jj, 3] + np.random.uniform(-2, 2)
                if self.X[ii, jj, 1] >= self.height:
                    self.X[ii, jj, 3] = -self.X[ii, jj, 3]
                    self.X[ii, jj, 1] = self.height
                if self.X[ii, jj, 1] <= 0:
                    self.X[ii, jj, 3] = -self.X[ii, jj, 3]
                    self.X[ii, jj, 1] = 0
                    
                
                

    def update(self, z):
        for ii in range(self.Np):
            ref_img_temp = np.zeros_like(self.ref_img)
            for jj in range(self.No):
                #ref_img_temp = add_circle_mag(ref_img_temp, self.X[ii, jj, 0:2], self.radiuses[jj])
                ref_img_temp = cv2.circle(ref_img_temp,(int(self.X[ii, jj, 0]), int(self.X[ii, jj, 1])),self.radiuses[jj],(255,255,255),-1) / 255.0
            L = np.exp(-self.beta * np.mean(np.power(z.astype(np.float64) -ref_img_temp,2)))
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
        X_output = self.ref_img.copy()
        for jj in range(self.No):
            #idx = np.argmax(self.W)
            xy = self.X[:, jj, 0:2].T.dot(self.W)
            #xy = self.X[idx, jj, 0:2]
            X_output = add_circle_mag(X_output, (int(xy[0]), int(xy[1])), self.radiuses[jj])
        return X_output
    
    
    
    
    
    
    
