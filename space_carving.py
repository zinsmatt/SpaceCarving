#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""

import scipy.io
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# Load camera matrices
data = scipy.io.loadmat("data/dino_Ps.mat")
data = data["P"]
projections = [data[0, i] for i in range(data.shape[1])]

# load images
files = sorted(glob.glob("data/*.ppm"))
images = []
for f in files:
    im = cv2.imread(f, cv2.IMREAD_UNCHANGED).astype(float)
    im /= 255
    images.append(im[:, :, ::-1])
    

# get silouhette from images
for im in images:
    temp = np.abs(im - [0.0, 0.0, 0.75])
    temp = np.sum(temp, axis=2)
    y, x = np.where(temp <= 1.1)
    im[y, x, :] = [0.0, 0.0, 0.0]
    im[im > 0] = 1.0
    im *= 255
    im = im.astype(np.uint8)

    kernel = np.ones((7, 7), np.uint8)
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)

#    plt.figure()
#    plt.imshow(im)
    
#%%
# create voxel grid
s = 40
x, y, z = np.mgrid[:s, :s, :s]
pts = np.vstack((x.flatten(), y.flatten(), z.flatten())).astype(float)
pts = pts.T
nb_points_init = pts.shape[0]
xmax, ymax, zmax = np.max(pts, axis=0)
pts[:, 0] /= xmax
pts[:, 1] /= ymax
pts[:, 2] /= zmax
center = pts.mean(axis=0)
pts -= center
pts /= 4

pts = np.vstack((pts.T, np.ones((1, nb_points_init))))

filled = []
for P, im in zip(projections, images):
    uvs = P @ pts
    uvs /= uvs[2, :]
    uvs = np.round(uvs).astype(int)
    fill = im[uvs[1, :], uvs[0, :]]
    filled.append(fill)
    






    
    


