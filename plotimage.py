#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 10:15:35 2019

@author: nour
"""

from scipy.misc import imshow
import argparse
import os
import sys
import matplotlib.pyplot as plt
import numpy as np


X = np.load('no_known_pl_set_gnorm.npy')
file = X[2]
img=plt.imshow(file.reshape(64,64))

f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(X[0].reshape(64,64))
axarr[0,1].imshow(X[5].reshape(64,64))
axarr[1,0].imshow(X[10].reshape(64,64))
axarr[1,1].imshow(X[15].reshape(64,64))