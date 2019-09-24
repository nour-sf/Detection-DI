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


X = np.load('no_known_pl_set_gnorm.npy')
file = X[1]
img=plt.imshow(file.reshape(64,64))