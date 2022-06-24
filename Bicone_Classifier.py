#!/usr/bin/env python
# coding: utf-8

# In[4]:


from cgi import test
from tkinter import EW
import numpy as np
import cv2
import scipy as scipy
import matplotlib.pyplot as plt
from astropy.io import fits
import polarTransform
from marvin.tools import Maps
from scipy.fft import fft
from scipy.stats import wasserstein_distance


# For later use:

# Find the nth occurance to resolve the repetition of '-'

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

# Normalize any given array to a given range
def normalize(arr, t_min = 0, t_max = 1):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)    
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


def generate_profile_histogram(plateifu, method = 'max', smooth = 10, cycle = 2):
    
    # Input: plateifu
    # method: output array begin with the max or min value in the array
    # smooth: Gaussian smoothening, the larger the smoother

    # Outout: Smoothened histogarm, plateifu

    
    # Load OIII MAPS from SDSS-MARVIN server
    maps = Maps(plateifu, bintype='SPX', template='MILESHC-MASTARSSP')
    oiii_ew = maps.emline_gew_oiii_5008.value

    # Transform to polar coordinate
  
    # Set center of image(just in case)
    w = oiii_ew.shape[0]
    h = oiii_ew.shape[1]

    polarImage, ptSettings = polarTransform.convertToPolarImage(oiii_ew, center=[round(w/2), round(h/2)])
    # Integrate the column
    EW_COL = [sum(x)/polarImage.T.shape[0] for x in zip(*polarImage.T)]

    # Exclude outlier using 3-sigma variant
    mean = np.mean(EW_COL)
    sd = np.std(EW_COL)
    EW_CLEAN = [x for x in EW_COL if (x >= mean - 2 * sd)]
    EW_CLEAN = [x for x in EW_CLEAN if (x <= mean + 2 * sd)]
    # Smoothening the curve using Gaussian filter
    EW_SMO = scipy.ndimage.gaussian_filter(EW_CLEAN, sigma = smooth)

    # To better identify the feature, plot two cycles of the galaxy
    EW_SMO_TW = []
    
    for i in range(0, cycle+1):
        EW_SMO_TW = EW_SMO_TW + list(EW_SMO)

    # Plot certain cycles

    if method == 'max':
        # limited the array to N cycles: but from max to max
        max_index = max(EW_SMO_TW, default = 0) # find max value
        start_index = [i for i, n in enumerate(EW_SMO_TW) if n == max_index][0] # find index of 1st max
        end_index = [i for i, n in enumerate(EW_SMO_TW) if n == max_index][cycle] # find index of 2nd max
        trunc_EW = np.array(EW_SMO_TW[start_index:end_index]) # truncate from 1st to 2nd max
    elif method == 'min':
        # limited the array to N cycles: but from max to max
        min_index = min(EW_SMO_TW, default = 0) # find min value
        start_index = [i for i, n in enumerate(EW_SMO_TW) if n == min_index][0] # find index of 1st min
        end_index = [i for i, n in enumerate(EW_SMO_TW) if n == min_index][cycle] # find index of 2nd min
        trunc_EW = np.array(EW_SMO_TW[start_index:end_index]) # truncate from 1st to 2nd min
    else:
        pass
    
    # Correct the 0 data error first:
    if len(trunc_EW) <= 4:
        trunc_EW = np.linspace(0,200,200)
    else:
        pass
    # Normalization 
    norm_EW = normalize(trunc_EW)
    
    # Make them all to the same length through interpolation
    x = np.linspace(0,len(norm_EW),len(norm_EW))
    y = norm_EW
    x2 = np.linspace(0,len(norm_EW),500)
    f_linear = scipy.interpolate.interp1d(x, y)
    intp_EW = f_linear(x2)
        
    
    return intp_EW, plateifu


def fourier_classifier(sample):
    
    # Importing data, use 2 cycles(default)
    test_EW_hist = sample[0]
    test_plateifu = sample[1]

    # 1. Set curve osillate around  y=0
    # 2. Take the FT result  from 1~50 because FT saturate at 0. 
    # Fourier Transform:
    yf = np.abs(fft(test_EW_hist-0.5)[1:50])
    yf.sort()
    loss = sum(np.diff(yf[-4:]))
  
    
    return loss, test_plateifu
    
    
    
    