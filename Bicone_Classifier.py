#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import cv2
import scipy as scipy
import matplotlib.pyplot as plt
from astropy.io import fits
import polarTransform
from marvin.tools import Maps
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


def generate_profile_histogram(plateifu, method = 'max', smooth = 5):
    
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
    EW_SMO_TW = list(np.append(EW_SMO, EW_SMO))

    if method == 'max':
        # limited the array to exactly one cycle: but from max to max
        max_index = max(EW_SMO_TW, default = 0) # find max value
        start_index = [i for i, n in enumerate(EW_SMO_TW) if n == max_index][0] # find index of 1st max
        end_index = [i for i, n in enumerate(EW_SMO_TW) if n == max_index][1] # find index of 2nd max
        trunc_EW = np.array(EW_SMO_TW[start_index:end_index]) # truncate from 1st to 2nd max
    elif method == 'min':
        # limited the array to exactly one cycle: but from max to max
        min_index = min(EW_SMO_TW, default = 0) # find min value
        start_index = [i for i, n in enumerate(EW_SMO_TW) if n == min_index][0] # find index of 1st min
        end_index = [i for i, n in enumerate(EW_SMO_TW) if n == min_index][1] # find index of 2nd min
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


# In[ ]:


def calculate_maximum(test):
    
    # Calculate the loss between the test galaxy and the 17 bicone galaxy
    
    test_EW_hist = test[0]
    test_plateifu = test[1]
    
    # Load examples
    BC_PATH = '/Users/runquanguan/Documents/Research/MaNGA-AGN/Pipeline&Instrction/obvious_bicone_feature_position.fits'
    FOLDER = '/Users/runquanguan/Documents/Research/MaNGA-AGN/Data/'
    
    hdul = fits.open(BC_PATH)
    hdu = hdul[1].data
    
    total_loss = []
    for data in hdu:
        plateifu = str(data[0])
        FILENAME = 'manga-' + plateifu + '-MAPS-SPX-MILESHC-MASTARSSP.fits'
        FILE_PATH = FOLDER+FILENAME
        # Define arrays
        BC_EW_hist = generate_profile_histogram(FILE_PATH)[0]

        # Find the loss using Earth Moving Distance
        loss = wasserstein_distance(test_EW_hist, BC_EW_hist)
        total_loss.append(loss)
    
    if test_plateifu in hdu['PLATEIFU']:
        final_loss = sum(total_loss)/(hdu.shape[0]-1)
    else:
        final_loss = sum(total_loss)/hdu.shape[0]
        
    return final_loss
    
    
    
    