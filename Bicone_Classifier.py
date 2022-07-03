#!/usr/bin/env python
# coding: utf-8

# In[4]:

from doctest import NORMALIZE_WHITESPACE
import cv2
import copy

import numpy as np
import scipy as scipy

import matplotlib.pyplot as plt
from astropy.io import fits

from scipy.fft import fft
from scipy.stats import wasserstein_distance

import polarTransform
from marvin.tools import Maps


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


def polar_EW(plateifu, ring = False, method = None , smooth = 10, cycle = 2):
    
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
    
    if ring == False:
        r1 = 0
        r2 = round(w/16)
    elif ring == True:
        r1 = round(w/2)
        r2 = round(3*w/4)
    else:
        r1 = 0
        r2 = round(w/2)

    polarImage, ptSettings = polarTransform.convertToPolarImage(oiii_ew, 
                                                                initialRadius=r1, finalRadius=r2, 
                                                                center=[round(w/2), round(h/2)])
    
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
        trunc_EW = EW_SMO_TW
    
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
        
    
    return intp_EW





def ellip_gen(plateifu = '7443-1902'):

    maps = Maps(plateifu, bintype='SPX', template='MILESHC-MASTARSSP')
    oiii_ew = maps.emline_gew_oiii_5008.value

    # Ellipticity Properties
    phi = maps.spx_ellcoo_elliptical_azimuth.value.round(decimals=2)
    r_re = maps.spx_ellcoo_r_re.value
    
    # We select the ring of r = 0.8 er ~ 1.2 er for EW
    # Make a new array of (EW, phi_er)
    ew_r_comb = np.array((oiii_ew, r_re)) 
    #Tanspose the array to make each element in form of (EW, phi_er)

    return ew_r_comb, phi

def ellip_ring_curve(ellip, in_r = 0.9, out_r = 1.1, sig = 5, cycle = 1, smooth = 3):

    ring = copy.deepcopy(ellip)

    # Inherent the output of the ellip_gen function
    ew_re_comb = ring[0]
    theta = ring[1]
    

    # Make non-ring data = 0, then output the EW only
    # First sort the data as pairs of (oiii, r_er) 
    for i in  ew_re_comb.T :
        # j is the r_er
        for j in i:
            # set zeros outside of the range 
            if j[1] < in_r or j[1] > out_r:
                j[0] = 0
            else:
                pass

    # Output: zeros except for Ring of EW

    # From now on, we only care about the EW distribution along the phi direction
    # So we 
    # 1. make a tuple in the format of (phi, EW).
    # 2. Select non-zero EW elements.
    # 3. Interpolate the phi vs. EW to 500
    # 4. make the curve EW curve plot

    # Sort the data as pairs of (phi), (oiii) 
    theta_EW = np.array((theta, ew_re_comb[0]))

    EW_stacked = []
 
    # Sort the data as pairs of (phi, oiii) 
    for i in theta_EW.T:
        # j[1] would be the oiii
        for j in i:
            if j[1] != 0:
                EW_stacked.append(j)
            else:
                pass
    
    # Turn the list to array
    EW_curve = np.array(EW_stacked)


    # In case of 0 array data
    if len(EW_curve) >= 4:
        # Sort the EW along phi direction
        EW_sort = EW_curve[EW_curve[:, 0].argsort()]

        # Save the oiii result only
        EW_COL = EW_sort.T[1]
        
        # Exclude outlier using 3-sigma variant
        mean = np.mean(EW_COL)
        sd = np.std(EW_COL)
        EW_CLEAN = [x for x in EW_COL if (x >= mean - sig * sd)]
        EW_CLEAN = [x for x in EW_CLEAN if (x <= mean + sig * sd)]

        # Reinforce the asymmetric bicone by normalizing half sides
         # Normalization 
        mid_pt = round(len(EW_CLEAN)/2)

        norm_A = normalize(EW_CLEAN[0:mid_pt])
        norm_B = normalize(EW_CLEAN[mid_pt:])
        norm_EW = np.concatenate((norm_A, norm_B))
        
        # To better identify the feature, plot n cycles of the galaxy
        EW_CLEAN_N = []
        
        for i in range(0, cycle):
            EW_CLEAN_N = EW_CLEAN_N + list(norm_EW)

        # Smoothening the curve using Gaussian filter
        EW_SMO = scipy.ndimage.gaussian_filter(EW_CLEAN_N, sigma = smooth)

    else:
        EW_SMO = np.linspace(0,1,100)
        
    
    # Make them all to the same length through interpolation
    x = np.linspace(0,len(EW_SMO),len(EW_SMO))
    y = EW_SMO
    x2 = np.linspace(0,len(EW_SMO),360*cycle)
    f_linear = scipy.interpolate.interp1d(x, y)
    intp_EW = f_linear(x2)
        
    
    return intp_EW


def fourier_classifier(EW_curve, n_peak = 5):
    
    # 1. Set curve osillate around  y=0
    # 2. Take the FT result  from 1~50 because FT saturate at 0. 
    # Fourier Transform:
    curve = copy.deepcopy(EW_curve)
    
    # Max on the curve
    max_value = curve.max()
    max_index = list(curve).index(max_value)
    
    yf = np.abs(fft(curve)[1:30])

    output_y = copy.deepcopy(yf)

    # Strongest frequency
    peak_value = max(yf)
    peak_index = list(yf).index(peak_value)
    
    four_intensity = yf[3]
    ten_intensity = sum([yf[5], yf[7], yf[9]])

    residue = (max_index)-90 # to 360*2 angle, diff to zero, positive
    loop_residue = min([residue%180, abs(residue-180)])
    abs_residue = min([loop_residue%180, abs(loop_residue-180)])

    n_index = n_peak * -1

    yf.sort()
    loss = sum(np.diff(yf[n_index:]))
 
    
    return output_y, (max_index, peak_index) , loss, (four_intensity, ten_intensity), abs_residue