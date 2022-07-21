#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy

import numpy as np
import scipy as scipy

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


from scipy.fft import fft
#from scipy.stats import wasserstein_distance
from scipy import ndimage

from marvin.tools import Maps
from marvin.tools.image import Image
from marvin.utils.general.images import showImage


# In[ ]:


#data = '8249-3702'

def ew_washing(data, snr = 5, test = False):
    # check oiii ivar and mask
    # ivar > 1
    # Mask != 1
    # OIII flux / flux error > 3

    maps = Maps(data, bintype='SPX', template='MILESHC-MASTARSSP')

    st_vel = maps.stellar_vel.value
    oiii_ew = maps.emline_gew_oiii_5008

    ew_value = copy.deepcopy(oiii_ew.value)
    ew_ivar = oiii_ew.ivar
    ew_snr = oiii_ew.snr
    ew_flag = oiii_ew.pixmask.bits

    ew_row = ew_value.shape[0]
    ew_col = ew_value.shape[1]

    mean = np.mean(ew_value)
    sd = np.std(ew_value)

    # Mask the EW  map
    for i in range(ew_row):
        for j in range(ew_col):
            # exclude IVAR = 0: 
            # https://www.sdss.org/dr17/manga/manga-tutorials/manga-faq/#WhydoyououtputIVAR(inversevariance)insteadoferrors?
            if ew_ivar[i][j] <= 1:
                ew_value[i][j] = 0
            # exclude S/N < setting
            elif ew_snr[i][j] <= snr:
                ew_value[i][j] = 0
            # exclude any flag for spaxel error
            elif len(ew_flag[i][j]) > 0:
                ew_value[i][j] = 0
             # Exclude 3-sigma data
            elif ew_value[i][j] <= mean - 3 * sd:
                ew_value[i][j] = 0
            elif ew_value[i][j] >= mean + 3 * sd:
                ew_value[i][j] = 0
            else:
                pass

    # Plot output
    if test == True:
        fig = plt.figure(figsize=(10, 7))
        rows = 2
        columns = 2
        fig.add_subplot(rows, columns, 1)
        plt.imshow(ew_value, cmap = 'viridis')

        fig.add_subplot(rows, columns, 2)
        plt.imshow(ew_ivar, cmap = 'viridis')

        fig.add_subplot(rows, columns, 3)
        plt.imshow(ew_snr, cmap = 'viridis')

        fig.add_subplot(rows, columns, 4)
        plt.imshow(st_vel, cmap = 'viridis')

        plt.show()
    else:
        pass
            
    return ew_value

def ew_integrat(ew_value):

    # Intergrate along r direction:
    curve = []
    for k in np.arange(0,360,dphi):
        bins = []
        for i in range(ew_row):
            for j in range(ew_col):
                if phi[i][j] >= k and phi[i][j] <= k+dphi and ew_value[i][j] != 0:
                    bins.append(ew_value[i][j])
                else:
                    pass
        if len(bins) == 0:
            curve.append(0)
        else:
            curve.append(sum(bins)/len(bins))

    # Plot N cycle
    for i in range(0, cycle):
                ew_cycle = curve + list(curve)

    # Gaussian smooth
    ew_smo = scipy.ndimage.gaussian_filter(ew_cycle, sigma = smooth)

    # Intepolate to N * 360 degree
    x = np.linspace(0,len(ew_smo),len(ew_smo))
    y = ew_smo
    x2 = np.linspace(0,len(ew_smo),360*cycle)
    f_linear = scipy.interpolate.interp1d(x, y, kind='linear')
    intp_EW = f_linear(x2)

    
    return intp_EW


# In[ ]:


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


# In[ ]:


def plot_something(data, save = True):
    # Plot Image, OIII, Star_v, Gas_v, Curve, FFT. 
    # in a 3X2 subplots
    
    fig = plt.figure(figsize=(10, 7))
    
    
    '''
    HEADER_PATH = '/Users/runquanguan/Documents/dapall-v3_1_1-3.1.0.fits'
    hdul = fits.open(HEADER_PATH)
    hdu = hdul[1].data
    plateifu = hdu['PLATEIFU']
    petro_angle = hdu['NSA_ELPETRO_PHI']
    sersic_angle = hdu['NSA_SERSIC_PHI']

    dapall_index = list(plateifu).index(data)
    p_slope = np.tan(petro_angle[dapall_index])
    s_slope = np.tan(sersic_angle[dapall_index])
    '''

    maps = Maps(data, bintype='SPX', template='MILESHC-MASTARSSP')
    
    st_vel = maps.stellar_vel
    ha_vel = maps.emline_gvel_ha_6564
    oiii_ew = maps.emline_gew_oiii_5008
    
    #np.flipud()


    x_tik = oiii_ew.shape[0]/4
    y_tik = oiii_ew.shape[1]/4
    
    
    phi = maps.spx_ellcoo_elliptical_azimuth.value
    
    for maps in [oiii_ew]:
        row = maps.value.shape[0]
        col = maps.value.shape[1]
        mean = np.mean(maps.value)
        sd = np.std(maps.value)
        for i in range(row):
            for j in range(col):
                # exclude IVAR = 0: 
                # https://www.sdss.org/dr17/manga/manga-tutorials/manga-faq/#WhydoyououtputIVAR(inversevariance)insteadoferrors?
                if maps.ivar[i][j] <= 1:
                    maps.value[i][j] = np.nan
                # exclude S/N < setting
                elif maps.snr[i][j] <= 10:
                    maps.value[i][j] = np.nan
                # exclude any flag for spaxel error
                elif len(maps.pixmask.bits[i][j]) > 0:
                    maps.value[i][j] = np.nan
                # Exclude 3-sigma data
                elif maps.value[i][j] <= mean - 3 * sd:
                    maps.value[i][j] = np.nan
                elif maps.value[i][j] >= mean + 3 * sd:
                    maps.value[i][j] = np.nan
                else:
                    pass

    for maps in [st_vel, ha_vel]:
        row = maps.value.shape[0]
        col = maps.value.shape[1]
        row = maps.value.shape[0]
        col = maps.value.shape[1]
        mean = np.mean(maps.value)
        sd = np.std(maps.value)
        for i in range(row):
            for j in range(col):
                if maps.value[i][j] == 0:
                    maps.value[i][j] = np.nan
                elif maps.ivar[i][j] <= 1:
                    maps.value[i][j] = np.nan
                elif maps.value[i][j] <= mean - 4 * sd:
                    maps.value[i][j] = np.nan
                elif maps.value[i][j] >= mean + 4 * sd:
                    maps.value[i][j] = np.nan
                else:
                    pass

    st_vel = np.flipud(st_vel.value)
    ha_vel = np.flipud(ha_vel.value)
    oiii_ew = np.flipud(oiii_ew.value)

    axis_indicat = np.zeros(phi.shape)

    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            if phi[i][j] >= 88 and phi[i][j] <= 92:
                axis_indicat[i][j] = 10
            elif phi[i][j] >= 268 and phi[i][j] <= 272:
                axis_indicat[i][j] = 10
            else:
                axis_indicat[i][j] = np.nan

    axis_indicat = np.flipud(axis_indicat)
                
    rows = 2
    columns = 3
    
    # Plot the 1st image
    fig.add_subplot(rows, columns, 1)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    plt.title("Image")
    im = Image(data)
    spaxel_len = 281*2
    factor = spaxel_len / axis_indicat.shape[0]
    large_mask = ndimage.zoom(axis_indicat, factor, order=0)
    # large_mask[large_mask > 0] = 1
    mask = large_mask
    im = showImage(plateifu = data)
    masked = np.ma.masked_where(mask == 0, mask)
    plt.xlabel('arcsec')
    plt.ylabel('arcsec')
    plt.imshow(im, interpolation='none',extent=[-25,25,-25,25])
    plt.imshow(masked, cmap = 'autumn', interpolation='none', alpha=0.3,extent=[-25,25,-25,25])

    # Plot the 2nd OIII EW map
    fig.add_subplot(rows, columns, 4)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    plt.title("OIII EW Map")
    plt.xlabel('arcsec')
    plt.ylabel('arcsec')
    plt.imshow(oiii_ew, cmap = 'viridis',extent=[-1*x_tik, x_tik, -1*y_tik, y_tik])
    plt.imshow(axis_indicat, cmap = 'autumn',extent=[-1*x_tik, x_tik, -1*y_tik, y_tik], interpolation='none', alpha=0.3)

    
    # Plot the 3rd Star v
    fig.add_subplot(rows, columns, 2)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    plt.title("Stellar Velocity")
    plt.xlabel('arcsec')
    plt.ylabel('arcsec')
    plt.imshow(st_vel, cmap = 'viridis',extent=[-1*x_tik, x_tik, -1*y_tik, y_tik])
    plt.imshow(axis_indicat, cmap = 'autumn',extent=[-1*x_tik, x_tik, -1*y_tik, y_tik], interpolation='none', alpha=0.3)

    
    # Plot the 4th Gas v
    fig.add_subplot(rows, columns, 5)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    plt.title("Ha Velocity")
    plt.xlabel('arcsec')
    plt.ylabel('arcsec')
    plt.imshow(ha_vel, cmap = 'viridis',extent=[-1*x_tik, x_tik, -1*y_tik, y_tik])
    plt.imshow(axis_indicat, cmap = 'autumn', extent=[-1*x_tik, x_tik, -1*y_tik, y_tik], interpolation='none', alpha=0.3)

    
    # Plot 5rd EW dR
    fig.add_subplot(rows, columns, 3)
    fig.set_figheight(15)
    fig.set_figwidth(30)
    plt.title("OIII EW dR")
    plt.xlabel('degree')
    plt.ylabel('EW/spaxel')
    curve = integra_r_dir(data)
    plt.plot(curve)


        
    # Plot 6th FTT
    fig.add_subplot(rows, columns, 6)
    fig.set_figheight(15)
    fig.set_figwidth(30)
    plt.title("Fourier Frequency")
    plt.xlabel('Hz')
    result = abs(fft(curve))   
    x = np.linspace(0.5, 15, 29)
    plt.plot(x, result[1:30])



    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.1, 
                    hspace=0.2)
    

    if save == True:
        plt.savefig(data+'_vis.png')
        plt.cla()
    else:
        plt.show()

def plot_full(data, save = True):
    # Plot Image, OIII, Star_v, Gas_v, Curve, FFT. 
    # in a 3X2 subplots
    
    fig = plt.figure(figsize=(10, 7))
    
    
    '''
    HEADER_PATH = '/Users/runquanguan/Documents/dapall-v3_1_1-3.1.0.fits'
    hdul = fits.open(HEADER_PATH)
    hdu = hdul[1].data
    plateifu = hdu['PLATEIFU']
    petro_angle = hdu['NSA_ELPETRO_PHI']
    sersic_angle = hdu['NSA_SERSIC_PHI']

    dapall_index = list(plateifu).index(data)
    p_slope = np.tan(petro_angle[dapall_index])
    s_slope = np.tan(sersic_angle[dapall_index])
    '''

    maps = Maps(data, bintype='SPX', template='MILESHC-MASTARSSP')
    
    st_vel = maps.stellar_vel
    ha_vel = maps.emline_gvel_ha_6564
    oiii_ew = maps.emline_gew_oiii_5008
    
    #np.flipud()


    x_tik = oiii_ew.shape[0]/4
    y_tik = oiii_ew.shape[1]/4
    
    
    phi = maps.spx_ellcoo_elliptical_azimuth.value

    for maps in [st_vel, ha_vel, oiii_ew]:
        row = maps.value.shape[0]
        col = maps.value.shape[1]
        mean = np.mean(maps.value)
        sd = np.std(maps.value)
        for i in range(row):
            for j in range(col):
                if maps.value[i][j] == 0:
                    maps.value[i][j] = np.nan
                elif maps.ivar[i][j] <= 1:
                    maps.value[i][j] = np.nan
                elif maps.value[i][j] <= mean - 4 * sd:
                    maps.value[i][j] = np.nan
                elif maps.value[i][j] >= mean + 4 * sd:
                    maps.value[i][j] = np.nan
                else:
                    pass


    st_vel = np.flipud(st_vel.value)
    ha_vel = np.flipud(ha_vel.value)
    oiii_ew = np.flipud(oiii_ew.value)

    axis_indicat = np.zeros(phi.shape)

    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            if phi[i][j] >= 88 and phi[i][j] <= 92:
                axis_indicat[i][j] = 10
            elif phi[i][j] >= 268 and phi[i][j] <= 272:
                axis_indicat[i][j] = 10
            else:
                axis_indicat[i][j] = np.nan

    axis_indicat = np.flipud(axis_indicat)
                
    rows = 2
    columns = 3
    
    # Plot the 1st image
    fig.add_subplot(rows, columns, 1)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    plt.title("Image")
    im = Image(data)
    spaxel_len = 281*2
    factor = spaxel_len / axis_indicat.shape[0]
    large_mask = ndimage.zoom(axis_indicat, factor, order=0)
    # large_mask[large_mask > 0] = 1
    mask = large_mask
    im = showImage(plateifu = data)
    masked = np.ma.masked_where(mask == 0, mask)
    plt.xlabel('arcsec')
    plt.ylabel('arcsec')
    plt.imshow(im, interpolation='none',extent=[-25,25,-25,25])
    plt.imshow(masked, cmap = 'autumn', interpolation='none', alpha=0.3,extent=[-25,25,-25,25])

    # Plot the 2nd OIII EW map
    fig.add_subplot(rows, columns, 4)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    plt.title("OIII EW Map")
    plt.xlabel('arcsec')
    plt.ylabel('arcsec')
    plt.imshow(oiii_ew, cmap = 'viridis',extent=[-1*x_tik, x_tik, -1*y_tik, y_tik])
    plt.imshow(axis_indicat, cmap = 'autumn',extent=[-1*x_tik, x_tik, -1*y_tik, y_tik], interpolation='none', alpha=0.3)

    
    # Plot the 3rd Star v
    fig.add_subplot(rows, columns, 2)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    plt.title("Stellar Velocity")
    plt.xlabel('arcsec')
    plt.ylabel('arcsec')
    plt.imshow(st_vel, cmap = 'viridis',extent=[-1*x_tik, x_tik, -1*y_tik, y_tik])
    plt.imshow(axis_indicat, cmap = 'autumn',extent=[-1*x_tik, x_tik, -1*y_tik, y_tik], interpolation='none', alpha=0.3)

    
    # Plot the 4th Gas v
    fig.add_subplot(rows, columns, 5)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    plt.title("Ha Velocity")
    plt.xlabel('arcsec')
    plt.ylabel('arcsec')
    plt.imshow(ha_vel, cmap = 'viridis',extent=[-1*x_tik, x_tik, -1*y_tik, y_tik])
    plt.imshow(axis_indicat, cmap = 'autumn', extent=[-1*x_tik, x_tik, -1*y_tik, y_tik], interpolation='none', alpha=0.3)

    
    # Plot 5rd EW dR
    fig.add_subplot(rows, columns, 3)
    fig.set_figheight(15)
    fig.set_figwidth(30)
    plt.title("OIII EW dR")
    plt.xlabel('degree')
    plt.ylabel('EW/spaxel')
    curve = integra_r_dir(data)
    plt.plot(curve)


        
    # Plot 6th FTT
    fig.add_subplot(rows, columns, 6)
    fig.set_figheight(15)
    fig.set_figwidth(30)
    plt.title("Fourier Frequency")
    plt.xlabel('Hz')
    result = abs(fft(curve))   
    x = np.linspace(0.5, 15, 29)
    plt.plot(x, result[1:30])



    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.1, 
                    hspace=0.2)
    

    if save == True:
        plt.savefig(data+'_full.png')
        plt.cla()
    else:
        plt.show()