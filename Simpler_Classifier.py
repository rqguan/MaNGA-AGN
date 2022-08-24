#!/usr/bin/env python
# coding: utf-8



import copy

import numpy as np
import scipy as scipy

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm


import math
import scipy.stats as stats
from scipy.fft import fft
#from scipy.stats import wasserstein_distance
from scipy import ndimage

from marvin.tools import Maps
from marvin.tools.image import Image
from marvin.utils.general.images import showImage

from matplotlib.colors import ListedColormap

from matplotlib.font_manager import FontProperties

from hyperopt import hp, fmin, tpe, space_eval

font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_style('normal')



viridis = cm.get_cmap('viridis', 256)
newcolors = viridis(np.linspace(0, 1, 256))
white = np.array([256/256, 256/256, 256/256, 1])
newcolors[:1, :] = white
viridis_vis = ListedColormap(newcolors)

plasma = cm.get_cmap('plasma', 256)
newcolors2 = plasma(np.linspace(0, 1, 256))
newcolors2[:1, :] = white
plasma_vis = ListedColormap(newcolors2)


#data = '8249-3702'

def normalize(arr, t_min = 0, t_max = 1):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)    
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr

def quality_test(data, snr = 3):
    # Check the quality of the map based on how many aero is in the curbe
    # return 1 if there are zero
    # return 0 if the curve is fine

    maps = Maps(data, bintype='SPX', template='MILESHC-MASTARSSP')

    oiii_ew = maps.emline_gew_oiii_5008

    ew_value = oiii_ew.value
    ew_ivar = oiii_ew.ivar
    ew_snr = oiii_ew.snr
    ew_flag = oiii_ew.pixmask.bits


    ew_row = ew_value.shape[0]
    ew_col = ew_value.shape[1]

    total_spaxel = 0
    data_spaxel = 0

    # Mask the EW  map
    for i in range(ew_row):
        for j in range(ew_col):
            if ew_ivar[i][j] != 0 and len(ew_flag[i][j]) == 0:
                total_spaxel = total_spaxel + 1
                if ew_snr[i][j] >= snr:
                    data_spaxel = data_spaxel + 1
                else:
                    pass
            else:
                pass



    ratio = data_spaxel / total_spaxel
    
    return data, ratio

def ew_integ(data, snr = 3, dphi = 5, cycle = 2, smooth = 2, test = False):
    # check oiii ivar and mask
    # ivar > 1
    # Mask != 1
    # OIII flux / flux error > 3

    maps = Maps(data, bintype='SPX', template='MILESHC-MASTARSSP')

    st_vel = maps.stellar_vel.value
    oiii_ew = maps.emline_gew_oiii_5008

    ew_value = oiii_ew.value
    ew_ivar = oiii_ew.ivar
    ew_snr = oiii_ew.snr
    ew_flag = oiii_ew.pixmask.bits

    ew_row = ew_value.shape[0]
    ew_col = ew_value.shape[1]

    phi = maps.spx_ellcoo_elliptical_azimuth.value
    #r_re = maps.spx_ellcoo_r_re.value


    # Mask the EW  map
    for i in range(ew_row):
        for j in range(ew_col):
            # exclude IVAR = 0: 
            # https://www.sdss.org/dr17/manga/manga-tutorials/manga-faq/#WhydoyououtputIVAR(inversevariance)insteadoferrors?
            if ew_ivar[i][j] == 0:
                ew_value[i][j] = np.nan
            # exclude S/N < setting
            elif ew_snr[i][j] <= snr:
                ew_value[i][j] = np.nan
            # exclude any flag for spaxel error
            elif len(ew_flag[i][j]) != 0:
                ew_value[i][j] = np.nan
            else:
                pass

            
    # Intergrate along r direction:
    curve = []
    for k in np.arange(0,360,dphi):
        bins = []
        for i in range(ew_row):
            for j in range(ew_col):
                if phi[i][j] >= k and \
                    phi[i][j] <= k+dphi and \
                    np.isnan(ew_value[i][j]) == False:
                    bins.append(ew_value[i][j])
                else:
                    pass
        if len(bins) == 0:
            curve.append(0)
        else:
            curve.append(sum(bins)/len(bins))



    ew_cycle = list(curve)*cycle


    # Gaussian smooth
    ew_smo = scipy.ndimage.gaussian_filter(ew_cycle, sigma = smooth)

    # Intepolate to N * 360 degree
    x = np.linspace(0,len(ew_smo),len(ew_smo))
    y = ew_smo
    x2 = np.linspace(0,len(ew_smo),360*cycle)
    f_linear = scipy.interpolate.interp1d(x, y, kind='linear')
    intp_EW = f_linear(x2)

    # Plot output
    if test == True:
        fig = plt.figure(figsize=(10, 7))
        rows = 3
        columns = 2
        fig.add_subplot(rows, columns, 1)
        plt.imshow(ew_value, cmap = 'viridis')

        fig.add_subplot(rows, columns, 2)
        plt.imshow(ew_ivar, cmap = 'viridis')

        fig.add_subplot(rows, columns, 3)
        plt.imshow(ew_snr, cmap = 'viridis')

        fig.add_subplot(rows, columns, 4)
        plt.imshow(st_vel, cmap = 'viridis')

        fig.add_subplot(rows, columns, 5)
        plt.plot(ew_cycle)

        fig.add_subplot(rows, columns, 6)
        plt.plot(intp_EW)

        plt.show()
    else:
        pass
    
    return intp_EW

def simple_integ(data, snr = 3, dphi = 5):
    # check oiii ivar and mask
    # ivar > 1
    # Mask != 1
    # OIII flux / flux error > 3

    maps = Maps(data, bintype='SPX', template='MILESHC-MASTARSSP')

    oiii_ew = maps.emline_gew_oiii_5008

    ew_value = oiii_ew.value
    ew_ivar = oiii_ew.ivar
    ew_snr = oiii_ew.snr
    ew_flag = oiii_ew.pixmask.bits

    ew_row = ew_value.shape[0]
    ew_col = ew_value.shape[1]

    phi = maps.spx_ellcoo_elliptical_azimuth.value
    #r_re = maps.spx_ellcoo_r_re.value


    # Mask the EW  map
    for i in range(ew_row):
        for j in range(ew_col):
            # exclude IVAR = 0: 
            # https://www.sdss.org/dr17/manga/manga-tutorials/manga-faq/#WhydoyououtputIVAR(inversevariance)insteadoferrors?
            if ew_ivar[i][j] == 0:
                ew_value[i][j] = np.nan
            # exclude S/N < setting
            elif ew_snr[i][j] <= snr:
                ew_value[i][j] = np.nan
            # exclude any flag for spaxel error
            elif len(ew_flag[i][j]) != 0:
                ew_value[i][j] = np.nan
            else:
                pass

            
    # Intergrate along r direction:
    curve = []
    for k in np.arange(0,360,dphi):
        bins = []
        for i in range(ew_row):
            for j in range(ew_col):
                if phi[i][j] >= k and \
                    phi[i][j] <= k+dphi and \
                    np.isnan(ew_value[i][j]) == False:
                    bins.append(ew_value[i][j])
                else:
                    pass
        if len(bins) == 0:
            curve.append(0)
        else:
            curve.append(sum(bins)/len(bins))

    return curve

def ring_integ(data, snr = 3, dphi = 10, ring_N = 5, test = False):

    maps = Maps(data, bintype='SPX', template='MILESHC-MASTARSSP')
    oiii_ew = maps.emline_gew_oiii_5008
    ew_value = oiii_ew.value
    ew_ivar = oiii_ew.ivar
    ew_snr = oiii_ew.snr
    ew_flag = oiii_ew.pixmask.bits
    ew_row = ew_value.shape[0]
    ew_col = ew_value.shape[1]
    phi = maps.spx_ellcoo_elliptical_azimuth.value
    r_re = maps.spx_ellcoo_r_re.value
    # Mask the EW  map

    for i in range(ew_row):
        for j in range(ew_col):
            # exclude IVAR = 0: 
            # https://www.sdss.org/dr17/manga/manga-tutorials/manga-faq/#WhydoyououtputIVAR(inversevariance)insteadoferrors?
            if ew_ivar[i][j] == 0:
                ew_value[i][j] = np.nan
            # exclude S/N < setting
            elif ew_snr[i][j] <= snr:
                ew_value[i][j] = np.nan
            # exclude any flag for spaxel error
            elif len(ew_flag[i][j]) != 0:
                ew_value[i][j] = np.nan
            else:
                pass

    r_re_list = []
    for i in range(ew_row):
        for j in range(ew_col):
            if np.isnan(ew_value[i][j]) == False:
                r_re_list.append(r_re[i][j])
            else:
                pass

    r_re_max = round(max(r_re_list))
    dr = (r_re_max - 0) / ring_N

    curve_list = []

    for r in np.linspace(0, r_re_max-dr, ring_N):
        curve = []
        for k in np.arange(0,360,dphi):
            bins = []
            for i in range(ew_row):
                for j in range(ew_col):
                    if phi[i][j] >= k and \
                        phi[i][j] <= k+dphi and \
                        r_re[i][j] >= r and \
                        r_re[i][j] <= r+dr and \
                        np.isnan(ew_value[i][j]) == False:
                        bins.append(ew_value[i][j])
                    else:
                        pass
            if len(bins) == 0:
                curve.append(0)
            else:
                curve.append(sum(bins)/len(bins))
        curve_list.append(curve)

    if test == True:
        for i in np.linspace(0,ring_N-1, ring_N):
            plt.plot(curve_list[int(i)]+2*i)
        plt.show()
    else:
        pass


    
    return data, curve_list, r_re_max

def fourier_classifier(EW_curve, n_peak = 5):
    
    # 1. Set curve osillate around  y=0
    # 2. Take the FT result  from 1~50 because FT saturate at 0. 
    # Fourier Transform:
    curve = copy.deepcopy(EW_curve)
    
    # Max on the curve
    max_value = curve.max()
    max_index = list(curve).index(max_value)
    
    yf = np.abs(fft(curve)[1:30])

    norm_yf = normalize(yf)
    output_y = copy.deepcopy(norm_yf)

    # Strongest frequency
    peak_value = max(norm_yf)
    peak_index = list(norm_yf).index(peak_value)
    
    four_intensity = norm_yf[3]
    ten_intensity = sum([norm_yf[1], norm_yf[5], norm_yf[7], norm_yf[9]])

    # Finding the 90 degree angle residue: 
    '''
    # Old Method
    residue = (max_index)-90 # to 360*2 angle, diff to zero, positive
    loop_residue = min([residue%180, abs(residue-180)])
    abs_residue = min([loop_residue%180, abs(loop_residue-180)])
    '''
    # New Method
    local_residue_l = []
    for i in np.arange(0,720,180):
        section = curve[i:i+180]
        local_max_value = section.max()
        local_max_index = list(section).index(local_max_value)
        local_max_residue = local_max_index-90
        local_residue_l.append(local_max_residue)
    #abs_residue = np.mean(local_residue_l)

    n_index = n_peak * -1

    norm_yf.sort()
    #loss = sum(np.diff(norm_yf[n_index:-1]))
    # define new loss
    loss = sum(norm_yf[n_index:-1])
 
    
    return output_y, (max_index, peak_index) , loss, (four_intensity, ten_intensity), local_residue_l

def deriv_classifier(EW_curve):
# 1. Take the derivative of the curve.
# 2. Compare the avg of 0-90, 90-180, 180-270, 270-360
# 3. Find relation eg: equality, similarity. 
    fir_deriv = []

    for i in np.linspace(0,70,71):
        j = int(i)
        diff = EW_curve[j+1] - EW_curve[j]
        fir_deriv.append(diff)

    a1 = sum(fir_deriv[0:17])
    a2 = sum(fir_deriv[18:35])
    a3 = sum(fir_deriv[36:53])
    a4 = sum(fir_deriv[54:71])

    return fir_deriv, [a1, a2, a3, a4]
        
def plot_compared(data, snr = 3, save = True):
    # Plot Image, OIII, Star_v, Gas_v, Curve, FFT. 
    # in a 3X2 subplots
    dphi = 5
    cycle = 1
    smooth = 0

    fig = plt.figure(figsize=(10, 7))
    
    
    maps = Maps(data, bintype='SPX', template='MILESHC-MASTARSSP')

    st_vel = maps.stellar_vel
    oiii_ew = maps.emline_gew_oiii_5008
    ew_copy = copy.deepcopy(oiii_ew)

    ew_value = oiii_ew.value
    ew_ivar = oiii_ew.ivar
    ew_snr = oiii_ew.snr
    ew_flag = oiii_ew.pixmask.bits

    ew_row = ew_value.shape[0]
    ew_col = ew_value.shape[1]

    phi = maps.spx_ellcoo_elliptical_azimuth.value
    #r_re = maps.spx_ellcoo_r_re.value

    #mean = np.mean(ew_value)
    #sd = np.std(ew_value)

    x_tik = oiii_ew.shape[0]/4
    y_tik = oiii_ew.shape[1]/4

    # Mask the EW  map
    for i in range(ew_row):
        for j in range(ew_col):
            # exclude IVAR = 0: 
            # https://www.sdss.org/dr17/manga/manga-tutorials/manga-faq/#WhydoyououtputIVAR(inversevariance)insteadoferrors?
            if ew_ivar[i][j] == 0:
                ew_value[i][j] = np.nan
            # exclude S/N < setting
            elif ew_snr[i][j] <= snr:
                ew_value[i][j] = np.nan
            # exclude any flag for spaxel error
            elif len(ew_flag[i][j]) != 0:
                ew_value[i][j] = np.nan
            else:
                pass

      # Intergrate along r direction:
    curve = []
    for k in np.arange(0,360,dphi):
        bins = []
        for i in range(ew_row):
            for j in range(ew_col):
                if phi[i][j] >= k and \
                    phi[i][j] <= k+dphi and \
                    np.isnan(ew_value[i][j]) == False:
                    bins.append(ew_value[i][j])
                else:
                    pass
        if len(bins) == 0:
            curve.append(0)
        else:
            curve.append(sum(bins)/len(bins))

    # Plot N cycle
    ew_cycle = list(curve)*cycle

    # Gaussian smooth
    ew_smo = scipy.ndimage.gaussian_filter(ew_cycle, sigma = smooth)

    # Intepolate to N * 360 degree
    x = np.linspace(0,len(ew_smo),len(ew_smo))
    y = ew_smo
    x2 = np.linspace(0,len(ew_smo),360*cycle)
    f_linear = scipy.interpolate.interp1d(x, y, kind='linear')
    intp_EW = f_linear(x2)



    # mask the st_vel and ew_full
    for maps in [st_vel, ew_copy]:
        row = maps.value.shape[0]
        col = maps.value.shape[1]
        flag = maps.pixmask.bits
        for i in range(row):
            for j in range(col):
                if len(flag[i][j]) > 0:
                    maps.value[i][j] = np.nan
                else:
                    pass


    st_vel = np.flipud(st_vel.value)
    ew_full = np.flipud(ew_copy.value)
    oiii_ew = np.flipud(ew_value)

    axis_indicat = np.zeros(phi.shape)

    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            if phi[i][j] >= 75 and phi[i][j] <= 105:
                axis_indicat[i][j] = 10
            elif phi[i][j] >= 255 and phi[i][j] <= 285:
                axis_indicat[i][j] = 10
            #elif r_re[i][j] <= 0.3:
                #axis_indicat[i][j] = 10
            else:
                axis_indicat[i][j] = np.nan

    axis_indicat = np.flipud(axis_indicat)
                
    rows = 2
    columns = 3
    
    # Plot the 1st image
    fig.add_subplot(rows, columns, 1)
    fig.set_figheight(10)
    fig.set_figwidth(10)
    plt.title("Image")
    im = Image(data)
    spaxel_len = 281*2
    factor = spaxel_len / axis_indicat.shape[0]
    large_mask = ndimage.zoom(axis_indicat, factor, order=0)
    # large_mask[large_mask > 0] = 1
    mask = large_mask
    im = showImage(plateifu = data)
    masked = np.ma.masked_where(mask == 0, mask)
    plt.xlabel('Spaxel (arcsec)', fontproperties=font)
    plt.ylabel('Spaxel (arcsec)', fontproperties=font)
    plt.imshow(im, interpolation='none',extent=[-25,25,-25,25])
    plt.imshow(masked, cmap = 'autumn', interpolation='none', \
        alpha=0.2,extent=[-25,25,-25,25])
    plt.legend(('mask: 75~105 and 255~285 degrees'),loc='upper right')

    # Plot the 2nd OIII EW map
    fig.add_subplot(rows, columns, 4)
    fig.set_figheight(10)
    fig.set_figwidth(10)
    plt.title("Masked EW map", fontproperties=font)
    plt.xlabel('Spaxel (arcsec)', fontproperties=font)
    plt.ylabel('Spaxel (arcsec)', fontproperties=font)
    plt.imshow(oiii_ew, cmap = 'viridis',extent=[-1*x_tik, x_tik, -1*y_tik, y_tik])
    plt.imshow(axis_indicat, cmap = 'autumn', interpolation='none', \
        alpha=0.2, extent=[-1*x_tik, x_tik, -1*y_tik, y_tik])

    
    # Plot the 3rd Star v
    fig.add_subplot(rows, columns, 2)
    fig.set_figheight(10)
    fig.set_figwidth(10)
    plt.title("Stellar Velocity", fontproperties=font)
    plt.xlabel('Spaxel (arcsec)', fontproperties=font)
    plt.ylabel('Spaxel (arcsec)', fontproperties=font)
    plt.imshow(st_vel, cmap = 'viridis',extent=[-1*x_tik, x_tik, -1*y_tik, y_tik])
    plt.imshow(axis_indicat, cmap = 'autumn', interpolation='none', \
        alpha=0.2, extent=[-1*x_tik, x_tik, -1*y_tik, y_tik])

    # Plot the 4th FULL EW
    fig.add_subplot(rows, columns, 5)
    fig.set_figheight(10)
    fig.set_figwidth(10)
    plt.title("Complete EW map", fontproperties=font)
    plt.xlabel('Spaxel (arcsec)', fontproperties=font)
    plt.ylabel('Spaxel (arcsec)', fontproperties=font)
    plt.imshow(ew_full, cmap = 'viridis',extent=[-1*x_tik, x_tik, -1*y_tik, y_tik])
    plt.imshow(axis_indicat, cmap = 'autumn', interpolation='none', \
        alpha=0.2, extent=[-1*x_tik, x_tik, -1*y_tik, y_tik])

    # Plot 5rd EW dR
    fig.add_subplot(rows, columns, 3)
    fig.set_figheight(20)
    fig.set_figwidth(20)
    plt.title("OIII EW dR", fontproperties=font)
    plt.xlabel('Angular Direction [Degrees]', fontproperties=font)
    plt.ylabel('EW/spaxel [Angstrom]', fontproperties=font)
    plt.plot(intp_EW)
    bound = [75, 105, 255, 285]#, 435, 465, 615, 645]
    for xc in bound:
        plt.axvline(x = xc, linestyle = '--', color = 'r')


        
    # Plot 6th FTT
    fig.add_subplot(rows, columns, 6)
    fig.set_figheight(10)
    fig.set_figwidth(20)
    plt.title("Fourier Frequency", fontproperties=font)
    plt.xlabel('Frequency (Hz)', fontproperties=font)
    #result = fourier_classifier(intp_EW)[0]
    #x = np.linspace(0.5, 15, 29)
    #plt.plot(x, result)
    plt.plot(fft(intp_EW)[0:30])



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

def integr_visual(data, snr = 3, save = True):

    fig = plt.figure(figsize=(7, 7))
    
    dphi = 5
    
    maps = Maps(data, bintype='SPX', template='MILESHC-MASTARSSP')


    oiii_ew = maps.emline_gew_oiii_5008
    ew_value = oiii_ew.value
    ew_ivar = oiii_ew.ivar
    ew_snr = oiii_ew.snr
    ew_flag = oiii_ew.pixmask.bits

    ew_row = ew_value.shape[0]
    ew_col = ew_value.shape[1]

    phi = maps.spx_ellcoo_elliptical_azimuth.value


    x_tik = oiii_ew.shape[0]/4
    y_tik = oiii_ew.shape[1]/4
    mask_value = ew_value.min() - 1

    # Mask the EW  map
    for i in range(ew_row):
        for j in range(ew_col):
            # exclude IVAR = 0: 
            # https://www.sdss.org/dr17/manga/manga-tutorials/manga-faq/#WhydoyououtputIVAR(inversevariance)insteadoferrors?
            if ew_ivar[i][j] == 0:
                ew_value[i][j] = np.nan
            # exclude S/N < setting
            elif ew_snr[i][j] <= snr:
                ew_value[i][j] = np.nan
            # exclude any flag for spaxel error
            elif len(ew_flag[i][j]) != 0:
                ew_value[i][j] = np.nan
            else:
                pass


    angle_integral = copy.deepcopy(ew_value)

    # Make the integral angle indicator
    for i in range(ew_row):
        for j in range(ew_col):
            if np.isnan(angle_integral[i][j]) == False:
                if phi[i][j]%20<=5:
                    angle_integral[i][j] = mask_value + 5
                elif phi[i][j]%20<=10:
                    angle_integral[i][j] = mask_value + 10
                elif phi[i][j]%20<=15:
                    angle_integral[i][j] = mask_value + 15
                else:
                    angle_integral[i][j] = mask_value + 20
            else:
                pass

    curve = []
    for k in np.arange(0,360,dphi):
        bins = []
        for i in range(ew_row):
            for j in range(ew_col):
                if phi[i][j] >= k and \
                    phi[i][j] <= k+dphi and \
                    np.isnan(ew_value[i][j]) == False:
                    bins.append(ew_value[i][j])
                else:
                    pass
        if len(bins) == 0:
            curve.append(0)
        else:
            curve.append(sum(bins)/len(bins))

    bin_indica = copy.deepcopy(ew_value)

    for k in np.arange(0,360,dphi):
        bins = []
        index = int(k/5)
        for i in range(ew_row):
            for j in range(ew_col):
                if phi[i][j] >= k and \
                    phi[i][j] <= k+dphi and \
                    np.isnan(bin_indica[i][j]) == False:
                    bin_indica[i][j] = curve[index]
                else:
                    pass

    x = np.linspace(0,len(curve),len(curve))
    y = curve
    x2 = np.linspace(0,len(curve),360)
    f_linear = scipy.interpolate.interp1d(x, y, kind='linear')
    intp_EW = f_linear(x2)


    rows = 2
    columns = 2
    
        
    # Plot angle indicator
    fig.add_subplot(rows, columns, 1)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    plt.title("5 Degrees Slices", fontproperties=font)
    plt.xlabel('Spexel [arcsec]', fontproperties=font)
    plt.ylabel('Spexel [arcsec]', fontproperties=font)
    plt.imshow(np.flipud(angle_integral), cmap = 'plasma', extent=[-1*x_tik, x_tik, -1*y_tik, y_tik])
    plt.grid(color='grey', linestyle='-', linewidth=0.2)

    # Plot bin value visualization
    fig.add_subplot(rows, columns, 3)
    fig.set_figwidth(15)
    plt.title("Slices Value", fontproperties=font)
    plt.xlabel('Spexel [arcsec]', fontproperties=font)
    plt.ylabel('Spexel [arcsec]', fontproperties=font)
    plt.imshow(np.flipud(bin_indica), cmap = 'viridis', extent=[-1*x_tik, x_tik, -1*y_tik, y_tik])
    plt.grid(color='grey', linestyle='-', linewidth=0.2)
    plt.colorbar(label = r'$\AA$')
  

    # Plot 3rd EW dR
    fig.add_subplot(rows, columns, 2)
    fig.set_figheight(15)
    fig.set_figwidth(20)
    plt.title("OIII EW along Angular Direction", fontproperties=font)
    plt.xlabel('Angular Direction [degrees]', fontproperties=font)
    plt.ylabel('EW/spaxel [Angstrom]', fontproperties=font)
    plt.plot(intp_EW, label = 'EW curve')
    plt.axvline(x = 90, linestyle = '--', color = 'r',label = r'90$\degree$')
    plt.axvline(x = 270, linestyle = '--', color = 'r',label = r'270$\degree$')
    plt.grid(color='grey', linestyle='-', linewidth=0.2)
    plt.legend(loc='upper right',prop=font)


        
    # Plot 4th FTT
    fig.add_subplot(rows, columns, 4)
    fig.set_figheight(15)
    fig.set_figwidth(20)
    plt.title("Fourier Frequency", fontproperties=font)
    plt.xlabel('Frequency [Hz]', fontproperties=font)
    #result = fourier_classifier(intp_EW)[0]
    #x = np.linspace(0.5, 15, 29)
    #plt.plot(x, result)
    plt.plot(abs(fft(intp_EW))[0:30], label = 'Absolute Value')
    plt.plot(np.real(fft(intp_EW))[0:30], label = 'Real Part', alpha =  0.5)
    plt.plot(np.imag(fft(intp_EW))[0:30], label = 'Imaginary Part', alpha = 0.5)
    bound = [2]
    for xc in bound:
        plt.axvline(x = xc, linestyle = '--', color = 'r', label='2 Hz')
    plt.grid(color='grey', linestyle='-', linewidth=0.2)
    plt.legend(loc='upper right',prop=font)



    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.8, 
                    top=0.6, 
                    wspace=0.2, 
                    hspace=0.2)
    

    if save == True:
        plt.savefig(data+'_interg.png')
        plt.cla()
    else:
        plt.show()

def mask_visual(data, snr = 3, save = True):

    fig = plt.figure(figsize=(7, 7))
    
    
    maps = Maps(data, bintype='SPX', template='MILESHC-MASTARSSP')


    oiii_ew = maps.emline_gew_oiii_5008


    ew_value = oiii_ew.value
    ew_ivar = oiii_ew.ivar
    ew_snr = oiii_ew.snr
    ew_flag = oiii_ew.pixmask.bits

    ew_row = ew_value.shape[0]
    ew_col = ew_value.shape[1]

    #phi = maps.spx_ellcoo_elliptical_azimuth.value

    x_tik = oiii_ew.shape[0]/4
    y_tik = oiii_ew.shape[1]/4

    mean = np.mean(ew_value)
    sd = np.std(ew_value)
    mask_value = ew_value.min() - 1


    ivar_map = copy.deepcopy(ew_value)   
    for i in range(ew_row):
        for j in range(ew_col):
            if ew_ivar[i][j] == 0:
                ivar_map[i][j] = mask_value
            else:
                pass
    
    snr_map = copy.deepcopy(ew_value)
    for i in range(ew_row):
        for j in range(ew_col):
            if ew_snr[i][j] <= snr:
                snr_map[i][j] = mask_value
            else:
                pass

    flag_map = copy.deepcopy(ew_value)
    for i in range(ew_row):
        for j in range(ew_col):
            if len(ew_flag[i][j]) != 0:
                flag_map[i][j] = mask_value
            else:
                pass

    comp_map = copy.deepcopy(ew_value)

    for i in range(ew_row):
        for j in range(ew_col):
            if ew_ivar[i][j] == 0:
                comp_map[i][j] = mask_value
            # exclude S/N < setting
            elif ew_snr[i][j] <= snr:
                comp_map[i][j] = mask_value
            # exclude any flag for spaxel error
            elif len(ew_flag[i][j]) != 0:
                comp_map[i][j] = mask_value
            else:
                pass


    rows = 2
    columns = 3
    
    fig.add_subplot(rows, columns, 1)
    fig.set_figheight(10)
    fig.set_figwidth(10)
    plt.title(str(data)+'Image')
    im = Image(data)
    # large_mask[large_mask > 0] = 1
    im = showImage(plateifu = data)
    plt.xlabel('Spaxel (arcsec)', fontproperties=font)
    plt.ylabel('Spaxel (arcsec)', fontproperties=font)
    plt.imshow(im, interpolation='none',extent=[-25,25,-25,25])


    # Plot EW full
    fig.add_subplot(rows, columns, 4)
    fig.set_figheight(15)
    fig.set_figwidth(15)  
    plt.title("Original EW map", fontproperties=font)
    plt.xlabel('Spexel [arcsec]', fontproperties=font)
    plt.ylabel('Spexel [arcsec]', fontproperties=font)
    plt.imshow(np.flipud(ew_value), cmap = 'viridis',extent=[-1*x_tik, x_tik, -1*y_tik, y_tik])
    plt.grid(color='grey', linestyle='-', linewidth=0.2)
    plt.colorbar(label = r'$\AA$')



    # Plot ivar
    fig.add_subplot(rows, columns, 2)
    fig.set_figheight(15)
    fig.set_figwidth(15)     
    plt.title("ivar=0 mask", fontproperties=font)
    plt.xlabel('Spexel [arcsec]', fontproperties=font)
    plt.ylabel('Spexel [arcsec]', fontproperties=font)
    plt.imshow(np.flipud(ivar_map), cmap = viridis_vis, extent=[-1*x_tik, x_tik, -1*y_tik, y_tik])
    plt.grid(color='grey', linestyle='-', linewidth=0.2)
    plt.colorbar(label = r'$\AA$')



    # Plot snr
    fig.add_subplot(rows, columns, 5)
    fig.set_figheight(15)
    fig.set_figwidth(15)        
    plt.title("SNR<3 mask", fontproperties=font)
    plt.xlabel('Spexel [arcsec]', fontproperties=font)
    plt.ylabel('Spexel [arcsec]', fontproperties=font)
    plt.imshow(np.flipud(snr_map), cmap = viridis_vis, extent=[-1*x_tik, x_tik, -1*y_tik, y_tik])
    plt.grid(color='grey', linestyle='-', linewidth=0.2)
    plt.colorbar(label = r'$\AA$')

    
    # Plot flag
    fig.add_subplot(rows, columns, 3)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    plt.title("Flag mask", fontproperties=font)
    plt.xlabel('Spexel [arcsec]', fontproperties=font)
    plt.ylabel('Spexel [arcsec]', fontproperties=font)
    plt.imshow(np.flipud(flag_map), cmap = viridis_vis, extent=[-1*x_tik, x_tik, -1*y_tik, y_tik])
    plt.grid(color='grey', linestyle='-', linewidth=0.2)
    plt.colorbar(label = r'$\AA$')
 
    
    # Plot COMP
    fig.add_subplot(rows, columns, 6)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    plt.title("3 masks combined", fontproperties=font)
    plt.xlabel('Spexel [arcsec]', fontproperties=font)
    plt.ylabel('Spexel [arcsec]', fontproperties=font)
    plt.imshow(np.flipud(comp_map), cmap = viridis_vis, extent=[-1*x_tik, x_tik, -1*y_tik, y_tik])
    plt.grid(color='grey', linestyle='-', linewidth=0.2)
    plt.colorbar(label = r'$\AA$')



    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=1.0, 
                    top=0.6, 
                    wspace=0.2, 
                    hspace=0.2)
    

    if save == True:
        plt.savefig(data+'_mask_vis.png')
        plt.cla()
    else:
        plt.show()

def mask_integ_visual(data, snr = 3, save = True):
    
    dphi = 5

    fig = plt.figure(figsize=(7, 7))
    
    
    maps = Maps(data, bintype='SPX', template='MILESHC-MASTARSSP')


    oiii_ew = maps.emline_gew_oiii_5008


    ew_value = oiii_ew.value
    ew_ivar = oiii_ew.ivar
    ew_snr = oiii_ew.snr
    ew_flag = oiii_ew.pixmask.bits

    ew_row = ew_value.shape[0]
    ew_col = ew_value.shape[1]

    phi = maps.spx_ellcoo_elliptical_azimuth.value

    x_tik = oiii_ew.shape[0]/4
    y_tik = oiii_ew.shape[1]/4

    #mean = np.mean(ew_value)
    #sd = np.std(ew_value)
    mask_value = np.nan


    ivar_map = copy.deepcopy(ew_value)   
    for i in range(ew_row):
        for j in range(ew_col):
            if ew_ivar[i][j] == 0:
                ivar_map[i][j] = mask_value
            else:
                pass
    
    snr_map = copy.deepcopy(ew_value)
    for i in range(ew_row):
        for j in range(ew_col):
            if ew_snr[i][j] <= snr:
                snr_map[i][j] = mask_value
            else:
                pass

    flag_map = copy.deepcopy(ew_value)
    for i in range(ew_row):
        for j in range(ew_col):
            if len(ew_flag[i][j]) != 0:
                flag_map[i][j] = mask_value
            else:
                pass

    comp_map = copy.deepcopy(ew_value)

    for i in range(ew_row):
        for j in range(ew_col):
            if ew_ivar[i][j] == 0:
                comp_map[i][j] = mask_value
            # exclude S/N < setting
            elif ew_snr[i][j] <= snr:
                comp_map[i][j] = mask_value
            # exclude any flag for spaxel error
            elif len(ew_flag[i][j]) != 0:
                comp_map[i][j] = mask_value
            else:
                pass

    angle_integral = copy.deepcopy(comp_map)

    # Make the integral angle indicator
    for i in range(ew_row):
        for j in range(ew_col):
            if np.isnan(angle_integral[i][j]) == False:
                if phi[i][j]%20<=5:
                    angle_integral[i][j] = 5
                elif phi[i][j]%20<=10:
                    angle_integral[i][j] = 10
                elif phi[i][j]%20<=15:
                    angle_integral[i][j] = 15
                else:
                    angle_integral[i][j] = 20
            else:
                pass

    curve = []
    for k in np.arange(0,360,dphi):
        bins = []
        for i in range(ew_row):
            for j in range(ew_col):
                if phi[i][j] >= k and \
                    phi[i][j] <= k+dphi and \
                    np.isnan(comp_map[i][j]) == False:
                    bins.append(comp_map[i][j])
                else:
                    pass
        if len(bins) == 0:
            curve.append(0)
        else:
            curve.append(sum(bins)/len(bins))

    x = np.linspace(0,len(curve),len(curve))
    y = curve
    x2 = np.linspace(0,len(curve),360)
    f_linear = scipy.interpolate.interp1d(x, y, kind='linear')
    intp_EW = f_linear(x2)


    rows = 3
    columns = 3
    
    fig.add_subplot(rows, columns, 1)
    fig.set_figheight(10)
    fig.set_figwidth(10)
    plt.title(str(data)+'Image')
    im = Image(data)
    # large_mask[large_mask > 0] = 1
    im = showImage(plateifu = data)
    plt.xlabel('Spaxel (arcsec)', fontproperties=font)
    plt.ylabel('Spaxel (arcsec)', fontproperties=font)
    plt.imshow(im, interpolation='none',extent=[-25,25,-25,25])


    # Plot EW full
    fig.add_subplot(rows, columns, 4)
    fig.set_figheight(15)
    fig.set_figwidth(15)  
    plt.title("Original EW map", fontproperties=font)
    plt.xlabel('Spexel [arcsec]', fontproperties=font)
    plt.ylabel('Spexel [arcsec]', fontproperties=font)
    plt.imshow(np.flipud(ew_value), cmap = 'viridis',extent=[-1*x_tik, x_tik, -1*y_tik, y_tik])
    plt.grid(color='grey', linestyle='-', linewidth=0.2)
    plt.colorbar(label = r'$\AA$')



    # Plot ivar
    fig.add_subplot(rows, columns, 2)
    fig.set_figheight(15)
    fig.set_figwidth(15)     
    plt.title("ivar=0 mask", fontproperties=font)
    plt.xlabel('Spexel [arcsec]', fontproperties=font)
    plt.ylabel('Spexel [arcsec]', fontproperties=font)
    plt.imshow(np.flipud(ivar_map), cmap = 'viridis', extent=[-1*x_tik, x_tik, -1*y_tik, y_tik])
    plt.grid(color='grey', linestyle='-', linewidth=0.2)
    plt.colorbar(label = r'$\AA$')



    # Plot snr
    fig.add_subplot(rows, columns, 5)
    fig.set_figheight(15)
    fig.set_figwidth(15)        
    plt.title("SNR<3 mask", fontproperties=font)
    plt.xlabel('Spexel [arcsec]', fontproperties=font)
    plt.ylabel('Spexel [arcsec]', fontproperties=font)
    plt.imshow(np.flipud(snr_map), cmap = 'viridis', extent=[-1*x_tik, x_tik, -1*y_tik, y_tik])
    plt.grid(color='grey', linestyle='-', linewidth=0.2)
    plt.colorbar(label = r'$\AA$')

    
    # Plot flag
    fig.add_subplot(rows, columns, 3)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    plt.title("Flag mask", fontproperties=font)
    plt.xlabel('Spexel [arcsec]', fontproperties=font)
    plt.ylabel('Spexel [arcsec]', fontproperties=font)
    plt.imshow(np.flipud(flag_map), cmap = 'viridis', extent=[-1*x_tik, x_tik, -1*y_tik, y_tik])
    plt.grid(color='grey', linestyle='-', linewidth=0.2)
    plt.colorbar(label = r'$\AA$')
 
    
    # Plot COMP
    fig.add_subplot(rows, columns, 6)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    plt.title("3 masks combined", fontproperties=font)
    plt.xlabel('Spexel [arcsec]', fontproperties=font)
    plt.ylabel('Spexel [arcsec]', fontproperties=font)
    plt.imshow(np.flipud(comp_map), cmap = 'viridis', extent=[-1*x_tik, x_tik, -1*y_tik, y_tik])
    plt.grid(color='grey', linestyle='-', linewidth=0.2)
    plt.colorbar(label = r'$\AA$')

    # Plot angle indicator
    fig.add_subplot(rows, columns, 7)
    fig.set_figheight(15)
    fig.set_figwidth(15)
    plt.title("5 Degrees Slices", fontproperties=font)
    plt.xlabel('Spexel [arcsec]', fontproperties=font)
    plt.ylabel('Spexel [arcsec]', fontproperties=font)
    plt.imshow(np.flipud(angle_integral), cmap = 'plasma', extent=[-1*x_tik, x_tik, -1*y_tik, y_tik])
    plt.grid(color='grey', linestyle='-', linewidth=0.2)

    fig.add_subplot(rows, columns, 8)
    fig.set_figheight(15)
    fig.set_figwidth(30)
    plt.title("OIII EW along Angular Direction", fontproperties=font)
    plt.xlabel('Angular Direction [degrees]', fontproperties=font)
    plt.ylabel('EW/spaxel [Angstrom]', fontproperties=font)
    plt.plot(intp_EW, label = 'EW curve')
    plt.axvline(x = 90, linestyle = '--', color = 'r',label = r'90$\degree$')
    plt.axvline(x = 270, linestyle = '--', color = 'r',label = r'270$\degree$')
    plt.grid(color='grey', linestyle='-', linewidth=0.2)
    plt.legend(loc='upper right',prop=font)


    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=1.0, 
                    top=1.2, 
                    wspace=0.2, 
                    hspace=0.2)
    

    if save == True:
        plt.savefig(data+'_mask_integ_vis.png', bbox_inches='tight')
        plt.cla()
        return data
    else:
        plt.show()

def triangle_model_loss(data):
    
    curve = Simpler_Classifier.simple_integ(data)

    def cone_model(h, left, right, cont):

        if left+right >= 36 or left+right <= 2 or h <= 1:
            model = np.linspace(0.1,0.1,36)

        else:

            wid = left + right
            radius = h
            w_res = math.floor((36-wid)/2)
            r_var = np.linspace(0,radius,w_res)

            y1 = []

            circ_area = np.pi * (radius**2)
            for i in r_var:

                l = radius - i
                angle_rad = np.arccos(l/radius)
                w = np.sin(angle_rad)*radius
                triangles = l*w

                angle_res = (np.pi*2) - (angle_rad*2)
                disc_area = (angle_res/(2*np.pi)) * circ_area

                total_area = circ_area - (disc_area + triangles)
                y1.append(total_area)


            y2 = copy.deepcopy(y1)
            y1.reverse()
            y = y2 + y1[1:]

            model = list(np.array(list(np.zeros(left)) + y + list(np.zeros(right))) + cont)

        x = np.linspace(0,len(model),len(model))
        y = model
        x2 = np.linspace(0,len(model),36)
        f_linear = scipy.interpolate.interp1d(x, y, kind='linear')
        intp_model = f_linear(x2)

        return intp_model

    def loss_function_a(arg):

        h, left, right, cont= arg
        onecone = cone_model(h,left, right, cont)
        loss = 0
        for i in range(36):
            loss = loss + abs(curve[i]-onecone[i])

        total_loss = loss.item()

        return total_loss

    def loss_function_b(arg):

        h, left, right, cont= arg
        onecone = cone_model(h,left, right, cont)
        loss = 0
        for i in range(36):
            loss = loss + abs(curve[i+36]-onecone[i])

        total_loss = loss.item()

        return total_loss

    space = [hp.uniform('h',0, 100), hp.randint('left',35), hp.randint('right',35),hp.uniform('cont',0, 3)]
    best_a = fmin(loss_function_a, space, algo=tpe.suggest, max_evals = 2000)
    best_b = fmin(loss_function_b, space, algo=tpe.suggest, max_evals = 2000)
    total_loss = loss_function_a(space_eval(space, best_a))+loss_function_b(space_eval(space, best_b))
    return total_loss


def norm_model_loss(data):
    
    curve = simple_integ(data)
    
    def cone_norm_model(mu, variance, scale, contin):
        x = np.linspace(0, 36, 36)
        sigma = math.sqrt(variance)
        cone = stats.norm.pdf(x, mu, sigma)*scale
        onecone = cone+contin
        return onecone


    def loss_function_c(arg):

        mu, variance, scale, contin= arg
        onecone = cone_norm_model(mu, variance, scale, contin)
        loss = 0
        for i in range(36):
            loss = loss + abs(curve[i]-onecone[i])

        total_loss = loss.item()

        return total_loss

    def loss_function_d(arg):

        mu, variance, scale, contin= arg
        onecone = cone_norm_model(mu, variance, scale, contin)
        loss = 0
        for i in range(36):
            loss = loss + abs(curve[i+36]-onecone[i])

        total_loss = loss.item()

        return total_loss

    space_norm = [hp.uniform('mu',0, 36), hp.uniform('variance',0, 36), hp.uniform('scale',0, 100), hp.uniform('contin',0, 10)]
    best_norm_c = fmin(loss_function_c, space_norm, algo=tpe.suggest, max_evals = 2000)
    best_norm_d = fmin(loss_function_d, space_norm, algo=tpe.suggest, max_evals = 2000)
    total_loss = loss_function_c(space_eval(space_norm, best_norm_c))+loss_function_d(space_eval(space_norm, best_norm_d))
    return total_loss


def shell(data):
    try:
        curve = ew_integ(data)
        result = fourier_classifier(curve)
        peak_index = result[1][1]
        peak_residue = result[2]
        angle_residue= result[4]
        return data, peak_index, peak_residue, angle_residue

    except:
        pass
       
def trial(data):
    try:
        curve = simple_integ(data)
        yf_abs = abs(fft(curve)[1:30])
        peak_value = max(yf_abs)
        peak_index = list(yf_abs).index(peak_value)
        return data, peak_index+1
    except:
        pass

def ring_test(data):
    try:
        
        result = ring_integ(data, ring_N=10)
        curve_list = result[1]
        r_re_max = result[2]

        peak_count = 0
        n = len(curve_list)

        for i in np.linspace(0,n-1,n):

            yf = fft(curve_list[int(i)])
            yf2 = yf[2]

            yf_abs = abs(yf[1:10])
            peak_value = max(yf_abs)
            peak_index = list(yf_abs).index(peak_value)

            if yf2 <= 0 and peak_index == 1:
                peak_count = peak_count + (10**i)
            else:
                pass


        return data, peak_count, r_re_max

    except:
        pass

def deriv_test(data):
    try:
        curve = simple_integ(data)
        result = deriv_classifier(curve)
        return data, result
    except:
        pass


