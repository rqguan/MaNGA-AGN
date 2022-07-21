

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


def ellip_gen(plateifu = '7443-1902'):

    maps = Maps(plateifu, bintype='SPX', template='MILESHC-MASTARSSP')
    oiii_ew = maps.emline_gew_oiii_5008.value

    # Ellipticity Properties
    phi = maps.spx_ellcoo_elliptical_azimuth.value.round(decimals=2)
    r_re = maps.spx_ellcoo_r_re.value
    
    # We select the ring of r = 0.8 er ~ 1.2 er for EW
    # Make a new array of (EW, phi_er)
    ew_r_comb = np.array((oiii_ew, r_re)) 
    ew_phi_comb = np.array((oiii_ew, phi)) 
    #Tanspose the array to make each element in form of (EW, phi_er)

    return ew_r_comb, phi, ew_phi_comb

def ellip_ring_curve(ellip, in_r , out_r , sig = 4, cycle = 2, smooth = 3):

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




def first_filter(plateifu, inner = 0.6, outer = 2, step = 0.2, width = 0.3, test = False):

    ellip = ellip_gen(plateifu)
    loss_list = []

    for i in np.arange(inner, outer, step, width):
        start, end = round(i,1), round(i,1)+width
        curve = ellip_ring_curve(ellip, in_r = start, out_r = end, cycle = 2)
        result = fourier_classifier(curve)
        
        if result[1][1] == 3:
            loss_list.append(result[2])
        else:
            loss_list.append(np.array([0]))
            

    if test == False:
        if sum(loss_list) >= 220:
            return plateifu, sum(loss_list)
        else:
            pass
    else:
        return plateifu, sum(loss_list)

def second_filter(plateifu, inner = 0.6, outer = 2, step = 0.2, width = 0.3, test = False):        
    
    ellip = ellip_gen(plateifu)
    axis_loss = []
    
    for i in np.arange(inner, outer, step):
        
        start, end = round(i,1), round(i,1)+width
        curve = ellip_ring_curve(ellip, in_r = start, out_r = end, cycle = 2)
        result = fourier_classifier(curve)

        # Find the difference between max_index and the nearest phase 
        axis_loss.append(result[4])

    if test == False:
        if sum(axis_loss) <= 225:
            return plateifu, sum(axis_loss)
        else:
            pass
    else:
        return plateifu, sum(axis_loss)
    

def plot_everything(data, inner = 0.6, outer = 2, step = 0.2, save = False):
    
    fig = plt.figure(figsize=(10, 7))

    maps = Maps(data, bintype='SPX', template='MILESHC-MASTARSSP')
    #st_vel = maps.stellar_vel.value
    oiii_ew = maps.emline_gew_oiii_5008.value
    phi = maps.spx_ellcoo_elliptical_azimuth.value
    r_re = maps.spx_ellcoo_r_re.value
    
    mean = np.mean(oiii_ew)
    sd = np.std(oiii_ew)
    
    for i in range(oiii_ew.shape[0]):
        for j in range(oiii_ew.shape[1]):
            if oiii_ew[i][j] >= mean+2*sd or oiii_ew[i][j]<=mean-2*sd:
                oiii_ew[i][j] = np.nan
            elif oiii_ew[i][j] == 0:
                oiii_ew[i][j] = np.nan
            else:
                pass

    axis_indicat = np.zeros(phi.shape)

    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            if phi[i][j] >= 88 and phi[i][j] <= 92:
                axis_indicat[i][j] = 10
            elif phi[i][j] >= 268 and phi[i][j] <= 272:
                axis_indicat[i][j] = 10
            else:
                axis_indicat[i][j] = np.nan
                
    for k in np.arange(inner, outer, step):
        for i in range(phi.shape[0]):
            for j in range(phi.shape[1]):
                if r_re[i][j] > k and r_re[i][j] < k+step:
                    axis_indicat[i][j] = 100*k
                else:
                    pass
                
                
    rows = 2
    columns = 2
    
    # Plot the 1st image
    fig.add_subplot(rows, columns, 1)
    plt.title("Image")
    im = Image(data)
    spaxel_len = 281*2
    factor = spaxel_len / axis_indicat.shape[0]
    large_mask = ndimage.zoom(axis_indicat, factor, order=0)
    # large_mask[large_mask > 0] = 1
    mask = np.flipud(large_mask)
    im = showImage(plateifu = data)
    masked = np.ma.masked_where(mask == 0, mask)
    plt.imshow(im, interpolation='none')
    plt.imshow(masked, cmap = 'hsv', interpolation='none', alpha=0.3)

    # Plot the 2nd image 
    fig.add_subplot(rows, columns, 2)
    plt.title("OIII Flux Map")
    plt.imshow(oiii_ew, cmap = 'viridis')
    plt.imshow(axis_indicat, cmap = 'hsv', interpolation='none', alpha=0.3)

    
    ellip = ellip_gen(data)
    color = cm.hsv(np.linspace(0, 1, 8))
    
    # Plot 3rd image
    fig.add_subplot(rows, columns, 3)
    plt.title("OIII Flux dR")
    for i,c in zip(np.linspace(0.6, 2, 8), color):
        start, end = round(i,1), round(i,1)+0.3
        curve = ellip_ring_curve(ellip, in_r = start, out_r = end, cycle = 2)
        plt.plot(curve, c = c, alpha=0.5)

        
    # Plot 4th image
    fig.add_subplot(rows, columns, 4)
    plt.title("Fourier Frequency")
    for i,c in zip(np.linspace(0.6, 2, 8), color):
        start, end = round(i,1), round(i,1)+0.3
        curve = ellip_ring_curve(ellip, in_r = start, out_r = end, cycle = 2)
        result = abs(fft(curve))   
        x = np.linspace(0.5, 15, 29)
        plt.plot(x, result[1:30], c = c, alpha=0.5)


    
    if save == True:
        plt.savefig(data+'_vis_image.png')
    else:    
        plt.show()
    
    

def plot_something(data, save = False):
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
    
    st_vel = np.flipud(maps.stellar_vel.value)
    ha_vel = np.flipud(maps.emline_gvel_ha_6564.value)
    oiii_ew = np.flipud(maps.emline_gew_oiii_5008.value)
    
    x_tik = oiii_ew.shape[0]/4
    y_tik = oiii_ew.shape[1]/4
    
    
    phi = maps.spx_ellcoo_elliptical_azimuth.value
    r_re = maps.spx_ellcoo_r_re.value
    
    for i in [oiii_ew, st_vel, ha_vel]:
        mean = np.mean(i)
        sd = np.std(i)
        for j in range(i.shape[0]):
            for k in range(i.shape[1]):
                if i[j][k] >= mean+2*sd or i[j][k]<=mean-2*sd:
                    i[j][k] = np.nan
                elif i[j][k] == 0:
                    i[j][k] = np.nan
                else:
                    pass

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

    
    ellip = Bicone_Classifier.ellip_gen(data)
    color = cm.viridis(np.linspace(0, 1, 8))
    
    # Plot 5rd EW dR
    fig.add_subplot(rows, columns, 3)
    fig.set_figheight(15)
    fig.set_figwidth(30)
    plt.title("OIII EW dR")
    plt.xlabel('degree')
    plt.ylabel('EW/spaxel')
    for i,c in zip(np.linspace(0.6, 2, 8), color):
        start, end = round(i,1), round(i,1)+0.3
        curve = Bicone_Classifier.ellip_ring_curve(ellip, in_r = start, out_r = end, cycle = 2)
        plt.plot(curve, c = c, alpha=0.7)


        
    # Plot 6th FTT
    fig.add_subplot(rows, columns, 6)
    fig.set_figheight(15)
    fig.set_figwidth(30)
    plt.title("Fourier Frequency")
    plt.xlabel('Hz')
    for i,c in zip(np.linspace(0.6, 2, 8), color):
        start, end = round(i,1), round(i,1)+0.3
        curve = Bicone_Classifier.ellip_ring_curve(ellip, in_r = start, out_r = end, cycle = 2)
        result = abs(fft(curve))   
        x = np.linspace(0.5, 15, 29)
        plt.plot(x, result[1:30], c = c, alpha=0.7)



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
    
