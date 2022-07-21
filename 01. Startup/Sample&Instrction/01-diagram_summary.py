# -*- coding: utf-8 -*-
from astropy.io import fits
from os.path import isfile
from matplotlib.colors import ListedColormap
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import pplot

import warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# BPT分类，NII和SII图
def classify_BPT_N2(O3,N2,Ha,Hb):
    # OIII-NII部分
    # 区分pure star-forming,composite与AGN    
    # 注意，双曲线中的大于小于要非常小心
    # 定义NII的BPT图的两条线为
    # Ka03: y=0.61/(x-0.05)+1.3(x<0.05)
    # Ke01: y=0.61/(x-0.47)+1.19(x<0.47)
    x = np.log10(N2)-np.log10(Ha)
    y = np.log10(O3)-np.log10(Hb)
    
    prop_o3_n2_SF = (
                     ((y<0.61/(x-0.05)+1.3) & (x<0.05) & (y<0.61/(x-0.47)+1.19))
                     & (np.isfinite(x) & np.isfinite(y))
                     )
    
    prop_o3_n2_comp = ((
                        ((y>0.61/(x-0.05)+1.3) & (y<0.61/(x-0.47)+1.19) & (x<0.05))
                        |
                        ((x>=0.05) & (x<0.47) & (y<0.61/(x-0.47)+1.19))
                        ) 
                          & (np.isfinite(x)&np.isfinite(y))
                          )
    
    prop_o3_n2_AGN = ((
                       ((y>0.61/(x-0.47)+1.19) & (x<0.47))
                       |
                       (x>=0.47)
                       )
                       & (np.isfinite(x)&np.isfinite(y))
                       )
    
    
    return prop_o3_n2_SF, prop_o3_n2_comp, prop_o3_n2_AGN

def classify_BPT_S2(O3,S2,Ha,Hb):
    # OIII-SII部分；区分Seyfert和LIER
    # 定义SII的BPT图的两条线为
    # y = 0.72/(x-0.32)+1.3; y = 1.89*x+0.76
    x = np.log10(S2)-np.log10(Ha)
    y = np.log10(O3)-np.log10(Hb)
    
    prop_o3_s2_SFcomp = (
                        ((y<0.72/(x-0.32)+1.3) & (x<0.32))
                         &(np.isfinite(x) & np.isfinite(y))
                         )
    
    prop_o3_s2_Seyf = (
                        (((y>0.72/(x-0.32)+1.3) & (x<0.32) & (y>1.89*x+0.76))
                        | 
                        ((x>=0.32) & (y>1.89*x+0.76)))
                        & (np.isfinite(x)&np.isfinite(y))
                        )
    prop_o3_s2_LIER = (
                        (((y>0.72/(x-0.32)+1.3) & (x<0.32) & (y<1.89*x+0.76))
                        | 
                        ((x>=0.32) & (y<1.89*x+0.76)))
                        & (np.isfinite(x)&np.isfinite(y))
                        )
    
    return prop_o3_s2_SFcomp, prop_o3_s2_Seyf, prop_o3_s2_LIER

# ===============================================================================
# 消光改正
def extinc_corr(flux_ha,flux_hb,flux_eml,eml_name):
    line_name = np.array(['OII-3727', 'OII-3729', 'Hb-4862', 'OIII-4960', 
                 'OIII-5008', 'OI-6302', 'NII-6549', 'Ha-6564', 
                 'NII-6585', 'SII-6718', 'SII-6732'])
    line_center = np.array([3727.09, 3729.88, 4862.69, 4960.30, 
                         5008.24, 6302.04, 6549.84, 6564.61, 
                         6585.23, 6718.32, 6732.71])
    
    ind_line = np.argwhere(line_name==eml_name)[0][0]
    # Cardelli et al. 1989
    x = 10000/line_center[ind_line]
    y = x - 1.82
    a = (1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 -
          0.77530*y**6 + 0.32999*y**7)
    b = (1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 +
          5.30260*y**6 - 2.09002*y**7)
    k_lambda = (a+b/3.1)*3.1
    
    # Calibration--Tremonti
    E_BV = np.full_like(flux_eml,0,dtype='float32')
    ind_ebv = (flux_ha>0)&(flux_hb>0)
    E_BV[ind_ebv] = 0.934 * np.log(flux_ha[ind_ebv]/flux_hb[ind_ebv]/2.86)
    
    ind_neg = (E_BV<0)
    E_BV[ind_neg] = 0
    
    flux_eml_corr = np.full_like(flux_eml,np.nan)
    flux_eml_corr[ind_ebv] = flux_eml[ind_ebv]/(10**(-0.4*k_lambda*E_BV[ind_ebv]))
    
    return flux_eml_corr

# ==============================================================================
# 对流量数据的应用
def flux_data(flux,flux_ivar,flux_mask,eml_name):
    eml_channel = np.array(['OII-3727','OII-3729','H12-3751','H11-3771','Hthe-3798',
                         'Heta-3836','NeIII-3869','HeI-3889','Hzet-3890','NeIII-3968',
                         'Heps-3971','Hdel-4102','Hgam-4341','HeII-4687','Hb-4862',
                         'OIII-4960','OIII-5008','NI-5199','NI-5201','HeI-5877',
                         'OI-6302','OI-6365','NII-6549','Ha-6564','NII-6585',
                         'SII-6718','SII-6732','HeI-7067','ArIII-7137','ArIII-7753',
                         'Peta-9017','SIII-9071','Pzet-9231','SIII-9533','Peps-9548'])
    ind_eml = np.argwhere(eml_channel==eml_name)[0][0]
    flux_eml = flux[ind_eml]
    flux_eml_ivar = flux_ivar[ind_eml]
    flux_eml_mask = flux_mask[ind_eml]
    snr_eml = flux_eml*np.sqrt(flux_eml_ivar)
    indplot_eml = (flux_eml_ivar>0)&((flux_eml_mask&2**30)==0)&(snr_eml>3)
    return flux_eml,indplot_eml

# ===============================================================================
# 从文件读取数据
miscont = fits.getdata('G:/work/test/thesis_program/General_Property/'+
                       'General_Property_V12/00-data/'+
                       'sampgal_sampmis_10control_mpl10.fits',1)
sampgal = fits.getdata('G:/work/test/thesis_program/General_Property/'+
                    'General_Property_V12/morphological_type/'+
                    '02-sampgal_morphology_cross_Chang.fits')
PA = fits.getdata('G:/work/test/thesis_program/General_Property/'+
                  'General_Property_V12/00-data/04-uniq_sampgal_cross_Chang.fits',1)
samp = fits.getdata('G:/work/DATA/dapall-v3_0_1-3.0.1-1.fits',1)
spin = fits.getdata('G:/work/test/thesis_program/General_Property/'+
                    'General_Property_V12/05-LambdaRe/'+
                    '02-cross_spin_ellipticity_with_chang.fits',1)
Chang = fits.getdata('G:/work/test/thesis_program/General_Property/'+
                     'General_Property_V12/00-data/04-uniq_sampgal_cross_Chang.fits',1)
AGN_class = fits.getdata('G:/work/test/thesis_program/General_Property/'+
                         'General_Property_V12/06-AGN_fraction/02-AGN_selection.fits',1)

# ===============================================================================
end_flg = 'begin'

for ind_miscont in tqdm(range(0,len(miscont))):
    plateifu_val = miscont['misalign'][ind_miscont]
    
    plate = plateifu_val.split('-')[0]
    ifu = plateifu_val.split('-')[1]
    
    ind_sampgal = np.argwhere(sampgal['plateifu']==plateifu_val)[0][0]
    ind_samp = np.argwhere(samp['plateifu']==plateifu_val)[0][0]
    ind_spin = np.argwhere(spin['plateifu']==plateifu_val)[0][0]
    ind_Chang = np.argwhere(Chang['plateifu']==plateifu_val)[0][0]
    ind_PA = np.argwhere(PA['plateifu']==plateifu_val)[0][0]
    ind_AGN_class = np.argwhere(AGN_class['plateifu']==plateifu_val)[0][0]
    
    # ==========================================================================
    # 导入MPL-10数据
    dir_cube1 = 'G:/work/MPL-10/MAPs/'+plate+'/'+ifu+'/'
    dir_cube2 = 'manga-'+plateifu_val+'-MAPS-SPX-MILESHC-MASTARHC2.fits.gz'
    if isfile(dir_cube1+dir_cube2)==False:
        continue
    cube = fits.open(dir_cube1+dir_cube2)
    
    snr = cube['SPX_SNR'].data
    
    # 分bin面积
    bin_area = cube['BIN_AREA'].data
    
    # 恒星速度
    ste_v = cube['STELLAR_VEL'].data
    ste_v_ivar = cube['STELLAR_VEL_IVAR'].data
    ste_v_mask = cube['STELLAR_VEL_MASK'].data
    
    # =============================================
    # 恒星速度弥散
    # 必须要对速度弥散作修正
    ste_v_sigma = cube['STELLAR_SIGMA'].data
    ste_v_sigma_ivar = cube['STELLAR_SIGMA_IVAR'].data
    ste_v_sigma_mask = cube['STELLAR_SIGMA_MASK'].data
    # https://sdss-mangadap.readthedocs.io/en/latest/corrections.html#corrections
    # For now, use the correction in the first channel of the STELLAR_SIGMACORR 
    # extension until the data in the second channel can be vetted.
    ste_v_sigma_corr = cube['STELLAR_SIGMACORR'].data[0]
    
    # 速度弥散只对修改值大于0的部分进行修改
    prop_corr = (ste_v_sigma**2-ste_v_sigma_corr**2)>0
    ste_v_sigma_new = ste_v_sigma
    ste_v_sigma_new[prop_corr] = np.sqrt(ste_v_sigma[prop_corr]**2
                                          -ste_v_sigma_corr[prop_corr]**2)
    
    # Ha流量   
    flux = cube['EMLINE_GFLUX'].data
    flux_ivar = cube['EMLINE_GFLUX_IVAR'].data
    flux_mask = cube['EMLINE_GFLUX_MASK'].data
        
    flux_ha = flux[23]
    flux_ha_ivar = flux_ivar[23]
    flux_ha_mask = flux_mask[23]
    flux_ha_err = 1./ np.sqrt(flux_ha_ivar)
    snr_ha = flux_ha/flux_ha_err
    pix = int(np.sqrt(snr_ha.size)/2)
        
    # Ha速度    
    v_gas = cube['EMLINE_GVEL'].data
    v_gas_ivar = cube['EMLINE_GVEL_IVAR'].data
    v_gas_mask = cube['EMLINE_GVEL_MASK'].data
        
    v_ha = v_gas[23]
    v_ha_ivar = v_gas_ivar[23]
    v_ha_mask = v_gas_mask[23]
    
    # =============================================
    # 气体速度弥散
    gas_v_sigma = cube['EMLINE_GSIGMA'].data[23]
    gas_v_sigma_ivar = cube['EMLINE_GSIGMA_IVAR'].data[23]
    gas_v_sigma_mask = cube['EMLINE_GSIGMA_MASK'].data[23]
    gas_v_sigma_corr = cube['EMLINE_INSTSIGMA'].data[23]
    
    # 速度弥散只对修改值大于0的部分进行修改
    prop_corr = (gas_v_sigma**2-gas_v_sigma_corr**2)>0
    gas_v_sigma_new = gas_v_sigma
    gas_v_sigma_new[prop_corr] = np.sqrt(gas_v_sigma[prop_corr]**2
                                          -gas_v_sigma_corr[prop_corr]**2)
    
    # ============================================
    # Dn4000
    Dn4000_map = cube['SPECINDEX'].data[44]
    Dn4000_ivar = cube['SPECINDEX_IVAR'].data[44]
    Dn4000_mask = cube['SPECINDEX_MASK'].data[44]
    indplot_Dn4000 = (Dn4000_ivar>0)&((Dn4000_mask&2**30)==0)
    
    # ============================================
    
    GEW = cube['EMLINE_GEW'].data
    GEW_ivar = cube['EMLINE_GEW_IVAR'].data
    GEW_mask = cube['EMLINE_GEW_MASK'].data
    
    # [OIII]等值宽度
    EW_o3_5008, indplot_EW_o3_5008 = flux_data(GEW,GEW_ivar,GEW_mask,'OIII-5008')
    # Ha等值宽度
    EW_ha, indplot_EW_ha = flux_data(GEW,GEW_ivar,GEW_mask,'Ha-6564')

    # 构造坐标轴
    # 坐标轴的定义要求每个spaxel的间隔为0.5
    # 这就是坐标如下定义的原因
    nx = (np.arange(ste_v.shape[0]) - ste_v.shape[0]/2)/2
    ny = (np.arange(ste_v.shape[1]) - ste_v.shape[1]/2)/2
    xpos, ypos = np.meshgrid(nx, ny, sparse=False, indexing='xy')
    # xpos = -xpos
      
    p50 = samp['NSA_ELPETRO_TH50_R'][ind_samp]
    # 注意，MaNGA中光学主轴的角度定义是从北边开始逆时针计算
    # 将椭圆从倾斜的位置修正到标准位置则需要减去90°
    pa_std = (-90 + samp['NSA_ELPETRO_PHI'][ind_samp])/180*np.pi
    ba = samp['NSA_ELPETRO_BA'][ind_samp]
    # 沿主轴方向，坐标不变；副轴方向，坐标拉长ba
    # 事实上设主轴为x'轴，副轴为y'轴，则若x=r*cos(theta),y=r*sin(theta)
    # x'=r*cos(theta-(phi-90)),y'=r*sin(theta-(phi-90))
    # 即x'=xcos(phi-90)+ycos(phi-90),y'=-xsin(phi-90)+ycos(phi-90)
    # 而phi-90即pa_std
    xpos2 = xpos*np.cos(pa_std)+ypos*np.sin(pa_std)
    ypos2 = (-xpos*np.sin(pa_std)+ypos*np.cos(pa_std))/ba
    r2 = np.sqrt(xpos**2+ypos**2)
    r0 = np.sqrt(xpos2**2+ypos2**2)
    r0 = r0/p50
    rad = np.sqrt(np.max(xpos**2+ypos**2))
    
    indplot_ste_high = (ste_v_ivar>0) & ((ste_v_mask&2**30)==0) & (r0 < 1.5)
    indplot_ha_high = (v_ha_ivar>0) & ((v_ha_mask&2**30)==0) & (r0<1.5)
    indplot_stesig_high = (ste_v_sigma_ivar>0) & ((ste_v_sigma_mask&2**30)==0) & (r0<1.5)
    indplot_ha = (flux_ha_ivar>0) & ((flux_ha_mask&2**30)==0) & (snr_ha>3)
    indplot_gas_sigma = ((gas_v_sigma_ivar>0)&((gas_v_sigma_mask&2**30)==0)&indplot_ha)
    
    # =================================================
    # BPT分类
    flux_ha, indplot_ha = flux_data(flux,flux_ivar,flux_mask,'Ha-6564')
    flux_hb, indplot_hb = flux_data(flux,flux_ivar,flux_mask,'Hb-4862')
    flux_o3_5008, indplot_o3_5008 = flux_data(flux,flux_ivar,flux_mask,'OIII-5008')
    flux_n2_6585, indplot_n2_6585 = flux_data(flux,flux_ivar,flux_mask,'NII-6585')
    
    SF, comp, AGN_NII = classify_BPT_N2(flux_o3_5008,flux_n2_6585,flux_ha,flux_hb)
    
    flux_s2_6732, indplot_s2_6732 = flux_data(flux,flux_ivar,flux_mask,'SII-6732')
    flux_s2_6718, indplot_s2_6718 = flux_data(flux,flux_ivar,flux_mask,'SII-6718')
    SFcomp, Seyfert, LIER = classify_BPT_S2(flux_o3_5008,
                                            flux_s2_6718+flux_s2_6732,flux_ha,flux_hb)
    
    # 限制区域
    limit_NII = indplot_ha & indplot_hb & indplot_o3_5008 & indplot_n2_6585
    limit_SII = (indplot_ha & indplot_hb & indplot_o3_5008 
                  & indplot_s2_6732 & indplot_s2_6718)
    
    NII_no_snr = SF + 2*comp + 3*AGN_NII
    SII_no_snr = SFcomp + 2*Seyfert + 3*LIER
       
    # ================================
    # 画图准备
    # 图1：星系光学图像
    fig = plt.figure(tight_layout=True,figsize = (28,20))
    gs = plt.GridSpec(4, 6, wspace=0.3, hspace=0.3)
        
    loc1 = 'G:/work/MPL-10/images_v3_0_1/'
    dir1 = loc1 + 'SDSS_manga-'+plateifu_val +'.png'
    if isfile(dir1) == True:
        im = plt.imread(dir1)
        ax0 = fig.add_subplot(gs[0, 0:2])
        ax0.imshow(im)
        ax0.axis('off')
        ax0.set_title(plateifu_val,fontsize=25)
        
    dir2 = 'G:/work/bundle-images/'+plateifu_val+'.jpg'
    if isfile(dir2)==True:
        im = plt.imread(dir2)
        ax1 = fig.add_subplot(gs[0:2, 3:5])
        ax1.imshow(im)
        ax1.axis('off')
        
    # 恒星速度盘
    ax2 = pplot.pplot(nx,ny,twodprop=ste_v, indplot_prop=indplot_ste_high, 
                      fig = fig, gs=gs, subx=1, suby=0, 
                      propname=r'$\rm{v_{stellar}}$',scale=50,equal=True)
    ang_ste = [0,np.pi] + np.radians(sampgal['ang_stellar'][ind_sampgal])
    ang_ste_err = np.radians(sampgal['ang_stellar_err'][ind_sampgal])
    ax2.plot(-rad*np.sin(ang_ste),rad*np.cos(ang_ste),'-',color='limegreen',linewidth='2')
    ax2.plot(-rad*np.sin(ang_ste+ang_ste_err),rad*np.cos(ang_ste+ang_ste_err)
              ,'k--',color='blue',linewidth='2')
    ax2.plot(-rad*np.sin(ang_ste-ang_ste_err),rad*np.cos(ang_ste-ang_ste_err)
              ,'k--',color='blue',linewidth='2')
    
    # 气体速度盘
    ax3 = pplot.pplot(nx,ny,twodprop=v_ha, indplot_prop=indplot_ha_high, 
                      fig = fig, gs=gs, subx=1, suby=1, 
                      propname=r'$\rm{v_{H\alpha}}$',scale=50,equal=True)
    ang_ha = [0,np.pi]+np.radians(sampgal['ang_ha'][ind_sampgal])
    ang_ha_err = np.radians(sampgal['ang_ha_err'][ind_sampgal])
    ax3.plot(-rad*np.sin(ang_ha),rad*np.cos(ang_ha),'-',color='limegreen',linewidth='2')
    ax3.plot(-rad*np.sin(ang_ha+ang_ha_err),rad*np.cos(ang_ha+ang_ha_err)
              ,'k--',color='blue',linewidth='2')
    ax3.plot(-rad*np.sin(ang_ha-ang_ha_err),rad*np.cos(ang_ha-ang_ha_err)
              ,'k--',color='blue',linewidth='2')
    
    # 恒星速度弥散
    ax4 = pplot.pplot(nx,ny,twodprop=ste_v_sigma_new, indplot_prop=indplot_stesig_high, 
                      fig = fig, gs=gs, subx=2, suby=0, 
                      propname=r'$\rm{\sigma_{stellar}}$',vmin=0,vmax=250,equal=False)
    
    # 气体速度弥散
    ax5 = pplot.pplot(nx,ny,twodprop=gas_v_sigma_new, indplot_prop=indplot_gas_sigma, 
                      fig = fig, gs=gs, subx=2, suby=1, 
                      propname=r'$\rm{\sigma_{gas}}$',vmin=0,vmax=250,equal=False)
    
    # ===================================================================
    # [NII]-BPT空间分辨图
    colors = ['white', 'blue', 'yellow', 'red','black']
    ax6 = fig.add_subplot(gs[2,3])
    
    NII_snr = np.full_like(NII_no_snr,np.nan)
    NII_snr[limit_NII] = NII_no_snr[limit_NII]
    
    im = ax6.imshow(NII_snr, vmin = 0, vmax = 4,
                    cmap=ListedColormap(colors),
                    extent = (np.min(nx),np.max(nx),np.min(ny),np.max(ny)),
                    origin='lower')
    cb = fig.colorbar(im,ax = ax6,fraction=0.05)
    cb.ax.tick_params(direction='in', labelsize = 10, length = 5, width=0.5)
    cb.set_ticks([0,1,2,3,4])
    cb.ax.set_yticklabels(['Not define','SF','composite','AGN','invalid'])
    ax6.set_title(r'$\rm{class_{NII}}$',fontsize=25)
    im = ax6.imshow(bin_area!=0,vmin=0,vmax=1,alpha=0.2,cmap=ListedColormap(['grey','None']),
                    extent = (np.min(nx),np.max(nx),np.min(ny),np.max(ny)),
                    origin='lower')
    
    # [SII]-BPT空间分辨图
    colors = ['white','blue','red','purple','black']
    ax7 = fig.add_subplot(gs[2,4])
    
    SII_snr = np.full_like(SII_no_snr,np.nan)
    SII_snr[limit_SII] = SII_no_snr[limit_SII]
    
    im = ax7.imshow(SII_snr, vmin = 0, vmax = 4,
                    cmap=ListedColormap(colors),
                    extent = (np.min(nx),np.max(nx),np.min(ny),np.max(ny)),
                    origin='lower')
    cb = fig.colorbar(im,ax = ax7)
    cb.ax.tick_params(direction='in', labelsize = 10, length = 5, width=0.5)
    cb.set_ticks([0,1,2,3,4])
    cb.ax.set_yticklabels(['Not define','SFcomp','Seyfert','LIER','invalid'])
    ax7.set_title(r'$\rm{class_{SII}}$',fontsize=25)
    im = ax7.imshow(bin_area!=0,vmin=0,vmax=1,alpha=0.2,cmap=ListedColormap(['grey','None']),
                    extent = (np.min(nx),np.max(nx),np.min(ny),np.max(ny)),
                    origin='lower')
    
    # ==========================================================================
    # [NII]-BPT曲线图
    ax8 = fig.add_subplot(gs[3,3])
    
    x = np.log10(flux_n2_6585[limit_NII])-np.log10(flux_ha[limit_NII])
    y = np.log10(flux_o3_5008[limit_NII])-np.log10(flux_hb[limit_NII])    
    r = np.sqrt(xpos[limit_NII]**2+ypos[limit_NII]**2)
    
    im = ax8.scatter(x,y,c=r,cmap=plt.cm.jet,marker='o',alpha=0.5)
    cb = fig.colorbar(im,ax = ax8)
    
    cb.ax.tick_params(direction='in', labelsize = 10, length = 5, width=0.5)
    
    x1 = np.linspace(-1.2,0.05,100)
    y1 = 0.61/(x1-0.05)+1.3
    x2 = np.linspace(-3.2,0.47,100)
    y2 = 0.61/(x2-0.47)+1.19
    ax8.plot(x1,y1,'k--',color='red')
    ax8.plot(x2,y2,color='blue')
    ax8.set_xlim(-1.0,0.4)
    ax8.set_ylim(-1.5,1.0)
    ax8.set_title(r'$\rm{BPT_{NII}}$',fontsize=20)
    
    
    # [SII]-BPT曲线图
    ax9 = fig.add_subplot(gs[3,4])
    
    x = (np.log10(flux_s2_6732[limit_SII]+flux_s2_6718[limit_SII])
          -np.log10(flux_ha[limit_SII]))
    y = np.log10(flux_o3_5008[limit_SII])-np.log10(flux_hb[limit_SII])
    r = np.sqrt(xpos[limit_SII]**2+ypos[limit_SII]**2)
    
    im = ax9.scatter(x,y,c=r,cmap=plt.cm.jet,marker='o',alpha=0.5)
    cb = fig.colorbar(im,ax = ax9)
    cb.ax.tick_params(direction='in', labelsize = 10, length = 5, width=0.5)

    x1 = np.linspace(-3,0.32,100)
    y1 = 0.72/(x1-0.32)+1.3
    x2 = np.linspace(-0.3,0.46,100)
    y2 = 1.89*x2+0.76
    ax9.plot(x1,y1,'k--',color='red')
    ax9.plot(x2,y2,color='blue')
    ax9.set_xlim(-1.5,1)
    ax9.set_ylim(-1.5,1.5)
    ax9.set_title(r'$\rm{BPT_{SII}}$',fontsize=25)
    # ==========================================================================
    # M*-SFR图
    ax10 = fig.add_subplot(gs[0,2])
    
    # SF,GV,QS分界线
    # SF/GV分界线：y = 0.86*x-9.29
    # GV/QS分界线：y = x-14.65
    x = np.linspace(8,12,100)
    y1 = 0.86*x-9.29
    y2 = x-14.65
    ax10.plot(x,y1,color='black',linestyle='--')
    ax10.plot(x,y2,color='black',linestyle='--')
    
    # 设置x,y轴范围
    ax10.set_xlim([8.5,12])
    ax10.set_ylim([-6,2])
    ax10.set_yticks(np.linspace(-6,1,8))
    ax10.set_xticks(np.linspace(8.5,11.5,7))
    
    mass_array = Chang['mass']
    SFR_array = Chang['SFR']
    mass_val = Chang['mass'][ind_Chang]
    SFR_val = Chang['SFR'][ind_Chang]
    ax10.scatter(mass_array,SFR_array,edgecolor='grey',facecolor='None',alpha=0.2)
    ax10.scatter(mass_val,SFR_val,s=80,facecolor='yellow',edgecolor='blue',
                  marker='^',linewidth=2)
    # ==========================================================================
    # lambda_Re
    ax11 = fig.add_subplot(gs[1,2])
    
    # slow rotator区域
    x = np.linspace(0,0.4,100)
    y = 0.08+x/4
    ax11.plot(x,y,color='black',linestyle='-',linewidth=2)
    y = np.linspace(0,0.08+0.4/4,100)
    x = np.full_like(y,0.4)
    ax11.plot(x,y,color='black',linestyle='-',linewidth=2)
    
    # 虚线作图
    eps = np.linspace(0,1,100)
    e = np.sqrt(1-(1-eps)**2)
    omega = 0.5*(np.arcsin(e)/np.sqrt(1-e**2)-e)/(e-np.arcsin(e)*np.sqrt(1-e**2))
    alpha = 0.15
    delta = 0.7*eps
    vsig_sq = ((1-delta)*omega-1)/(alpha*(1-delta)*omega+1)
    k = 1.1
    lamb = k*np.sqrt(vsig_sq)/np.sqrt(1+k**2*vsig_sq)
    ax11.plot(eps,lamb,color='magenta',linestyle='-',linewidth=2)
    
    i_const = np.radians(np.array([15,30,45,60,75]))
    for ind in range(0,len(i_const)):
        eps_obs = 1-np.sqrt(1+eps*(eps-2)*np.sin(i_const[ind])**2)
        vsig_sq_obs = vsig_sq*(np.sin(i_const[ind]))**2/(1-delta*np.cos(i_const[ind])**2)
        lamb_obs = k*np.sqrt(vsig_sq_obs)/np.sqrt(1+k**2*vsig_sq_obs)
        ax11.plot(eps_obs,lamb_obs,color='magenta',linestyle='--',linewidth=2)
        
    eps_int = np.array([0,0.2,0.4,0.6,0.8,0.9,0.98])
    alpha=0.15
    k=1.1
    
    i = np.radians(np.linspace(0,90,90))
    for ind in range(0,len(eps_int)):
        eps_obs = 1-np.sqrt(1+eps_int[ind]*(eps_int[ind]-2)*np.sin(i)**2)
        e = np.sqrt(1-(1-eps_int[ind])**2)
        delta = 0.7*eps_int[ind]
        omega = 0.5*(np.arcsin(e)/np.sqrt(1-e**2)-e)/(e-np.arcsin(e)*np.sqrt(1-e**2))
        vsig_sq = ((1-delta)*omega-1)/(alpha*(1-delta)*omega+1)
        vsig_sq_obs = vsig_sq*np.sin(i)/np.sqrt(1-delta*np.cos(i)**2)
        lamb_obs = k*np.sqrt(vsig_sq_obs)/np.sqrt(1+k**2*vsig_sq_obs)
        ax11.plot(eps_obs,lamb_obs,color='magenta',linestyle=':',linewidth=2)
        
    lambda_Re_val = spin['lambda_Re'][ind_spin]
    epsilon_val = spin['epsilon'][ind_spin]
    lambda_Re_array = spin['lambda_Re']
    epsilon_array = spin['epsilon']
    ax11.scatter(epsilon_array,lambda_Re_array,edgecolor='grey',alpha=0.1,facecolor='None')
    ax11.scatter(epsilon_val,lambda_Re_val,s=80,facecolor='yellow',edgecolor='blue',
                  marker='^',linewidth=2)
    ax11.set_xlim(0,1)
    
    # ===========================================================================
    # stellar v/sigma
    ax13 = pplot.pplot(nx,ny,twodprop=np.abs(ste_v)/ste_v_sigma_new, 
                        indplot_prop=indplot_stesig_high, 
                      fig = fig, gs=gs, subx=3, suby=0, 
                      propname=r'$\rm{v/\sigma(star)}$',vmin=0,vmax=2,equal=False)
    im = ax13.imshow(bin_area!=0,vmin=0,vmax=1,alpha=0.2,cmap=ListedColormap(['grey','None']),
                    extent = (np.min(nx),np.max(nx),np.min(ny),np.max(ny)),
                    origin='lower')
    # gas v/sigma
    ax14 = pplot.pplot(nx,ny,twodprop=np.abs(v_ha)/gas_v_sigma_new, indplot_prop=indplot_gas_sigma, 
                      fig = fig, gs=gs, subx=3, suby=1, 
                      propname=r'$\rm{v/\sigma(gas)}$',vmin=0,vmax=2,equal=False)
    im = ax14.imshow(bin_area!=0,vmin=0,vmax=1,alpha=0.2,cmap=ListedColormap(['grey','None']),
                    extent = (np.min(nx),np.max(nx),np.min(ny),np.max(ny)),
                    origin='lower')
    
    # Dn4000
    ax15 = pplot.pplot(nx,ny,twodprop=Dn4000_map, indplot_prop=indplot_Dn4000, 
                      fig = fig, gs=gs, subx=3, suby=2, vmin=1, vmax=2,
                      propname=r'$\rm{D_n4000}$',equal=False)
    im = ax15.imshow(bin_area!=0,vmin=0,vmax=1,alpha=0.2,cmap=ListedColormap(['grey','None']),
                    extent = (np.min(nx),np.max(nx),np.min(ny),np.max(ny)),
                    origin='lower')
    
    # ===========================================================================
    # Halpha流量
    ax17 = pplot.pplot(nx,ny,twodprop=flux_ha, indplot_prop=indplot_ha, 
                      fig = fig, gs=gs, subx=2, suby=2, vmin=0,
                      propname=r'$\rm{flux(H\alpha)}$',scale=50,equal=False)
    im = ax17.imshow(bin_area!=0,vmin=0,vmax=1,alpha=0.2,cmap=ListedColormap(['grey','None']),
                    extent = (np.min(nx),np.max(nx),np.min(ny),np.max(ny)),
                    origin='lower')
    
    # ===========================================================================
    # [OIII] EW图
    ax12 = pplot.pplot(nx,ny,twodprop=EW_o3_5008, indplot_prop=indplot_EW_o3_5008, 
                      fig = fig, gs=gs, subx=1, suby=5, vmin=0,
                      propname=r'$\rm{EW_{[OIII]}}$',scale=50,equal=False)
    im = ax12.imshow(bin_area!=0,vmin=0,vmax=1,alpha=0.2,cmap=ListedColormap(['grey','None']),
                    extent = (np.min(nx),np.max(nx),np.min(ny),np.max(ny)),
                    origin='lower')
    
    # ===========================================================================
    # Halpha EW图
    ax18 = pplot.pplot(nx,ny,twodprop=EW_ha, indplot_prop=indplot_EW_ha, 
                      fig = fig, gs=gs, subx=2, suby=5, vmin=0,
                      propname=r'$\rm{EW_{H\alpha}}$',scale=50,equal=False)
    im = ax18.imshow(bin_area!=0,vmin=0,vmax=1,alpha=0.2,cmap=ListedColormap(['grey','None']),
                    extent = (np.min(nx),np.max(nx),np.min(ny),np.max(ny)),
                    origin='lower')
    
    # 综合信息
    ax16 = fig.add_subplot(gs[0, 5])
    ax16.axis('off')
    ax16.text(0,1,'SF type:'+sampgal['SF_GV_QS'][ind_sampgal],fontsize=18)
    ax16.text(0,0.8,'Morph type:'+sampgal['morph_class'][ind_sampgal],fontsize=18)
    ax16.text(0,0.6,r'$\Delta$PA:'+str(PA['DeltaPA_revise'][ind_PA]),fontsize=18)
    ax16.text(0,0.4,'AGN class:'+str(AGN_class['AGN_class'][ind_AGN_class]),fontsize=18)
    
    if sampgal['morph_class'][ind_sampgal]=='S':
        morph_val = 'LTG'
    elif sampgal['morph_class'][ind_sampgal]=='E':
        morph_val = 'ETG'
    elif sampgal['morph_class'][ind_sampgal]=='S0':
        morph_val = 'S0'
    else:
        morph_val = 'not_classify'
        
    fig.savefig('./pic/misalign/'+morph_val+'/'+sampgal['SF_GV_QS'][ind_sampgal]
                    +'/'+plateifu_val+'.png')

    fig.clear()
    plt.close(fig)
    cube.close()
    # plt.show()
    # end_flg = input()
    # if end_flg=='q':
    #     break
    