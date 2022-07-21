# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================================================


def pplot(nx,ny,twodprop, indplot_prop, fig, gs, subx, suby, propname, vmax=None, vmin=None,
          scale=None,equal=False):
    # nx,ny表示坐标网格
    # twodprop-二维性质图, indplot_prop-对二维性质图的坐标限制
    # fig为总图，gs表示设置的子图分布
    # subx, suby表示subplot的位置
    # propname表示子图名称
    # equal表示标度是否设置为最大值最小值的绝对值相同
    
    map_twodprop = np.full_like(twodprop, np.nan)
    map_twodprop[indplot_prop] = twodprop[indplot_prop]
        
    ax = fig.add_subplot(gs[subx,suby])
    
    # =========================================================================
    # 调整绘图区域的大小，使之尽可能布满整个坐标网，即去掉map_twodprop中的nan列
    # for ind_nan_x in range(0,int(map_twodprop.shape[0]/2)):
    #     if (np.sum(~np.isnan(map_twodprop[ind_nan_x])) == 0 and 
    #         np.sum(~np.isnan(map_twodprop[map_twodprop.shape[0]-1-ind_nan_x])) ==0):
    #         ind_nan_x = ind_nan_x + 1
    #     else:
    #         break
    # for ind_nan_y in range(0,int(map_twodprop.shape[1]/2)):
    #     if (np.sum(~np.isnan(map_twodprop[:,ind_nan_y])) == 0 and 
    #         np.sum(~np.isnan(map_twodprop[:,map_twodprop.shape[1]-1-ind_nan_y])) ==0):
    #         ind_nan_y = ind_nan_y + 1
    #     else:
    #         break
    # ind_nan_d4000 = min(ind_nan_x, ind_nan_y)

    # map_twodprop_new = map_twodprop[ind_nan_d4000-1:map_twodprop.shape[0]-ind_nan_d4000
    #                                 ,ind_nan_d4000-1:map_twodprop.shape[1]-ind_nan_d4000]
    # =========================================================================
    # 但是以上方法使得坐标尺度不再以角秒为单位
    # map_twodprop_new = map_twodprop # 修改为原列表，以角秒为单位
    
    #### 调整colorbar的尺度 ################
    #### 如果超过scale的最大值与最小值，则color bar最大值取85百分位，最小值取15百分位 #####
    
    # if vmin != -100:
    #     minvalue = vmin
    # if 'vmax' in dir():
    #     maxvalue = vmax
    
    if scale is not None:
        if vmax is not None:
            maxvalue = vmax
        else:
            if np.nanmax(map_twodprop) < scale:
                maxvalue = np.nanmax(map_twodprop)
            else:
                maxvalue = np.nanpercentile(map_twodprop,85)
        if vmin is not None:
            minvalue = vmin
        else:
            if np.nanmin(map_twodprop) > -scale:
                minvalue = np.nanmin(map_twodprop)
            else:
                minvalue = np.nanpercentile(map_twodprop,15)
    elif scale is None:
        maxvalue = vmax
        minvalue = vmin
    
    if equal:
        tmp = max(abs(maxvalue),abs(minvalue))
        maxvalue = tmp
        minvalue = -tmp
                

    # =======================================
    # 画图
    im = ax.imshow(map_twodprop, vmin = minvalue, vmax = maxvalue,
                   extent = (np.min(nx),np.max(nx),np.min(ny),np.max(ny)),
                    origin= 'lower',cmap = plt.cm.jet)    
    cb = fig.colorbar(im,ax = ax,fraction=0.05)
    cb.ax.tick_params(direction='in', labelsize = 15, length = 5, width=1.0)
    
    xpos, ypos = np.meshgrid(nx, ny, sparse=False, indexing='xy')
    ax.set_title(propname,fontsize=25)
    corr_min = min(np.min(xpos[indplot_prop]),np.min(ypos[indplot_prop]))-0.5
    corr_max = max(np.max(xpos[indplot_prop]),np.max(ypos[indplot_prop]))+0.5
    ax.set_xlim(corr_min,corr_max)
    ax.set_ylim(corr_min,corr_max)
    return ax
