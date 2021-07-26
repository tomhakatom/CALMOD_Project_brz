# Codes used for the cloud base and coupling calculations
import numpy as np
import netCDF4 as nc4
import misc_codes
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# Initialization for retrieving cloud base temperature heights
lut_w = np.genfromtxt('/home/tgoren/research/analysis/calmod/codes/lut_w.txt') # look up table for mixing ratio
lut_gamma_w = np.genfromtxt('/home/tgoren/research/analysis/calmod/codes/lut_gamma_w.txt') # look up table for saturated lapse rate
lut_z = np.genfromtxt('/home/tgoren/research/analysis/calmod/codes/lut_z.txt') # look up table for cloud layerdepth
lut_ro = np.genfromtxt('/home/tgoren/research/analysis/calmod/codes/lut_ro.txt') # look up table for density

# Calculating cloud base temperature and height
def get_cld_H(T_top, lwp):
    # The code uses satellite obsrved cloud top temperature (kelvin) and LWP. It assumes cloud top pressure of 850 hpa, but this can be replace from cloud top pressure from reanalysis
    P = 850 # P top, constant
    # checking if there are nans
    if np.isnan(T_top) or np.isnan(lwp):
        return np.nan
    if T_top < 263.13:
        return np.nan    
    T = T_top-273.13
    T_ind = int(np.round(T,1)*100)+1000 # T top, +1000 to fix the index for -10 degrees
    P_ind = int(np.round(P)-800) # P top
    # Calculating cloud base temperature and height
    # Ttop - cloud top temperature; lwp - lwp
    LWP_obs = lwp
    LWP_adi = 0
    w_temp_old = lut_w[T_ind,P_ind]*1000 # w at the layer below, necessary for the condensed water calculation
    dz_depth = 0
    #T_mat = np.zeros(100)*np.nan
    #P_mat = np.zeros(100)*np.nan
    #W_mat = np.zeros(100)*np.nan
    i=0
    
    while LWP_adi < LWP_obs:
        P_ind = P_ind+1 # converting the pressure to index (for lut)
        T_ind = int(T*100)+1000 # converting the tempertature to index (for the lut)
        
        ro_temp = lut_ro[T_ind,P_ind] # read density
        dz_temp = lut_z[T_ind,P_ind]  # read layer depth
        dz_depth += lut_z[T_ind,P_ind]  # sum the depth
        w_temp = lut_w[T_ind,P_ind]*1000  # read mixing ratio
        LWP_adi_temp = (w_temp - w_temp_old) * dz_temp * ro_temp  # calculate LWP for every leyer
        #print(LWP_adi_temp)
        LWP_adi += LWP_adi_temp # summing up the LWP of layers
        T += lut_gamma_w[T_ind,P_ind]*dz_temp # adjusting the temperature, given the saturated lapse rate
        #T_ind = int(T*100)+1000 # converting the tempertature to index (for the lut)
        #P_ind = P_ind+1 # converting the pressure to index (for lut)

        # monitoring
        #T_mat[i] = T_ind
        #P_mat[i] = P_ind
        #W_mat[i] =  w_temp
        #print LWP_adi
        #print dz_depth
        #i+=1
    return T, dz_depth # cloud depth, cloud base temperature

# Calculating cloud base temperature and height - GIVEN P (pressure)
def get_cld_H_p(T_top, lwp, p):
    # The code uses satellite obsrved cloud top temperature (kelvin) and LWP. It assumes cloud top pressure of 850 hpa, but this can be replace from cloud top pressure from reanalysis
    P = p # pressure given an input to the function
    # checking if there are nans
    if np.isnan(T_top) or np.isnan(lwp):
        return np.nan
    if T_top < 263.13:
        return np.nan    
    T = T_top-273.13
    T_ind = int(np.round(T,1)*100)+1000 # T top, +1000 to fix the index for -10 degrees
    P_ind = int(np.round(P)-800) # P top
    # Calculating cloud base temperature and height
    # Ttop - cloud top temperature; lwp - lwp
    LWP_obs = lwp
    LWP_adi = 0
    w_temp_old = lut_w[T_ind,P_ind]*1000 # w at the layer below, necessary for the condensed water calculation
    dz_depth = 0
    #T_mat = np.zeros(100)*np.nan
    #P_mat = np.zeros(100)*np.nan
    #W_mat = np.zeros(100)*np.nan
    i=0
    
    while LWP_adi < LWP_obs:
        P_ind = P_ind+1 # converting the pressure to index (for lut)
        T_ind = int(T*100)+1000 # converting the tempertature to index (for the lut)
        
        ro_temp = lut_ro[T_ind,P_ind] # read density
        dz_temp = lut_z[T_ind,P_ind]  # read layer depth
        dz_depth += lut_z[T_ind,P_ind]  # sum the depth
        w_temp = lut_w[T_ind,P_ind]*1000  # read mixing ratio
        LWP_adi_temp = (w_temp - w_temp_old) * dz_temp * ro_temp  # calculate LWP for every leyer
        #print(LWP_adi_temp)
        LWP_adi += LWP_adi_temp # summing up the LWP of layers
        T += lut_gamma_w[T_ind,P_ind]*dz_temp # adjusting the temperature, given the saturated lapse rate
        #T_ind = int(T*100)+1000 # converting the tempertature to index (for the lut)
        #P_ind = P_ind+1 # converting the pressure to index (for lut)

        # monitoring
        #T_mat[i] = T_ind
        #P_mat[i] = P_ind
        #W_mat[i] =  w_temp
        #print LWP_adi
        #print dz_depth
        #i+=1
    return T, dz_depth # cloud depth, cloud base temperature

# Calculating LTS - Lower-troposphere stability
def get_LTS(temperature_track_calipso):
    R_cp = 0.286
    theta_700 = temperature_track_calipso[0] * (1000/700.)**R_cp
    theta_1000 = temperature_track_calipso[1] * (1000/1000.)**R_cp
    LTS = theta_700 - theta_1000
    return LTS

def read_processed_data(f_name, var_list):
    # extract the processed coupling/cloud base data
    # Input: directory of the file, file name and variable list to be extracted
    # Output: A dictionary with the variables
    output_data = {}
    nc_h = nc4.Dataset(f_name)
    for var in var_list:
        output_data[var] = nc_h[var][:]
    return output_data


def save_to_netCDF(product_vars):
    # Saves the variables into a netCDF file_name
    print('save to NetCDF')
    f_name_nc = '/media/tgoren/WD_2TB/tgoren/data/calmod/statistics/'  + \
                str(product_vars['case_name']) + '_calmod.nc'
    calmod_product = nc4.Dataset(f_name_nc, 'w')

    # Global Attributes
    calmod_product.description = 'Coupling and cloud base product (A-Train)'
    import time
    calmod_product.history = 'Created ' + time.ctime(time.time())
    calmod_product.source = 'T. Goren, Leipzig University'
    calmod_product.summary =  product_vars['case_name']
    # Create dimensions
    dim = int(np.array(product_vars['lat'].shape)[0])
    length = calmod_product.createDimension('length', dim)
    case_name = calmod_product.createDimension('case_name', 1)
    
    # Create variables
    #file_name= calmod_product.createVariable('file_name', str)
    #file_name.description = 'file_name'  # Variable attribute
    #file_name = product_vars['file_name']
    
    lat = calmod_product.createVariable('lat', np.float64, ('length'))
    lat.units = 'Degree_north'
    lat.description  = 'Latitude south is negative'
    lat[:] =  product_vars['lat']
    
    lon = calmod_product.createVariable('lon', np.float64, ('length'))
    lon.units = 'Degree_east'
    lon.description = 'Longitude west is negative'  # Variable attribute
    lon[:] = product_vars['lon'] 

    coupling_index = calmod_product.createVariable('coupling_index', np.float64, ('length'))
    #lon.units = ''
    coupling_index.description = 'negative is coupled'  # Variable attribute
    coupling_index[:] = product_vars['coupling_index']

    coupling_index_no_cld = calmod_product.createVariable('coupling_index_no_cld', np.float64, ('length'))
    #lon.units = ''
    coupling_index_no_cld.description = 'negative is coupled'  # Variable attribute
    coupling_index_no_cld[:] = product_vars['coupling_index_no_cld']
    
    cloud_top_height = calmod_product.createVariable('cloud_top_height', np.float64, ('length'))
    cloud_top_height.units = 'km'
    cloud_top_height.description = 'cloud_top_height'  # Variable attribute
    cloud_top_height[:] = product_vars['cloud_top_height']     

    cloud_base_height = calmod_product.createVariable('cloud_base_height', np.float64, ('length'))
    cloud_base_height.units = 'km'
    cloud_base_height.description = 'cloud_base_height'  # Variable attribute
    cloud_base_height[:] = product_vars['cloud_base_height']

    calipso_cloud_base_height = calmod_product.createVariable('calipso_cloud_base_height', np.float64, ('length'))
    calipso_cloud_base_height.units = 'meters'
    calipso_cloud_base_height.description = 'calipso_cloud_base_height'  # Variable attribute
    calipso_cloud_base_height[:] = product_vars['calipso_cloud_base_height']

    lwp_avg= calmod_product.createVariable('lwp_avg', np.float64, ('length'))
    lwp_avg.units = 'gr/m^2'
    lwp_avg.description = 'mean lwp for a 10x10 pixels for the highest 15th percentile'  # Variable attribute
    lwp_avg[:] = product_vars['lwp_avg']

    lwp_avg_all = calmod_product.createVariable('lwp_avg_all', np.float64, ('length'))
    lwp_avg_all.units = 'gr/m^2'
    lwp_avg_all.description = 'mean lwp for a 10x10 pixels for all pixels'  # Variable attribute
    lwp_avg_all[:] = product_vars['lwp_avg_all']
    
    re_avg= calmod_product.createVariable('re_avg', np.float64, ('length'))
    re_avg.units = 'mu'
    re_avg.description = 'mean re for 10x10 pixels for the highest 15th percentile'  # Variable attribute
    re_avg[:] = product_vars['re_avg']

    tau_avg= calmod_product.createVariable('tau_avg', np.float64, ('length'))
    tau_avg.units = ''
    tau_avg.description = 'mean optical thickness for 10x10 pixels for the highest 15th percentile'  # Variable attribute
    tau_avg[:] = product_vars['tau_avg']

    cloud_top_temperature_avg= calmod_product.createVariable('cloud_top_temperature_avg', np.float64, ('length'))
    cloud_top_temperature_avg.units = 'K'
    cloud_top_temperature_avg.description = 'mean cloud top temperature for 10x10 pixels for the highest 15th percentile'  # Variable attribute
    cloud_top_temperature_avg[:] = product_vars['cloud_top_temperature_avg']

    nd_avg= calmod_product.createVariable('nd_avg', np.float64, ('length'))
    nd_avg.units = '#/m^3 at cloud top'
    nd_avg.description = 'mean CDNC for 10x10 pixels for the highest 15th percentile'  # Variable attribute
    nd_avg[:] = product_vars['nd_avg']

    R06_avg= calmod_product.createVariable('R06_avg', np.float64, ('length'))
    R06_avg.units = '0.6 reflectance'
    R06_avg.description = 'scene mean reflectance'  # Variable attribute
    R06_avg[:] = product_vars['R06_avg']

    SST= calmod_product.createVariable('SST', np.float64, ('length'))
    SST.units = 'C'
    SST.description = 'SST'  # Variable attribute
    SST[:] = product_vars['SST']

    wind_speed= calmod_product.createVariable('wind_speed', np.float64, ('length'))
    wind_speed.units = 'm/s'
    wind_speed.description = 'wind_speed'  # Variable attribute
    wind_speed[:] = product_vars['wind_speed']

    LTS = calmod_product.createVariable('LTS', np.float64, ('length'))
    LTS.units = 'K'
    LTS.description = 'Lower Tropospheric Stability'  # Variable attribute
    LTS[:] = product_vars['LTS']

    divergence = calmod_product.createVariable('divergence', np.float64, ('length'))
    divergence.units = 'S-1'
    divergence.description = 'Averaged profile 1000-900 hpa'  # Variable attribute
    divergence[:] = product_vars['divergence']

    precip_w = calmod_product.createVariable('precipitable_water', np.float64, ('length'))
    precip_w.units = 'mm'
    precip_w.description = 'Precipitable water 500-800 hpa'  # Variable attribute
    precip_w[:] = product_vars['precip_w']
    
    csat_max_ref= calmod_product.createVariable('csat_max_ref', np.float64, ('length'))
    csat_max_ref.units = 'dbZ'
    csat_max_ref.description = 'CloudSat maximun reflectivity within a column below 2.5km. The lowest most level are ignored because of ground clutter'  # Variable attribute
    csat_max_ref[:] = product_vars['csat_max_ref']

    CF = calmod_product.createVariable('CF', np.float64, ('length'))
    #CF.units = 'CF'
    CF.description = 'cloud fraction'  # Variable attribute
    CF[:] = product_vars['CF']

    MODIS_CF = calmod_product.createVariable('MODIS_CF', np.float64, ('length'))
    #CF.units = 'CF'
    MODIS_CF.description = 'MODIS cloud fraction'  # Variable attribute
    MODIS_CF[:] = product_vars['MODIS_CF']

    MODIS_CF_mask = calmod_product.createVariable('MODIS_CF_mask', np.float64, ('length'))
    #CF.units = 'CF'
    MODIS_CF_mask.description = 'MODIS cloud fraction mask'  # Variable attribute
    MODIS_CF_mask[:] = product_vars['MODIS_CF_mask']
    
    case_name= calmod_product.createVariable('case_name',  np.float64, ('length'))
    #.units = ''
    case_name.description = 'case_date'  # Variable attribute
    case_name[:] = product_vars['case_name_vector']
    
    '''
    = calmod_product.createVariable('', np.float64, ('length'))
    .units = ''
    .description = ''  # Variable attribute
    [:] = product_vars['']
    '''

    calmod_product.close()

# Plot 6 panels (with SST)
def plot_sst(f_name_modis_short,y_axis_alt_km,calipso_alt_mbl,calipso_top_alt_swath,coupling_index,coupled, decoupled,cldsat_reflectivity_swath,cldsat_height_swath_col, cldsat_height_swath_col_ind, cld_base_H, cld_base_H_1px_run_mean, modis_swath_track,  modis_swath_width, sst_swath_area, v_wnd_track_calipso_short,u_wnd_track_calipso_short, calipso_lat_swath, cldsat_alt_mbl_length, calipso_base_alt_swath,wnd_speed_swath_ln,wind_vectors_interval,calipso_lon_swath):

    height_max = y_axis_alt_km + 0.5  # so the upper ytick will be shown
    height_min = 0
    height_ticks = np.arange(height_min, height_max, 0.5)
    height_ticks_inds = np.digitize(height_ticks, calipso_alt_mbl[::-1])
    x = np.arange(0, np.size(calipso_top_alt_swath), 1) # array for the x axis
    y = np.zeros(np.size(calipso_top_alt_swath)) # array for the y axis
    coupling_index[coupling_index<-5] = np.nan
    running_mean_coupling_index = misc_codes.running_mean(coupling_index, 50)
    
    # Coupling
    fig = plt.figure()
    ax1 = plt.subplot(6, 1, 1)    
    ax1_1 = ax1.scatter(x, coupled, color= 'blue', edgecolors='blue', s=1) # +1 so that the points will be above the zero line (for the coupling threshold of 1 degree
    ax1_1 = ax1.scatter(x, decoupled, color = 'red', edgecolors='red', s=1)
    ax1.set_xlim(0,coupled.shape[0])
    ax1.set_ylim([-6, 6])
    # add line of the A-train track
    plt.hold(True)
    # Drawing a line at heigh 0
    ax1.plot(x, y, '-k', lw=2)
    ax1.xaxis.set_visible(False)
    #ax1.plot(running_mean_coupling_index, color='k')
    ax1.set_title('Calculated air-sea temperature difference (red-decoupled, blue-coupled)',fontsize=20)
    #ax1.set_position([0.125, 0.762, 0.9, 0.9])
    
    # Cloud Sat
    ax3 = plt.subplot(6, 1, 2)
    ax4 = ax3.twinx()
    cmap_cs = plt.get_cmap('jet') # Color map
    cmap_cs.set_bad('k')  # setting the nan values to black
    swath_l_ax3 = calipso_top_alt_swath.shape[0] # length of x axis
    ax3_3 = ax3.imshow(np.rot90(cldsat_reflectivity_swath), origin='lower', aspect='auto', cmap=cmap_cs, interpolation='None', extent=[0, swath_l_ax3,  0, cldsat_alt_mbl_length], vmax=20, vmin=-40)
    ax3.get_yticks()
    cldsat_ytick_labels = np.squeeze(cldsat_height_swath_col[cldsat_height_swath_col_ind])

    # Organize the Y axis - the first 3 line are already executed in the calipso part
    # height_max = y_axis_alt_km + 0.5 # so the upper ytick will be shown
    height_ticks_inds = np.digitize(height_ticks * 1000, cldsat_height_swath_col[
        cldsat_height_swath_col_ind][::-1])  # Height of each layer in cldsat
    ax3.get_yticks()
    ax3.set_yticks(height_ticks_inds)
    ax3.set_yticklabels(height_ticks)
    # Color bar location [left, bottom, width, height] in precentage
    cbaxes = fig.add_axes([0.905, 0.648, 0.01, 0.114])
    cbar3 = plt.colorbar(ax3_3, cax=cbaxes)
    ax3.xaxis.set_visible(False)
    # Add cloud top, base and depth
    plt.hold(True)
    x = np.arange(swath_l_ax3)
    ax4_4 = ax4.plot(x, calipso_top_alt_swath * 1000, 'ko', markersize=2) # cloud top
    ax4_4 = ax4.plot(x, cld_base_H*1000, 'ko', markersize=2) # cloud base
    ax4_4 = ax4.plot(x, cld_base_H_1px_run_mean*1000, color='0.75', linewidth=2) # cloud base
    #ax4_4 = ax4.plot(x, cld_D+1000, 'w', markersize=0.1) # cloud depth
    #ax4_4 = ax4.plot(x, layer_base_alt_swath[
    #    :, 0] * 1000, 'w', markersize=0.1)

    # Add cloud base from calipso
    ax4_4 = ax4.plot(x, calipso_base_alt_swath*1000, 'w', linewidth=2) # cloud base

    ax4.set_ylim([cldsat_ytick_labels[-1], cldsat_ytick_labels[0]])
    ax4.get_yaxis().set_ticks([])
    ax4.set_xlim([0, swath_l_ax3])
    ax3.set_title('CloudSat reflectivity [dBZ]',fontsize=20)
    print(1)
    # Modis -re
    ax5 = plt.subplot(6, 1, 3)
    ax5_5 = ax5.imshow(modis_swath_track['Cloud_Effective_Radius'], aspect='auto', vmin=0, vmax=30)
    # add line of the A-train track
    plt.hold(True)
    ax5.plot([0, modis_swath_track['Cloud_Effective_Radius'].shape[1]], [
        modis_swath_width / 2, modis_swath_width / 2], '-k', lw=2)
    ax5.set_xlim([0, modis_swath_track['Cloud_Effective_Radius'].shape[1]])
    ax5.set_ylim([0, modis_swath_width])
    #div = make_axes_locatable(ax5)
    #cax = div.append_axes("right", size="1%", pad=0.05)
    # [left, bottom, width, height]
    cbaxes = fig.add_axes([0.905, 0.511, 0.01, 0.114]) # [left, bottom, width, height],
    cbar5 = plt.colorbar(ax5_5, cax=cbaxes)
    ax5.xaxis.set_visible(False)
    ax5.yaxis.set_visible(False)
    ax5.set_title('MODIS Effective Radius [${\mu}$m]',fontsize=20)
    
    # Modis - lwp
    modis_swath_track['Cloud_Water_Path'][modis_swath_track['Cloud_Water_Path']>500] = 500
    #modis_cwp_swath[modis_cwp_swath>600] = np.nan
    ax6 = plt.subplot(6, 1, 4)
    ax6_6 = ax6.imshow(modis_swath_track['Cloud_Water_Path'], aspect='auto', interpolation='nearest', vmin=0, vmax=500)
    # add line of the A-train track
    plt.hold(True)
    ax6.plot([0, modis_swath_track['Cloud_Water_Path'].shape[1]], [
        modis_swath_width / 2, modis_swath_width / 2], '-k', lw=2)
    ax6.set_xlim([0, modis_swath_track['Cloud_Water_Path'].shape[1]])
    ax6.set_ylim([0, modis_swath_width])
    #div = make_axes_locatable(ax5)
    #cax = div.append_axes("right", size="1%", pad=0.05)
    # [left, bottom, width, height]
    cbaxes = fig.add_axes([0.905, 0.374, 0.01, 0.114])
    cbar6 = plt.colorbar(ax6_6, cax=cbaxes)
    ax6.xaxis.set_visible(False)
    ax6.yaxis.set_visible(False)
    ax6.set_title('MODIS Liquid Water Path [$\mathregular{gr/m^{2}}$]',fontsize=20)
    fig_name = f_name_modis_short[10:]
    
    # SST and wind vectors
    ax7 = plt.subplot(6, 1, 5)
    #ax7.set_xlim([0, sst_swath_area.shape[0]])        
    #ax7_7 = ax7.imshow(np.rot90(sst_swath_area,-1)-273, origin='lower', aspect='auto', extent=[0, swath_l_ax3,  0, 300])#sst_swath_area.shape[0]])
    ax7_7 = ax7.imshow(np.rot90(sst_swath_area,-1)-273, aspect='auto', extent=[0, swath_l_ax3,  0, 300],interpolation='nearest')
    plt.hold(True)
    plt.quiver(np.arange(0,wnd_speed_swath_ln,wind_vectors_interval),150, v_wnd_track_calipso_short,u_wnd_track_calipso_short*(-1),scale=0.1, units='y', width=6, headwidth=6) #scale = 2, units='width', width=20, headwidth=1) # Wind vectors
    cbaxes = fig.add_axes([0.905, 0.237, 0.01, 0.114])
    cbar7 = plt.colorbar(ax7_7, cax=cbaxes)
    ax7.xaxis.set_visible(False)
    ax7.yaxis.set_visible(False)
    #ax7.set_ylim([-5, 30])
    ax7.set_title('Sea Surface Temperature [C] and wind vectors',fontsize=20)
    fig_name = f_name_modis_short[10:]
    
    # Modis - reflectance
    ax8 = plt.subplot(6, 1, 6)
    cmap = plt.get_cmap('Greys_r')
    cmap.set_bad(color='k')
    ax8_8 = ax8.imshow(modis_swath_track['Atm_Corr_Refl'], cmap=cmap, aspect='auto')
    # adding line of the A-train track
    plt.hold(True)
    ax8.plot([0, modis_swath_track['Atm_Corr_Refl'].shape[1]], [
        modis_swath_width / 2, modis_swath_width / 2], '-k', lw=2)
    ax8.set_xlim([0, modis_swath_track['Atm_Corr_Refl'].shape[1]])
    ax8.set_ylim([0, modis_swath_width])
    cbaxes = fig.add_axes([0.905, 0.10, 0.01, 0.114]) # location of the subfigure
    cbar8 = plt.colorbar(ax8_8, cax=cbaxes)
    cmap = plt.get_cmap
    ax8.yaxis.set_visible(False)
    ax8.set_title('MODIS 0.86 Reflectance',fontsize=20)
    # Organizing the x axis - using data of lat/lon from CALIPSO (ax1)
    lat_min_xaxis = np.min(calipso_lat_swath)  # to round up
    lat_max_xaxis = np.max(calipso_lat_swath)
    ax1_xtick_labels = np.array([np.arange(lat_min_xaxis, lat_max_xaxis, 2)])[0]
    xtick_labels_inds = np.digitize(ax1_xtick_labels, calipso_lat_swath)
    ax1_lon_xtick_labels = np.around(calipso_lon_swath[xtick_labels_inds], decimals=1) -360 # xlabels - lon
    x_join_lable = zip(ax1_xtick_labels,ax1_lon_xtick_labels) # This is the joined x axes labels - lat and lon
    x_join_lable = map(lambda f:'{:.1f}\n{:.1f}'.format(f[0],f[1]), x_join_lable)
    xtick_labels_inds_ax6 = np.digitize(ax1_xtick_labels, calipso_lat_swath)
    ax1.get_xticks()
    ax8.set_xticks(xtick_labels_inds_ax6)
    ax8.set_xticklabels(x_join_lable, fontsize=20)# ax1_xtick_labels)

    #plt.tight_layout()
    fig.set_size_inches(25,16)
    fig.savefig(fig_name + '_coupled_sst'+'.pdf')#, bbox_inches='tight')
    
