## Plotting the collocated A-Train data on a 4 panel plot
# There are 5 input files: (1) MODIS cloud product
# collection 06 (2) CALIPSO L2 (3) CloudSat (4) ECMWF sst

import numpy as np
import scipy
import netCDF4 as nc4
import h5py
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import hdf
import os
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import spatial
from scipy import stats 
from csat import MODIS
import datetime
import sys
import glob
import calmod_calipso_alt
import read_data
import calmod_support
import misc_codes 

def calmod_auto_run(flag, lat_low_lim=None, lot_high_lim=None, f_name_modis_2=None, f_name_modis=None, fname_calipso_l2=None, fname_cldsat=None):
    # Flag =1 : Automatic process (all files of all cases in one directory
    # Flag =2 : Process directory
    # Flag =3 : Merge swaths
    # Flag =4 : lat/lon limits

    print ('running calmod_v18')
    #folder = sys.argv[1] # read the first argument that is given in the command line while executing the script

    modis_swath_width = 300  # i.e. the width of modis along the A-Train track - I think 300 is a fair band to correlate spatially cloud properties
    y_axis_alt_km = 2.5
    y_axis_alt_m = y_axis_alt_km * 1000

    if flag == 1:
        print ('running all-cases-in-one-directory-mode')        
        #print (f_name_modis, fname_calipso_l2, fname_cldsat)
        # For automatoc run_calmod
        ##  Get file names  ##
        f_name_modis_short = f_name_modis[0:22]
        # Becauce ecmwf file in a different directory
        directory_ecmwf = '/home_local/tgoren/research/analysis/calmod/case_analysis/data/ecmwf/' #'/media/tgoren/intenso02/tgoren/data/calmod/ecmwf/'
        year = f_name_modis_short[10:14] #   f_name_modis.split('/')[-3]
        f_ecmwf_list = os.listdir(directory_ecmwf)
        fname_ecmwf_temp_surface = 'ecmwf_' + year + '_surface.nc'
        #fname_ecmwf_temp_pressure = 'ecmwf_' + year + '_pressure.nc'
        fname_ecmwf_surface = directory_ecmwf + fname_ecmwf_temp_surface
        #fname_ecmwf_pressure  = directory_ecmwf + fname_ecmwf_temp_pressure
        
        else:
            #if flag == 2 or flag == 3 or flag == 4:
            # For folder case by case run (2); merge swaths (3); lat/lon limits (4)
            print ('Running on case-folder-mode')
            # Read files        
            f_initials = ['MYD06_L2', 'GEOPROF', 'CAL_LID_L2_']
            f_ind = np.zeros(np.shape(f_initials), dtype=int)
            f_list = glob.glob("*.hdf") #os.listdir("*.hdf")
            for l, j in enumerate(f_initials):
                f_ind[l] = np.array([i for i, s in enumerate(f_list) if j in s])
                print (j)

            ##  Get file names  ##
            directory = os.getcwd()
            fname_cldsat = (directory + '/' + f_list[f_ind[1]])
            #fname_calipso_l1 = (directory + '/' + f_list[f_ind[2]])
            fname_calipso_l2 = (directory + '/' + f_list[f_ind[2]])
            f_name_modis = (directory + '/' + f_list[f_ind[0]])
            f_name_modis_short = f_name_modis.split('/')[-1][0:22]
            # Becauce the ecmwf file in a different directory
            directory_ecmwf = '/home/tgoren/research/analysis/calmod/ECMWF/' #'/media/tgoren/intenso02/tgoren/data/calmod/ecmwf/'
            year = f_name_modis_short[10:14] #   f_name_modis.split('/')[-3]
            f_ecmwf_list = os.listdir(directory_ecmwf)
            fname_ecmwf_temp_surface = 'ecmwf_' + year + '_surface.nc'
            #fname_ecmwf_temp_pressure = 'ecmwf_' + year + '_pressure.nc'
            fname_ecmwf_surface = directory_ecmwf + fname_ecmwf_temp_surface
            #fname_ecmwf_pressure  = directory_ecmwf + fname_ecmwf_temp_pressure


    ## Read files ##
    # modis
    modis_vars = ['Cloud_Effective_Radius', 'Cloud_Water_Path', 'Cloud_Optical_Thickness',\
                  'cloud_top_temperature_1km', 'Cloud_Multi_Layer_Flag', \
                  'Cloud_Water_Path_Uncertainty','Cloud_Effective_Radius_16',\
                  'Cloud_Optical_Thickness_Uncertainty', 'Cloud_Effective_Radius_Uncertainty',\
                  'Atm_Corr_Refl', 'Cloud_Fraction', 'Latitude', 'Longitude',\
                  'Cloud_Mask_1km', 'Retrieval_Failure_Metric','Cloud_Mask_1km']
    modis_data = read_data.read_modis(f_name_modis, modis_vars)

    if flag == 3:
        # Merging swaths; Input: modis_data_file_2
        # modis - FILE 2
        modis_vars = ['Cloud_Effective_Radius', 'Cloud_Water_Path', 'Cloud_Optical_Thickness',\
              'cloud_top_temperature_1km', 'Cloud_Multi_Layer_Flag', \
              'Cloud_Water_Path_Uncertainty','Cloud_Effective_Radius_16',\
              'Cloud_Optical_Thickness_Uncertainty', 'Cloud_Effective_Radius_Uncertainty',\
                      'Atm_Corr_Refl', 'Cloud_Fraction','Latitude', 'Longitude',\
                      'Cloud_Mask_1km', 'Retrieval_Failure_Metric','Cloud_Mask_1km']
        modis_data_file_2 = read_data.read_modis(f_name_modis_2, modis_vars)
        # Merging the 2 MODIS dictionaries
        modis_data_2 = {}
        for var in modis_vars:
            modis_data_2[var] = np.append(modis_data[var], modis_data_file_2[var],axis=0)
        modis_data = copy.copy(modis_data_2)
        
    # get calipso level 1
    #calipso_l1_vars = ['Latitude', 'Longitude', 'Total_Attenuated_Backscatter_532']
    #calipso_data_l1 = read_calipso_l1(calipso_l1_vars)

    # Calipso level 2
    calipso_l2_vars = ['Latitude','Longitude','Layer_Top_Altitude','Layer_Base_Altitude', 'Layer_Top_Temperature']
    calipso_data_l2 = read_data.read_calipso_l2(fname_calipso_l2, calipso_l2_vars)

    # Cldsat
    cldsat_vars = ['Latitude', 'Longitude', 'Radar_Reflectivity', 'Height', 'Profile_time', 'UTC_start']
    data_cldsat = read_data.read_cldsat(fname_cldsat,cldsat_vars)

    # ECMWF surface - SST and winds
    ecmwf_surface_vars = ['time', 'longitude', 'latitude','sst', 'u10','v10']
    ecmwf_surface_data = read_data.read_ecmwf(fname_ecmwf_surface, ecmwf_surface_vars)

    # ECMWF surface - pressure level data (T, qs)
    #ecmwf_pressure_vars = ['time', 't', 'q']
    #ecmwf_pressure_data = read_data.read_ecmwf(fname_ecmwf_pressure, ecmwf_pressure_vars)


    # Interpolated field - MODIS
    modis_lat = MODIS.field_interpolate(modis_data['Latitude'], 5)
    modis_lon = MODIS.field_interpolate(modis_data['Longitude'], 5)
    modis_cf_product_regridded = MODIS.field_interpolate(modis_data['Cloud_Fraction'], 5)
    '''
    ## Calipso level 1  ##
    atten_532 = calipso_data_l1['Total_Attenuated_Backscatter_532']
    atten_532[np.where(atten_532 < -1000)] = np.nan
    calipso_lat_l1 = np.squeeze(calipso_data_l1['Latitude'])
    calipso_lon_l1 = np.squeeze(calipso_data_l1['Longitude'])
    '''

    ## Calipso level 2  ##
    calipso_lat_l2 = np.squeeze(calipso_data_l2['Latitude'])
    calipso_lon_l2 = np.squeeze(calipso_data_l2['Longitude'])
    layer_top_alt = calipso_data_l2['Layer_Top_Altitude'][:,0]
    layer_top_alt[np.where(layer_top_alt < 0)] = np.nan
    layer_top_alt[np.where(layer_top_alt >= 3)] = np.nan # Limit to clouds below 3 km
    layer_top_alt[np.where(layer_top_alt == -9999)] = np.nan
    layer_base_alt =  calipso_data_l2['Layer_Base_Altitude'][:,0]
    layer_base_alt[np.where(layer_base_alt <= 0)] = np.nan
    layer_base_alt[np.where(layer_base_alt > 3)] = np.nan # Limit to clouds below 3 km
    layer_base_alt[np.where(layer_base_alt == -9999)] = np.nan
    layer_top_temperature = calipso_data_l2['Layer_Top_Temperature'][:,0]
    layer_top_temperature[np.where(layer_top_temperature == -9999)] = np.nan

    ##  Co-location CALIPSO-MODIS  ##
    # Finding max/min of the modis swath
    if flag == 3 or flag == 4:
        # Merge swaths
        modis_lat_max = lot_high_lim
        modis_lat_min = lat_low_lim
    else:
        modis_lat_max = np.max(modis_lat) # lat_high_lim # these are not accurate lat (i.e., not in the middle of the swath) !!
        modis_lat_min = np.min(modis_lat)# lat_low_lim
    modis_lon_max = np.max(modis_lon)
    modis_lon_min = np.min(modis_lon)

    # calipso L2 pixels that match those of MODIS min/max
    calipso_lat_swath_ind_extra = np.where(np.logical_and(calipso_lat_l2 < modis_lat_max, calipso_lat_l2 > modis_lat_min))[0]  # extra - because it goes from the max/min of the modis swath (and not in the middle)
    calipso_lat_swath_extra = calipso_lat_l2[calipso_lat_swath_ind_extra]
    calipso_lon_swath_extra = calipso_lon_l2[calipso_lat_swath_ind_extra]
    calipso_top_alt_swath_extra = layer_top_alt[calipso_lat_swath_ind_extra]
    calipso_base_alt_swath_extra = layer_base_alt[calipso_lat_swath_ind_extra]
    calipso_top_temperature_swath_extra = layer_top_temperature[calipso_lat_swath_ind_extra]
    
    #  KDtree
    modis_lat_lon = zip(modis_lat.ravel(), modis_lon.ravel())
    modis_latlon_kd = spatial.cKDTree(list(modis_lat_lon))
    # Finding the MODIS indeces that match calipso
    # quering the indeces of the nearest points
    ind_mod_cal_colloc = modis_latlon_kd.query(
        list(zip(calipso_lat_swath_extra, calipso_lon_swath_extra)))  # MODIS indices of the pixels that within the middle of the swath (this is needed because I cut the calipso data based on the min and max of modis). The length is the same as the calipso input - one modis index per calipso index
    cal_swath_ind = np.where(ind_mod_cal_colloc[0][:] < 0.01)[0] # MODIS indices that match CALIPSO (distance limit) - this reduces the size because now it is the most accurate collocation

    # Converting index to line/col
    mod_ind_row = ind_mod_cal_colloc[1][cal_swath_ind] // 1350 # Converting index to line/col 1350 is MODIS swath number of pixels
    mod_ind_col = ind_mod_cal_colloc[1][cal_swath_ind] % 1350 # Converting index to line/col
    #modis_track_indices_1km = zip(mod_ind_row, mod_ind_col) # CALIPSO resolution

    ## Get averaged calipso data for modis resolution ##
    mod_ind_row_unq = np.unique(mod_ind_row)
    mod_ind_col_unq = mod_ind_col[np.unique(mod_ind_row, return_index=True)[1]] # get unique modis col 
    modis_track_row_col_indices_1km = list(zip(mod_ind_row_unq,mod_ind_col_unq)) # MODIS resolution

    calipso_lat_swath = np.zeros(np.shape(mod_ind_row_unq)[0])*np.nan
    calipso_lon_swath = np.zeros(np.shape(mod_ind_row_unq)[0])*np.nan
    calipso_top_alt_swath = np.zeros(np.shape(mod_ind_row_unq)[0])*np.nan
    calipso_base_alt_swath = np.zeros(np.shape(mod_ind_row_unq)[0])*np.nan
    calipso_top_temperature_swath = np.zeros(np.shape(mod_ind_row_unq)[0])*np.nan
    for i,j in enumerate(mod_ind_row_unq):
        calipso_lat_swath[i] = np.nanmean(calipso_lat_swath_extra[cal_swath_ind[np.where(mod_ind_row == j)]])
        calipso_lon_swath[i] = np.nanmean(calipso_lon_swath_extra[cal_swath_ind[np.where(mod_ind_row == j)]])
        calipso_top_alt_swath[i] = np.nanmean(calipso_top_alt_swath_extra[cal_swath_ind[np.where(mod_ind_row == j)]])
        calipso_base_alt_swath[i] = np.nanmean(calipso_base_alt_swath_extra[cal_swath_ind[np.where(mod_ind_row == j)]])
        calipso_top_temperature_swath[i] = np.nanmean(calipso_top_temperature_swath_extra[cal_swath_ind[np.where(mod_ind_row == j)]])

    ##  Get accurate modis lat/lon  min/max  ##
    modis_lat_track = modis_lat[mod_ind_row_unq, mod_ind_col_unq]
    modis_lon_track = modis_lon[mod_ind_row_unq, mod_ind_col_unq]
    modis_lat_min_accurate = np.min(modis_lat_track)
    modis_lat_max_accurate = np.max(modis_lat_track)

    # Find the calipso indicies that are within the modis swath
    layer_top_alt = layer_top_alt[calipso_lat_swath_ind_extra[cal_swath_ind]]

    # Get calipso altitude indicies fot the MBL
    # call the altitude table from "calmod_support" package
    calipso_alt = calmod_calipso_alt.get_calipso_alt()
    calipso_alt_mbl_ind = np.where(
        np.logical_and(calipso_alt > 0, calipso_alt < y_axis_alt_km))
    calipso_alt_mbl = np.array(calipso_alt[calipso_alt_mbl_ind])
    calipso_alt_mbl_length = calipso_alt_mbl.size


    ##  Adding X pixels to each side along the track and extracting the along track line data  ##
    modis_vars_swath = ['Cloud_Effective_Radius', 'Cloud_Water_Path','Cloud_Multi_Layer_Flag',
                        'Cloud_Optical_Thickness_Uncertainty', 'Cloud_Water_Path_Uncertainty',\
                        'Cloud_Effective_Radius_Uncertainty', 'Cloud_Optical_Thickness',\
                        'cloud_top_temperature_1km','Cloud_Effective_Radius_16']  # I excluded latitude because it has different resolution
    modis_swath_track = {}  # create an empty dictionary for the wide track (+buffer)
    modis_swath_track_line = {}  # create an empty dictionary for the wideline along the track
    for var in modis_vars_swath:
        modis_swath_track[var] = np.zeros(
            (modis_swath_width, len(mod_ind_row_unq)))*np.nan
        modis_swath_track_line[var] = modis_data[var][mod_ind_row_unq, mod_ind_col_unq]
        for offset in range(0, modis_swath_width, 1):
            modis_swath_track[var][offset] = modis_data[var][
                mod_ind_row_unq, mod_ind_col_unq + (offset - modis_swath_width // 2)]

    # Get modis reflectance at 0.86 - different dimentions than the cloud microphysical variables
    modis_ref_086_swath = modis_data['Atm_Corr_Refl'][:, :, 0]#-modis_data['Atm_Corr_Refl'][:, :, 4]
    modis_swath_track['Atm_Corr_Refl'] = np.zeros(
        (modis_swath_width, len(mod_ind_row_unq)))*np.nan
    for offset in range(0, modis_swath_width, 1):
        modis_swath_track['Atm_Corr_Refl'][offset] = modis_ref_086_swath[mod_ind_row_unq, mod_ind_col_unq + (offset - modis_swath_width // 2)]
        modis_swath_track_line['Atm_Corr_Refl'] = modis_data['Atm_Corr_Refl'][:, :, 1][mod_ind_row_unq, mod_ind_col_unq]
        # mask the nan's. Then in the plotting these are "bad" and will be displayed as black
        modis_swath_track['Atm_Corr_Refl'] = np.ma.array(
            modis_swath_track['Atm_Corr_Refl'], mask=np.isnan(modis_swath_track['Atm_Corr_Refl']))
    
    # Using the failed retrievals - because they are actually correct for these clouds ONLY! Proof is given to reviewers in the manuscript
    failed_tau_metric = modis_data['Retrieval_Failure_Metric'][:, :, 0]
    modis_swath_track['Retrieval_Failure_Metric'] = np.zeros(
        (modis_swath_width, len(mod_ind_row_unq)))*np.nan
    for offset in range(0, modis_swath_width, 1):
        modis_swath_track['Retrieval_Failure_Metric'][offset] = failed_tau_metric[mod_ind_row_unq, mod_ind_col_unq + (offset - modis_swath_width // 2)]
        #modis_swath_track_line['Cloud_Mask_1km'] = modis_data['Cloud_Mask_1km'][:, :, 1][mod_ind_row_unq, mod_ind_col_unq]
    
    # CalculatING Nd
    modis_data['nd'] = read_data.calculate_Nd(modis_data['Cloud_Effective_Radius'], modis_data['Cloud_Optical_Thickness'])
    modis_swath_track['nd'] = read_data.calculate_Nd(modis_swath_track['Cloud_Effective_Radius'],         modis_swath_track['Cloud_Optical_Thickness'])
    modis_swath_track_line['nd'] = read_data.calculate_Nd (modis_swath_track_line['Cloud_Effective_Radius'], modis_swath_track_line['Cloud_Optical_Thickness'])

    ##  Cldsat  ##
    cldsat_lat = data_cldsat['Latitude']
    cldsat_lon = data_cldsat['Longitude']
    cldsat_reflectivity = data_cldsat['Radar_Reflectivity']
    # Scaling Factor that was not applied automatically)
    cldsat_reflectivity = cldsat_reflectivity / 100
    cldsat_reflectivity[cldsat_reflectivity < -40] = np.nan
    cldsat_height = data_cldsat['Height']
    # Find the Cldsat indicies that are within the modis swath
    cldsat_lat_range = np.where(np.logical_and(
        cldsat_lat >= modis_lat_min_accurate, cldsat_lat <= modis_lat_max_accurate))[0]
    cldsat_lat_extra = cldsat_lat[cldsat_lat_range]
    # cldsat_reflectivity_swath = cldsat_reflectivity[cldsat_lat_range]
    cldsat_lon_range = np.where(np.logical_and(
        cldsat_lon >= modis_lon_min, cldsat_lon <= modis_lon_max))[0]
    cldsat_lon_extra = cldsat_lon[cldsat_lon_range]
    cldsat_reflectivity_swath = cldsat_reflectivity[np.intersect1d(
        cldsat_lat_range, cldsat_lon_range, assume_unique=True)]
    # get cldsat altitude indicies fot the MBL
    cldsat_height_swath = cldsat_height[cldsat_lat_range]
    # Get one column of the height vector (the height is not constant)
    cldsat_height_swath_col = cldsat_height_swath[1]
    cldsat_height_swath_col_ind = np.array(np.where(np.logical_and(
        cldsat_height_swath_col >= 0, cldsat_height_swath_col < y_axis_alt_m)))[0]
    cldsat_alt_mbl_length = cldsat_height_swath_col_ind.size
    cldsat_reflectivity_swath = np.squeeze(
        cldsat_reflectivity_swath[:, cldsat_height_swath_col_ind])

    csat_to_mod_ind = np.round(np.linspace(0,cldsat_reflectivity_swath.shape[0],modis_lat_track.size))
    csat_to_mod_ind[csat_to_mod_ind==cldsat_reflectivity_swath.shape[0]] = cldsat_reflectivity_swath.shape[0]-1 # because the rounding is up at the last position and is thus higher than the index length
    csat_lat_swath = np.zeros(csat_to_mod_ind.shape)*np.nan
    #csat_lon_swath = np.zeros(np.shape(csat_to_mod_ind_row_unq)[0])*np.nan
    csat_max_ref_swath =  np.zeros(csat_to_mod_ind.shape)*np.nan 

    try: # In case Cloudsat is not available
        cld_sat_interval = 88#30# 46 # # of pixels
        #for i,j in enumerate(csat_to_mod_ind.astype(int)):
        st = np.where(csat_to_mod_ind.astype(int) >cld_sat_interval)[0][0]
        end = np.where(csat_to_mod_ind.astype(int) > (csat_to_mod_ind.astype(int)-cld_sat_interval))[0][-1]
        for i,j in enumerate(csat_to_mod_ind.astype(int)[st:end]):        
            ref_box = copy.copy(cldsat_reflectivity_swath[j-cld_sat_interval:j+cld_sat_interval,0:-3])
            ref_box[ref_box<-15] = np.nan
            column_max = np.nanmax(ref_box,1)
            ref_percentile = np.nanpercentile(column_max,85)
            ref_percentile_mask = column_max > ref_percentile

            # Averaging Z and then convert it to dBZ
            z = 10**(column_max[ref_percentile_mask]/10)
            #z = 10**(column_max/10) # All R>-15
            csat_max_ref_swath[i+cld_sat_interval] = 10*np.log10(np.nanmean(z))
            # Averaging dBZ
            #csat_max_ref_swath[i+cld_sat_interval] = np.nanmean(column_max[ref_percentile_mask])            
            #ref_percentile = np.nanpercentile(ref_box,95)
            #ref_percentile_mask = ref_box > ref_percentile
            #csat_max_ref_swath[i] =np.nanmean(ref_box[ref_percentile_mask])
            #csat_max_ref_swath[i] = np.nanmax(cldsat_reflectivity_swath[j-cld_sat_interval:j+cld_sat_interval,0:-3])
    except:
        print ('no cloudsat')

    ##  ECMWF surface - SST and winds  ##
    time_index = int(f_name_modis_short[14:17])-1  # The day of the year; -1 is needed because python indeces starts at 0, and Julian days with 1
    ecmwf_lon = ecmwf_surface_data['longitude'][:]
    # Converting longitude to be 0-360 like in ECMWF
    calipso_lon_swath[calipso_lon_swath < 0] = calipso_lon_swath[
        calipso_lon_swath < 0] + 360
    ecmwf_lat = ecmwf_surface_data['latitude'][:]
    sst = ecmwf_surface_data['sst'][time_index,:,:]
    u_wnd = ecmwf_surface_data['u10'][:][time_index,:,:]
    v_wnd = ecmwf_surface_data['v10'][:][time_index,:,:]
    ecmwf_time_num =  ecmwf_surface_data['time'][:][time_index] # Getting the date and printing it on the screen
    ecmwf_time = [datetime.timedelta(ecmwf_time_num/24) + datetime.datetime(1900,1,1,0,0,0)][0]
    print (ecmwf_time)

    # Co-location of ECMWF with calipso
    # intiger values of calipso lat and lon
    calipso_swath_lat_int = calipso_lat_swath.astype(int)
    calipso_swath_lon_int = calipso_lon_swath.astype(int)
    # creating matices with zeros (to be filled)
    ecmwf_calipso_lat_ind = np.zeros(calipso_swath_lat_int.size)
    ecmwf_calipso_lon_ind = np.zeros(calipso_swath_lat_int.size)
    # Only lat(in CALIPSO) matter because A-train goes south-north - CALIPSO
    for i in range(0, calipso_swath_lat_int.size, 1):
        # unique values of calipso lat and lon
        calipso_swath_lat_int_unq = calipso_swath_lat_int[i]
        calipso_swath_lon_int_unq = calipso_swath_lon_int[i]
        ecmwf_calipso_lat_ind[i] = np.where(calipso_swath_lat_int_unq == ecmwf_lat)[0][0]
        ecmwf_calipso_lon_ind[i] = np.where(calipso_swath_lon_int_unq == ecmwf_lon)[0][0]
    ecmwf_calipso_lat_ind = ecmwf_calipso_lat_ind.astype(int)
    ecmwf_calipso_lon_ind = ecmwf_calipso_lon_ind.astype(int)
    # For SST
    sst_track_calipso_orig = sst[ecmwf_calipso_lat_ind, ecmwf_calipso_lon_ind]
    sst_track_calipso_orig = sst_track_calipso_orig - 273.13 # to C
    # Smoothing the SST - moving average
    sst_track_calipso = np.zeros(sst_track_calipso_orig.shape)*np.nan
    sst_track_calipso[50:-49] = misc_codes.moving_average(sst_track_calipso_orig,100)

    # SST over the region of the ground track
    sst_max_lat_ind = np.max(ecmwf_calipso_lat_ind)
    sst_min_lat_ind = np.min(ecmwf_calipso_lat_ind)
    sst_max_lon_ind = np.max(ecmwf_calipso_lon_ind)
    sst_min_lon_ind = np.min(ecmwf_calipso_lon_ind)
    sst_swath_area = sst[sst_min_lat_ind:sst_max_lat_ind, sst_min_lon_ind-5:sst_max_lon_ind+5]

    # For winds
    u_wnd_track_calipso = u_wnd[ecmwf_calipso_lat_ind, ecmwf_calipso_lon_ind]
    v_wnd_track_calipso = v_wnd[ecmwf_calipso_lat_ind, ecmwf_calipso_lon_ind]
    wnd_speed_swath = np.sqrt(u_wnd_track_calipso**2 + v_wnd_track_calipso**2)

    # Preparing wind vectors - I select only few locations along the swath
    wnd_speed_swath_ln = wnd_speed_swath.size
    wind_vectors_interval = 50# 250 # For the plotting
    wnd_speed_swath_short = wnd_speed_swath[np.arange(0, wnd_speed_swath_ln, 100)] # selected points along the swath
    wnd_speed_swath_short_ln = wnd_speed_swath_short.size
    u_wnd_track_calipso_short = u_wnd_track_calipso[np.arange(0, wnd_speed_swath_ln, wind_vectors_interval)]
    v_wnd_track_calipso_short = v_wnd_track_calipso[np.arange(0, wnd_speed_swath_ln, wind_vectors_interval)]
    #wnd_speed_swath_short = wnd_speed_swath[np.arange(0, wnd_speed_swath_ln, 250)]

    '''
    ## Save to NetCDF
    # Organize the data into a dictionary
    case_name = int(f_name_modis_short[10:].replace('.','')) # converting to int to be save as "case id"
    case_name_vector = np.ones(len(modis_lwp_footprint))* case_name
    product_vars = {'lat':calipso_lat_swath, 'lon':calipso_lon_swath, \
                    'coupling_index': coupling_index,\
                    'coupling_index_no_cld':coupling_index_no_cld,'cloud_top_height':\
                    calipso_top_alt_swath_footprint,'cloud_base_height':cld_base_H,\
                    'calipso_cloud_base_height':\
                    calipso_base_alt_swath*1000, 'lwp_avg':modis_lwp_footprint , \
                    're_avg':modis_re_footprint, 'tau_avg':modis_tau_footprint,\
                    'R06_avg':modis_R06_footprint,\
                    'cloud_top_temperature_avg':modis_cldT_footprint, 'nd_avg':modis_nd_footprint,\
                    'SST':sst_track_calipso,'wind_speed':wnd_speed_swath, 'LTS':LTS,\
                    'divergence': divergence, 'precip_w': precip_w, \
                    'csat_max_ref':csat_max_ref_swath,'CF':modis_cf_box,\
                    'MODIS_CF':modis_cf_product, 'MODIS_CF_mask':modis_cf_mask,\
                    'case_name':case_name,'case_name_vector':case_name_vector}

    #calmod_support.save_to_netCDF(product_vars)
    '''
