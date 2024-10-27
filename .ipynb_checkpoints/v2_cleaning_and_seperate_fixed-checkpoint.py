## V2
### python script for extracting stats about specified h5 files
# 4. h5 -> seperate particles -> cleaning -> png images !

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from glob import glob
import seaborn as sns
#import xesmf as xe
import pandas as pd
#import def_homebrew as hb ## homemade functions xox
from scipy.special import gamma
#import netCDF4 as nc
from datetime import datetime, timedelta

import h5py ####
from PIL import Image
import os

from scipy.ndimage import convolve, label
from skimage.measure import regionprops, find_contours
from scipy.spatial import ConvexHull, distance_matrix
from skimage.morphology import remove_small_holes ## remove holes <3
from scipy.ndimage import binary_fill_holes
from skimage import measure
import tensorflow as tf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
## path to raw h5
# 2ds !! be careful of channels
path_h5_ds_0 = '/gws/nopw/j04/dcmex/users/ezriab/raw_h5/2ds/ch_0/'
path_h5_ds_1 = '/gws/nopw/j04/dcmex/users/ezriab/raw_h5/2ds/ch_1/'
# hvps
path_h5_hvps = '/gws/nopw/j04/dcmex/users/ezriab/raw_h5/hvps/'
base_save_path = '/gws/nopw/j04/dcmex/users/ezriab/'
# - # - # - # - # - # - # - # - # - # - # - # - EDIT BITS # - # - # - # - # - # -# - # - # - # - # - # -# - #

path = path_h5_ds_0 #################### edit depending on 2ds channel / probe ##############################
## setting thresholds / res for attaining good particle final images
fill_hole_threshold = 5 # max number pixels contained within particle that is filled in

minimum_area = 15 # very quick metric to stop the processing of particles with area < 15 pixels

length_threshold = 300 #300 # mu - need this minimum length of max dimension to extract the particle
pixel_resolution = 10 # mu
desired_image_size = 200 # (assume we want a square image) 200 x 200

# - # - # - # - # - # - # - # - # - # - # - # -# - # - # - # - # - # -# - # - # - # - # - # -# - # - # - # - #
''' practice on 1 file '''
file_list = '/gws/nopw/j04/dcmex/users/ezriab/raw_h5/2ds/ch_0/Export_base220730153000.h5'
file_names = 'Export_base220730153000.h5'
save_path = base_save_path+'processed_images/2ds/ch_0/'

'''
if os.path.exists(path):
    # get string of full path + filenames in specif location
    file_list = glob(path+'Export_base*.h5') 
    
    # just get file names
    file_names = [os.path.basename(file_path) for file_path in file_list]
else:
    print("NOT REAL OH NO")

'''

## adding automation - makes sure to change when processing images !!!!!!!!!!!!!
if '2ds' in file_list[0]:
    if 'ch_0' in file_list[0]:
        #save_path = base_save_path+'processed_stats/ch_0/'
        save_path = base_save_path+'processed_images/2ds/ch_0/'
        particle_type = 'ch_0'

    elif 'ch_1' in file_list[0]:
        #save_path = base_save_path+'processed_stats/ch_1/'
        save_path = base_save_path+'processed_images/2ds/ch_1/'
        particle_type = 'ch_1'

elif 'hvps' in file_list[0]:
    #save_path = base_save_path+'processed_stats/hvps/'
    save_path = base_save_path+'processed_images/2ds/hvps/'
    particle_type = 'hvps'


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

## functions to make code run smoothly
def stats_description(bw_crystal, fill_hole_thresh):
    #take binary image, fill in small holes and returns object containing stats about crystal
    
    filled_particle = remove_small_holes(bw_crystal.image, area_threshold=fill_hole_thresh) # fill in voids within binary image - better estimation of stats # may need to be altered
    
    # can see the filled in particle if needs be
    #plt.imshow(filled_particle, cmap='gray')
    
    if filled_particle.shape[0] < 2 or filled_particle.shape[1] < 2:
        return filled_particle, None
        
    contours = measure.find_contours(filled_particle, 0.5)
    if contours:
        contour = max(contours, key=lambda x: x.shape[0])  # Sort contours by area (largest first) and select the largest contour
        
        labeled_image = measure.label(filled_particle)  # Label the image based on the threshold
        region = measure.regionprops(labeled_image)[0]  # Assumes largest labeled region corresponds to largest contour
        
        return filled_particle, region
    else:
        return filled_particle, None
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## function to calculate truncation of particle
def calc_truncation(a_slice, particle):
    # the intial slice is the raw 2ds data - of whole array, and particle is one selected by regionprops (and has to be 1s + 0s)
    ## assume slices are small + don't contain too many odd bits
    # first convert to 0 and 1 for calculation of truncation
    alt_crystal = np.where(a_slice == 255, 0, 1) # i.e. 255 (blank area)=0, and where crystal is (was 0) = 1
    # so sum up number of particle pixels are on the edge of the slice
    first_diode = sum(alt_crystal[0,:]) 
    last_diode = sum(alt_crystal[-1,:])

    ## this calculates how many pixels are top / bottom of the particle + then infer number pixels touching
    top_particle = np.sum(particle[0] == 1)
    bottom_particle = np.sum(particle[-1] == 1)

    n_top, n_bottom = 0, 0  # Initialize variables, default 0 when conditions are not met

    # Top pixel touching logic
    if first_diode != 0 and first_diode >= top_particle:
        n_top = top_particle
    elif first_diode == 0:
        n_top = 0

    # Bottom pixel touching logic
    if last_diode != 0 and last_diode >= bottom_particle:
        n_bottom = bottom_particle
    elif last_diode == 0:
        n_bottom = 0

    return n_top, n_bottom # number pixels touching top / bottom respectively

###  set up dataframe, used to extract from raw h5 file + has stats about the particle
columns = [
    "name",
    "date",
    "slice_s_idx",
    "slice_e_idx",
    "start_time",
    "end_time",
    "d_max",
    "d_min",
    "orientation",
    "centroid",
    "area",
    "perimeter",
    "circularity",
    "y0",
    "y1",
    "probe",
    "first_diode_trunc",
    "last_diode_trunc"
    ]

#start# outer loop for processing each file ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
for j in range(len(file_list)):
	

    ## make folder within correct directory for each file
    long_date_string = file_names[j][-15:-3]
    if not os.path.exists(save_path+long_date_string):
        os.makedirs(save_path+long_date_string)
    flight_save_loc = save_path+long_date_string+'/'

    
    h5_file = h5py.File(file_list[j],'r')
    print(f'running {file_names[j]}')
    particle_df = pd.DataFrame(columns=columns) # empty df for each day of flights
    
    try:
        h5_image = h5_file['ImageData']
        h5_time = h5_file['ImageTimes']
    except KeyError as e:
        print(f"Dataset missing in file: {file_names[j]}. Error: {e}")
        continue
    
    ##### make xarray of useful time data #####
    sec_since = h5_time[:,0]
    pixel_slice = h5_time[:,1]
    pix_sum = pixel_slice.cumsum(dtype = 'int')
    
    ## make useful datetime format (not seconds since midnight)
    # using the file name for reference
    date_str = file_names[j][-15:-9]
    starting_date = datetime.strptime(date_str, '%y%m%d')
    time_deltas = [timedelta(seconds=float(sec)) for sec in sec_since]
    utc_time = [starting_date + delta for delta in time_deltas]
    
    time_xr =xr.Dataset({
    'utc_time':utc_time,
    'pixel_slice': pixel_slice,
    'pix_sum': pix_sum})
    
    ## cleaning of whole h5 file - quick removal of tiny particles
    ## this has been edited - allow for corresponding time
    pix_sum = time_xr['pix_sum']
    utc_time = time_xr['utc_time']
    
    # Calculate the difference
    diff = np.diff(pix_sum.values)
    
    # Create a mask where the difference is greater than 4 - i.e. select segments of significant sizze
    mask = diff > 4
    # Apply the mask to select the corresponding values from pix_sum and utc_time
    selected_pix_sum = pix_sum[:-1][mask]
    selected_utc_time = utc_time[:-1][mask]
    print('started processing')
    ## accompanying txt file with info about processing 
    with open(f'{flight_save_loc}running_{long_date_string}.txt', "w") as file:
        # start # inner loop for processing each slice ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
        for i in range(len(selected_pix_sum)-2):
        #for i in range(len(selected_pix_sum)-2):
        
            # pull out selected area + do analysis
            one_crystal = h5_file['ImageData'][:,int(selected_pix_sum[i]):int(selected_pix_sum[i+1])] # extract 1 crystal
            
            binary_image = (one_crystal == 0) ## important, convert regions where 0 = True (our bits of interest), all else false
            
            labeled_image, num_features = label(binary_image) # identify connected true areas
            # labeled_image = array, with each true area given a number to identify them
            # num_features = number of unique connected components in image. Have to literally have adjacent pixel, not diagonal (this will make them seperate)
            
            props = regionprops(labeled_image) # creates quick list of properties describing each feature detected in the image.
            ## (features are measured in ~ pixels)
            
            if props:
                #start # inner loop for processing each particle # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
                for particle in props:
                # go through each particle detected
                    # quickly get rid of tiny particles
                    if particle.area >= minimum_area:
                        ## more complex stats
                        filled_part, spec_region = stats_description(particle,fill_hole_threshold)
                        
                        ### quite important, this is remove unwanted particles
                        # putting in some conditions to take measurements: d_max needs to be min size, and so does d_min - 4 pixels (remove streak particles)
                        if spec_region and spec_region.major_axis_length * pixel_resolution >= length_threshold and spec_region.minor_axis_length > 4:
                            ## basic info
                            coords = particle.coords # basically gives coords of each point of interest
                            x_values = np.unique(coords[:, 1])
                            s_idx = int(selected_pix_sum[i] + x_values[0])
                            e_idx = int(selected_pix_sum[i] + x_values[-1])
            
            
                             ## truncation calc
                            first_diode, last_diode = calc_truncation(one_crystal, filled_part)
                            
                            ## using circularity calculation from Crosier et al. 2011
                            circularity_calc = np.divide((spec_region.perimeter**2),(4*np.pi*spec_region.area))
                            
                            # nice way of saving data - lenth + measurements are correct in microns
                            one_particle_data = {
                                    #"image_index": image_index,
                                    "name": f'{s_idx}_{particle.label}_{particle_type}',
                                    "date" : date_str,
                                    #"particle_label": particle.label,
                                    "slice_s_idx": s_idx,
                                    "slice_e_idx": e_idx,
                                    "start_time": str(selected_utc_time[i].values).split('T')[1], # more friendly time
                                    "end_time": str(selected_utc_time[i+1].values).split('T')[1], # more friendly time
                                    "d_max": spec_region.major_axis_length * pixel_resolution, ## d_max
                                    "d_min": spec_region.minor_axis_length * pixel_resolution, ## d_min
                                    "orientation": spec_region.orientation,
                                    "centroid": spec_region.centroid,
                                    "area": (spec_region.area * (pixel_resolution**2)),
                                    "perimeter": (spec_region.perimeter * pixel_resolution),
                                    "circularity": circularity_calc,
                                    "y0": coords[0][0],
                                    "y1": coords[-1][0],
                                    "probe": particle_type,
                                    "first_diode_trunc": first_diode,
                                    "last_diode_trunc": last_diode
                                    }
                            #print(f'{s_idx} done')
                            one_particle_data_df = pd.DataFrame([one_particle_data])
                            particle_df = pd.concat([particle_df, one_particle_data_df], ignore_index=True)
                            file.write(f'{s_idx}_{particle.label} stats done \n')
                            
                            ############################# re size + save image ##################################################
                            filled_part = filled_part.astype(np.float32) ## convert to float 0 and 1s
                            filled_part = np.expand_dims(filled_part, axis=-1) ## add extra dimention - this is for adding padding
            
                            imagex = tf.image.resize_with_crop_or_pad(filled_part, desired_image_size, desired_image_size)
    
                            ## this is checking the image is not blank - written in seperate txt file
                            if np.all(imagex == 0):
                                file.write(f'{s_idx}_{particle.label} image is blank - 0s \n')
                                #print("The image is blank (all pixels are zero).")
                            elif np.all(imagex == imagex[0, 0, 0]):
                                file.write(f'{s_idx}_{particle.label} image is blank - constant values \n')
    
                            ## save image
                            # Remove the extra dimension if needed
                            image_np = imagex.numpy().squeeze()
                            
                            # Save the image using matplotlib
                            plt.imsave(f'{flight_save_loc}{s_idx}_{particle.label}.png', image_np, cmap="gray")
                            file.write(f'{s_idx}_{particle.label} image saved \n')
                    ###################################################################################################
                #end # inner loop for processing each particle # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
        ## save the stats
        particle_df.to_csv(f'{flight_save_loc}flight_{long_date_string}.csv', index=False)
        print(f'flight_{long_date_string} done')
        '''
        if not os.path.exists(f'{flight_save_loc}flight_{long_date_string}.csv'):
            particle_df.to_csv(f'{flight_save_loc}flight_{long_date_string}.csv', index=False) 
            print(f'flight_{long_date_string}.csv saved sucessfully!')
        else:
            print("file already exists")
        
        h5_file.close()    
        '''
        # end # inner loop for processing each slice ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~#
        
    #end# outer loop for processing each file ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
