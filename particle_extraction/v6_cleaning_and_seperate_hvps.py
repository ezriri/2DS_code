## V6- removed some limits when extracting particles - i.e. no aspect ratio threshold or no minimum diameter (probably removing columns)
# multiple diameters + final CNN trunaction calculation 
# threshold diameter updated to Euclidean Diameter
### python script for extracting stats about specified h5 files
# 4. h5 -> seperate particles -> cleaning -> png images !
# no duplicates, new time recorded - only seconds since midnight, new aspect ratio (using max / min x/y coordinates)

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from glob import glob
import seaborn as sns
import pandas as pd
from scipy.special import gamma
from datetime import datetime, timedelta

import h5py ####
from PIL import Image
import os
import psutil

from scipy.ndimage import convolve, label
from skimage.measure import regionprops, find_contours
from scipy.spatial import ConvexHull, distance_matrix
from scipy.spatial.distance import pdist
from skimage.morphology import remove_small_holes ## remove holes <3
from scipy.ndimage import binary_fill_holes
from skimage import measure
import tensorflow as tf

from numba import jit
# - # - # - # - # - # - # - # - # - # - # - # - EDIT BITS # - # - # - # - # - # -# - # - # - # - # - # -# - # - # - # - # - # - # -# - # - # - # - # - # -# - # - # - # - # - # - # -# - # - # - # - # - # -# - # - # - # - # 
particle_type = 'hvps' #'ch0', 'ch1', 'hvps' ###  depending on 2ds channel / probe ##############################

## setting thresholds / res for attaining good particle final images
fill_hole_threshold = 5 # max number pixels contained within particle that is filled in
minimum_area = 15 # very quick metric to stop the processing of particles with area < 15 pixels
desired_image_size = 200 # (assume we want a square image) 200 x 200 for 
# - # - # - # - # - # - # - # - # - # - # - # -# - # - # - # - # - # -# - # - # - # - # - # -# - # - # - # - # - # - # - # - # - # -# - # - # - # - # - # -# - # - # - # - # - # - # -# - # - # - # - # - # -# - # - # - # - #
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if particle_type != 'ch0'and particle_type != 'ch1' and particle_type != 'hvps':
    print('please specify correct probe')
    raise SystemExit

#################  automation, for quick getting files + saving
base_save_path = '/gws/nopw/j04/dcmex/users/ezriab/'

if particle_type != 'hvps':
    length_threshold = 100 # mu - need this minimum length of max dimension to extract the particle
    pixel_resolution = 10 # mu for 2DS
    
    if particle_type == 'ch0':
        path = '/gws/nopw/j04/dcmex/users/ezriab/raw_h5/2ds/ch0/' # raw h5
        stats_save_path = base_save_path+'processed_stats/2ds/ch0_v6/'
        save_path = base_save_path+'processed_images/2ds/ch0_v6/'

    elif particle_type == 'ch1':
        path = '/gws/nopw/j04/dcmex/users/ezriab/raw_h5/2ds/ch1/' # raw h5
        stats_save_path = base_save_path+'processed_stats/2ds/ch1_v6/'
        save_path = base_save_path+'processed_images/2ds/ch1_v6/'

elif particle_type == 'hvps':
    length_threshold = 1500 # mu - need this minimum length of max dimension to extract the particle
    pixel_resolution = 150 # mu for HVPS
    path = '/gws/nopw/j04/dcmex/users/ezriab/raw_h5/hvps/' # raw h5
    stats_save_path = base_save_path+'processed_stats/hvps_v6/'
    save_path = base_save_path+'processed_images/hvps_v6/'

if os.path.exists(path):
    # get string of full path + filenames in specif location
    file_list = glob(path+'Export_base*.h5') 
    
    # just get file names
    file_names = [os.path.basename(file_path) for file_path in file_list]
else:
    print(f"something has gone wrong, {path} does not exist")
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
## functions to make code run smoothly
def stats_description(bw_crystal, fill_hole_thresh):
    #take binary image, fill in small holes and returns object containing stats about crystal
    filled_particle = remove_small_holes(bw_crystal.image, area_threshold=fill_hole_thresh) # fill in voids within binary image - better estimation of stats # may need to be altered
    
    # this checks if there is an actual particle. if .shape[0] < 2 (row count <2) or shape[1] (column count <2)
    if filled_particle.shape[0] < 2 or filled_particle.shape[1] < 2:
        return None, None
        
    contours = measure.find_contours(filled_particle, 0.5)
    if contours:
        contour = max(contours, key=lambda x: x.shape[0])  # Sort contours by area (largest first) and select the largest contour
        
        labeled_image = measure.label(filled_particle)  # Label the image based on the threshold
        region = measure.regionprops(labeled_image)[0]  # Assumes largest labeled region corresponds to largest contour
        
        return filled_particle, region
    else:
        return None, None
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
## quite important, to see how memory is being used
def log_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024**2:.2f} MB")  # RSS in MB
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

## function to calculate truncation of particle
@jit(nopython=True) # Enables full optimization by numba
def calc_truncation(particle_coords):
    ## so much simpler, looking at list of coordinates making up a particle, then summing ones in 0 and 127 row - i.e. first + last diode
    lst_first_diode = [coord for coord in particle_coords if coord[0] == 0]
    lst_last_diode = [coord for coord in particle_coords if coord[0] == 127]

    n_top = len(lst_first_diode)
    n_bottom = len(lst_last_diode)

    return n_top, n_bottom # number pixels touching top / bottom respectively

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -  #function to find x / y coords and calculate aspect ratio.
@jit(nopython=True)
def coords_etc(particle_coords):
    x_values = np.unique(particle_coords[:, 1])
    y_cds = (particle_coords[0][0],particle_coords[-1][0])
    x_cds = (x_values[0], x_values[-1])

    y_length = y_cds[1] - y_cds[0]
    x_length = x_cds[1] - x_cds[0]
    
    max_dim = max([y_length,x_length])
    min_dim = min([y_length,x_length])
    aspect_ratio = max_dim/min_dim
    
    return x_cds, y_cds, aspect_ratio

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
###  set up dataframe, used to extract from raw h5 file + has stats about the particle
columns = [
    "name",
    "date",
    "slice_s_idx",
    "slice_e_idx",
    #"utc_start", #hh:mm:ss 
    #"utc_end", #hh:mm:ss 
    "ss_start", ## seconds since midnight on the day - should help with deadtime calculations
    "ss_end", ## start / end time tell us time particle was imaged
    "x_coords",
    "y_coords",
    #"ellipse_d_max", # um
    #"ellipse_d_min", # um
    "Euclidean_d_max", # um
    "Feret_d_max", # um
    "area", # um2
    "perimeter", # um
    "circularity",
    "probe",
    "first_diode_trunc",
    "last_diode_trunc",
    "image_trunc",
    "aspect_ratio" ## this is calculated from max / min x/y coordinates length
    ]

# number crunching time ! crunch crunch ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
print('begin processing')
log_memory_usage()

## i am also keeping track of indentation     -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    - 
# 1. loop: - process each file   -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    - 
for j in range(len(file_list)):
    h5_additional_label = str(j+1) # this is to get unique names (how Jonny split up h5 -> some days have multiple -> a problem for me)
    
    ## make folder within correct directory for each file
    long_date_string = file_names[j][-15:-3]
    long_string_unique = f'{save_path}{long_date_string}_{h5_additional_label}' # adding on this label to the folders too
    
    if not os.path.exists(long_string_unique):
        os.makedirs(long_string_unique)
    flight_save_loc = long_string_unique+'/'
        
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
    date_day = date_str[-2:]
    
    starting_date = datetime.strptime(date_str, '%y%m%d')
    time_deltas = [timedelta(seconds=float(sec)) for sec in sec_since]
    utc_time = [starting_date + delta for delta in time_deltas]
    
    time_xr =xr.Dataset({
    'utc_time':utc_time,
    'sec_since':sec_since,
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

# 2. with: txt file, write to continously about progress of image processing    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    - 
    with open(f'{flight_save_loc}running_{long_date_string}.txt', "w") as file:
# 3. loop: processing each slice  -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -
        for i in range(len(selected_pix_sum)-2):        
            # pull out selected area + do analysis
            one_crystal = h5_file['ImageData'][:,int(selected_pix_sum[i]):int(selected_pix_sum[i+1])] # extract 1 crystal
            
            binary_image = (one_crystal == 0) ## important, convert regions where 0 = True (our bits of interest), all else false
            
            labeled_image, num_features = label(binary_image) # identify connected true areas
            # labeled_image = array, with each true area given a number to identify them
            # num_features = number of unique connected components in image. Have to literally have adjacent pixel, not diagonal (this will make them seperate)
            
            props = regionprops(labeled_image) # creates quick list of properties describing each feature detected in the image.
            ## (features are measured in ~ pixels)
# 4. if: are there particles in the slice?   -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -
            if props:
# 5. loop: yes there are particles in the slice, we shall go through them one by one -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    - 
                for particle in props:
                # go through each particle detected
                    # quickly get rid of tiny particles
# 6. if: does the particle meet required minumum area? if too small, skip  -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    - 
                    if particle.area >= minimum_area:
                        ## more complex stats
                        filled_part, spec_region = stats_description(particle,fill_hole_threshold)

# 7. if: does particle have a measured region + properties?  -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -
                        if spec_region:
                            ## euclidean diameter calculation
                            a_particle_coords = particle.coords # basically gives coords of each point of interest [row,column]
                            distances = pdist(a_particle_coords)
                            euclidean_dim = np.max(distances) # max distance between 2 sets of coordinates
                            
# 8. if: does the particle meet the size thresold?  -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -    -
# only particles making it past this point will be recorded
                            ## main threshold, need minimum diameter size to be of use
                            if euclidean_dim * pixel_resolution >= length_threshold:
                                ## basic info
                                x_coords, y_coords, aspect_ratio_value = coords_etc(a_particle_coords)
                                
                                s_idx = int(selected_pix_sum[i] + x_coords[0])
                                e_idx = int(selected_pix_sum[i] + x_coords[-1])
                
                                ## truncation calc
                                # cnn - in final pic size, particle may be truncated if very long, this tells us how many pixels may be cut off 
                                image_trunc = x_coords[-1] - desired_image_size 
                                if image_trunc < 0:
                                    image_trunc = 0
                                # normal trunc calculation - on actual probe
                                first_diode, last_diode = calc_truncation(a_particle_coords)
                                
                                ## using circularity calculation from Crosier et al. 2011
                                circularity_calc = np.divide((spec_region.perimeter**2),(4*np.pi*spec_region.area))
                                particle_name = f'{s_idx}_{particle.label}_{date_day}{particle_type}_{h5_additional_label}'
                                
                                # nice way of saving data - lenth + measurements are correct in microns
                                one_particle_data = {
                                        #"image_index": image_index,
                                        "name": particle_name,
                                        "date" : date_str,
                                        "slice_s_idx": s_idx,
                                        "slice_e_idx": e_idx,
                                        #"start_time": str(selected_utc_time[i].values).split('T')[1], # more friendly time
                                        #"end_time": str(selected_utc_time[i+1].values).split('T')[1], # more friendly time
                                        #"ellipse_d_max": spec_region.major_axis_length * pixel_resolution, ## d_max (equivalent ellipse)
                                        #"ellipse_d_min": spec_region.minor_axis_length * pixel_resolution, ## d_min (equivalent ellipse)
                                        "ss_start":time_xr['sec_since'][s_idx].values,
                                        "ss_end":time_xr['sec_since'][e_idx].values,
                                        "x_coords": x_coords,
                                        "y_coords": y_coords,
                                        "Euclidean_d_max": euclidean_dim * pixel_resolution,
                                        "Feret_d_max":spec_region.feret_diameter_max * pixel_resolution,
                                        "area": (spec_region.area * (pixel_resolution**2)),
                                        "perimeter": (spec_region.perimeter * pixel_resolution),
                                        "circularity": circularity_calc,
                                        "probe": particle_type,
                                        "first_diode_trunc": first_diode,
                                        "last_diode_trunc": last_diode,
                                        "image_trunc": image_trunc,
                                        "aspect_ratio": aspect_ratio_value  
                                        }

                                one_particle_data_df = pd.DataFrame([one_particle_data])
                                particle_df = pd.concat([particle_df, one_particle_data_df], ignore_index=True)
                                file.write(f'{particle_name} stats done \n')
                                
                                ############################# re size + save image ##################################################
                                filled_part = filled_part.astype(np.float32) ## convert to float 0 and 1s
                                filled_part = np.expand_dims(filled_part, axis=-1) ## add extra dimention - this is for adding padding
                
                                imagex = tf.image.resize_with_crop_or_pad(filled_part, desired_image_size, desired_image_size)
        
                                ## this is checking the image is not blank - written in seperate txt file
                                if np.all(imagex == 0):
                                    file.write(f'{particle_name} image is blank - 0s \n')
                                elif np.all(imagex == imagex[0, 0, 0]):
                                    file.write(f'{particle_name} image is blank - constant values \n')
        
                                ## save image
                                # Remove the extra dimension if needed
                                image_np = imagex.numpy().squeeze()
                                
                                # Save the image using matplotlib
                                plt.imsave(f'{flight_save_loc}{particle_name}.png', image_np, cmap="gray")
                                file.write(f'{particle_name} image saved \n')
                    ###################################################################################################
                #end # inner loop for processing each particle # ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ #
        ## save the stats  
# ~ ~ ~ # within loop 2. this happens for each h5 file:
        if not os.path.exists(f'{stats_save_path}flight_{long_date_string}.csv'):
            particle_df.to_csv(f'{stats_save_path}flight_{long_date_string}.csv', index=False) 
            print(f'flight_{long_date_string}.csv saved sucessfully!')
        else:
            print("file already exists")
        print('end of file')
        log_memory_usage()
        h5_file.close()
        
        # end # inner loop for processing each slice ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~#
        
    #end# outer loop for processing each file ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~    
