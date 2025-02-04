

## SUPERVISOR VERSION - subsampling from my already labelled images!
## script to run to do random sample of images, make a csv containing this list and group the images into a folder
## also includes a 10% overlap of sample split between people.

import random
import xarray as xr
import numpy as np
from glob import glob
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy.stats import iqr
import os
from glob import glob
import shutil


image_loc = '/gws/nopw/j04/dcmex/users/ezriab/image_labelling/2ds_10000_sample/' ## where are all the images / og csv file
to_sample_from_csv = '2ds_10000_new_2nd_fixed_dup.csv' ## assume 'name' 'number_label' 'note' column set up - note should say if any weird images to avoid 
base_path_to_save = '/gws/nopw/j04/dcmex/users/ezriab/image_labelling/supervisor_label/' # where output to be saved

### importantly, the numbers have to be nicely divisible by one another !!!
n_sample = 900
n_people = 3 ## how many people do we want this split between
overlap_per = 10 ## percentage of overlap images, i.e. 10% total, but this is split between the n_people


# ~~ do not edit past this point ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
# functions gunctions #######################################################
def save_csv(image_list, save_full_path):
    csv_column_names = ['image_name','number_label']
    csv_dic = dict.fromkeys(csv_column_names)
    csv_dic['image_name'] = image_list
    csv_df = pd.DataFrame.from_dict(csv_dic)
    random_csv_df.to_csv(save_full_path, index=False)

def make_folder_move_image(set_of_images, start_loc, end_loc):
    # make sub-folder etc
    if not os.path.exists(end_loc):
        os.makedirs(end_loc)
        print(f'created {end_loc}')    
    
    ## loop though and move each image
    for image in set_of_images:
        current_image_loc = f'{image_loc}{image}.png'
        
        if os.path.exists(current_image_loc) and os.path.exists(end_loc):
            shutil.copy(current_image_loc, end_loc)
############################################################################

# some of my labelled images are a bit weird, so will ignore.
original_10000_sample = pd.read_csv(image_loc+to_sample_from_csv)
usuable_images = original_10000_sample[original_10000_sample['note'] != 'a']
usuable_images_list = list(usuable_images['name'])

# main sample of images
random.seed(42) # !! v important, means we choose same set images every time. 
r_sample_list = random.sample(usuable_images_list, k=n_sample)

# # # # # # now split the sample between supervisors # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
n_sample = int(len(r_sample_list) / n_people)
segments = list(range(0,len(r_sample_list)+1,n_sample)) # segment up data
per_subsample = int(segments[1]/overlap_per)
dic_of_sample = {}

for i in range(n_people):
    segment_of_sample = r_sample_list[segments[i]:segments[i+1]] # this is one sample of data
    overlap_sample = random.sample(segment_of_sample, per_subsample) # represents a subsample of data that will be used to overlap checking
    dic_of_sample[str(i)] = [segment_of_sample,overlap_sample]

## e.g. dic_of_sample['0'][0] = 300 sample images that the person needs to label
##      dic_of_sample['0'][1] = random 10% of the 300 that other people need to label


# # # now deal with the overlap images # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
## now getting complex, need to split the second ovelap between the ones not extracted from
n_people_int = list(range(0,n_people))
overlap_value = len(dic_of_sample['0'][1])
n_split = n_people -1 ## how many the subsample needs to be split between
n_sample_overlap = int(overlap_value/n_split)
overlap_segs = list(range(0, overlap_value+1, n_sample_overlap)) # segment up overlap data
n_people_list = [str(value) for value in n_people_int]

dic_of_overlap = {key: [] for key in n_people_list } ## corresponds to first one, i.e. [0] sample is [0] seperate overlap sample
# so dic_of_overlap[0] = images that overlap with 1 / 2

for i in range(n_people):
    # remove the current value
    append_list = [item for item in n_people_int if item != i]
    subsample_data = dic_of_sample[str(i)][1]
    #print(len(subsample_data))
    for j in range(n_split):
        dic_of_overlap[str(append_list[j])].append(subsample_data[overlap_segs[j]:overlap_segs[j+1]])
        # so each value has 2 lists - that needs to be combined

## e.g. 3 people label - so each person has subsample of 30 to split between other two
##      dic_of_overlap['0'][0] = 15 random images from dic_of_sample['1'][1]
##      dic_of_overlap['0'][1] = 15 random images from dic_of_sample['2'][1]
## both these lists joined together -> the overlap to also sample


# #
## final labelling csv # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
overlapping_images = []
final_lists = {}

for i in range(n_people):
    original_big_sample = dic_of_sample[str(i)][0]

    for j in range(n_split):
        overlap_images = dic_of_overlap[str(i)][j]
        original_big_sample.extend(overlap_images)
        overlapping_images.extend(overlap_images) # want big list of all the overlapping images
    
    final_lists[str(i)] = original_big_sample

## final_lists contains 3 keys for each supervisor, with their final respective list for labelling
## then overlapping_images contains the list of images that overlap

## saving time # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
## loop through main supervisor script - their specific set of images ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
for supervisor_number in final_lists:
    ## csv name matches to the folder corresponding to images
    corresponding_name = f'supervisor_{supervisor_number}'
    csv_save_path = f'{base_path_to_save}{corresponding_name}.csv'
    set_of_images = final_lists[supervisor_number]
    image_save_path = f'{base_path_to_save}{corresponding_name}/'
    ## csv saving
    save_csv(set_of_images, image_save_path)
    ## function which does what it says
    make_folder_move_image(set_of_images, image_loc, image_save_path)
    print(f'moved {supervisor_number} images!')


## do the overlapping images seperately ## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
overlap_csv_save_path = f'{base_path_to_save}overlapping_images.csv'
overlap_image_save_path = f'{base_path_to_save}overlapping_images/'
## csv saving
save_csv(overlapping_images, overlap_csv_save_path)

make_folder_move_image(overlapping_images, image_loc, overlap_image_save_path)
print(f'moved overlapping images!')

