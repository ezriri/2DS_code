## (08/02/25)
# so many frustrating iterations and find out about duplicate names
# this script finds the days which have multiple h5 files associated to them, then add a exentsion to the image and csv file


import numpy as np
import pandas as pd
import os
#from PIL import Image
from glob import glob
import csv
import matplotlib.pyplot as plt
from IPython.display import clear_output
import math
import shutil

from numba import jit
import h5py

## need to go through folders, days which have more than 1 folder, need to have extension _1 or _2 added onto each name and folder 
folder_extension = '2ds/ch0/' ## 'hvps/' '2ds/ch0/' '2ds/ch1/' !!!!!!!!!!!!!!!!!!!!!!!

stats_path = '/gws/nopw/j04/dcmex/users/ezriab/processed_stats/' # !!!!!!!!!!!!!!!!!!!!!!!
image_path = '/gws/nopw/j04/dcmex/users/ezriab/processed_images/' # !!!!!!!!!!!!!!!!!!!!!!!

# csvs with particle stats also attached
csv_full_path = f'{stats_path}{folder_extension}particle_stats/'

base_folder_path = f'{image_path}{folder_extension}'
# maybe best to do it by folder instead, still follow the same naming convention
# get all folder names that have duplicate days, list them out
# open the csvs associated with them 
# loop though csvs, re-name names, and do the same for all the images

list_dir = os.listdir(base_folder_path) ## includes some others that don't contain images
#image_dir_mask = [folder.isnumeric() for folder in list_dir]
image_dir = [] # just ones with images
shortened_dir = []
for folder in list_dir:
    if '220' in folder:
        image_dir.append(folder)
        shortened_dir.append(folder[:6])


# need to find the duplicate days
dup_folders = []
day_overlap = []
for i in range(len(shortened_dir)):
    for j in range(len(shortened_dir)):
        if i ==j:
            continue # these are obviously the same
        else:
            og_folder = shortened_dir[i]
            compare_folder = shortened_dir[j]
            if og_folder == compare_folder:
                dup_folders.append(image_dir[i])
                day_overlap.append(og_folder)

days_dic = {key: [] for key in day_overlap}
#days_dic = dict.fromkeys(day_overlap)
key_list = list(days_dic)
for day in key_list:
    for overlap in dup_folders:
        overlap_short = overlap[:6]
        #print(f'{day} {overlap_short}')
        if day == overlap_short:
            days_dic[day].append(overlap)

## some of the lists have duplicates, so just adding this second cleaning step
for k in days_dic:
    og_list = days_dic[k]
    updated_list = list(set(og_list))
    days_dic[k] = updated_list


## this is the main bulk of changing
for key in days_dic:
    number_dup = len(days_dic[key])
    for i in range(number_dup):
        specif_folder = days_dic[key][i]
        additional_label = f'_{str(i+1)}'
        # find the corresponding csv (of envi stats):
        base_folder_name = f'{csv_full_path}{specif_folder}_envi_stats'
        current_folder_stats = f'{base_folder_name}.csv'
        updated_folder_stats = f'{base_folder_name}{additional_label}.csv'
        
        if os.path.exists(current_folder_stats):
            # probably worth making a new csv with the updated names
            print('its worked!')
            df = pd.read_csv(current_folder_stats) # og names
            new_df = df.copy()
            ## change each name
            new_df['name'] = new_df['name'] + additional_label
            new_df.to_csv(updated_folder_stats)
        
        # now fix the images  
        image_folder = f'{base_folder_path}{specif_folder}/' # contains corresponding images
        if os.path.exists(image_folder):
            img_file_list = glob(image_folder+'*.png') 
            img_file_names = [os.path.basename(file_path) for file_path in img_file_list]
            for specif_img in img_file_names:
                base_name = specif_img[:-4]
                new_name = f'{base_name}{additional_label}.png'
                # time for the re-naming
                os.rename(f'{image_folder}{specif_img}', f'{image_folder}{new_name}')

        print(f'updated {specif_folder}'
        '''
        ## re-name the image folders
        og_folder_name = f'{image_path}{specif_folder}'
        new_folder_name = f'{image_path}{specif_folder}{additional_label}'
        os.rename(og_folder_name, new_folder_name)
		'''