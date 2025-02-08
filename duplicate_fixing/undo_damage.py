## making mistakes while re-naming images and csvs? 
# not sure, they are not aligning when re-naming them

# this script removed the newly created csvs + undo the naming scheme added onto the images

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

stats_path = '/gws/nopw/j04/dcmex/users/ezriab/processed_stats/'
image_path = '/gws/nopw/j04/dcmex/users/ezriab/processed_images/'
folder_extension ='2ds/ch1/' ## 'hvps/' '2ds/ch0/' '2ds/ch1/'

file_endings_to_remove = ['_1','_2','_3','_4']
full_path = image_path+folder_extension

list_dir = os.listdir(full_path)
for i in range(len(list_dir)):
    specif_dir = list_dir[i]
    path_to_images = full_path + specif_dir + '/'
    png_path = glob(path_to_images+'*.png') 
    img_file_names = [os.path.basename(file_path) for file_path in png_path]
    
    try:
        file_ending = img_file_names[0][-6:-4]
        bool_file = any(i in file_ending for i in file_endings_to_remove) ## tell us if the additional ending has been added
        print(path_to_images)
        print(file_ending)
        
        if bool_file == True:
            for specif_img in img_file_names:
                base_name = specif_img[:-4]
                
                new_name = f'{base_name[:-2]}.png'
                #print(f'{specif_img} {new_name}')
                os.rename(f'{path_to_images}{specif_img}', f'{path_to_images}{new_name}')
                
        
    except IndexError: # some folders have no images
        continue
    
# and delete the csvs 
csv_full_path = f'{stats_path}{folder_extension}particle_stats/'

csv_list = glob(csv_full_path+'*.csv') 

file_names = [os.path.basename(file_path) for file_path in csv_list]
changed_file_list = []
for file in file_names:
    if len(file) >27:
        changed_file_list.append(file)

for file in changed_file_list:
    path = csv_full_path+file
    if os.path.exists(path):
        os.remove(path)
        print(f'{path} removed')