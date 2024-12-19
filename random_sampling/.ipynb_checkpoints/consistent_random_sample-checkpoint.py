## script to run to do random sample of images, make a csv containing this list and group the images into a folder
# may need to add in new steps if i have to re-sample again

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

## which csv? - this only does 1 csv at a time 
ds_csv = '2ds/all_2ds.csv'
hvps_csv = 'hvps/particle_stats/all_hvps.csv'
all_csv = 'all_2ds_hvps.csv'
data_path = '/gws/nopw/j04/dcmex/users/ezriab/processed_stats/'

full_csv_path = data_path + ds_csv # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ 
random_save_name = 'test_random_sample' # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ 
n_sample = 10 # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ # ~ 

# ~~ do not edit past this point ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
save_loc = '/gws/nopw/j04/dcmex/users/ezriab/image_labelling/' # for csv + images to be stored together
full_save_path = save_loc+random_save_name+'/'
# make new folder for images + csv 
if not os.path.exists(save_loc+random_save_name):
    os.makedirs(save_loc+random_save_name)
    print(f"{random_save_name} folder created successfully!")
else:
    print("Folder already exists.")


# function to slice out images that don't make the threshold of being viable for the CNN
def reduce_full_df(big_df):
    smaller_df = big_df[
    (big_df['Euclidean_d_max'] >= 300) &
    (big_df['first_diode_trunc'] == 0) &
    (big_df['last_diode_trunc'] == 0) &
    (big_df['image_trunc'] == 0) &
    (~big_df.isnull().any(axis=1))]  # Exclude rows with NaN
    return smaller_df

csv_column_names = ['image_name','number_label']#,'habit']
random_csv_dic = dict.fromkeys(csv_column_names)

#### actual sampling + save csv happening here # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
full_df = pd.read_csv(full_csv_path)
reduced_df = reduce_full_df(full_df)
reduced_name_lst = list(reduced_df['name'])
random.seed(42) # !! v important, means we choose same set images every time. 
r_sample_list = random.sample(reduced_name_lst, k=n_sample)
random_csv_dic[csv_column_names[0]] = r_sample_list

random_csv_df = pd.DataFrame.from_dict(random_csv_dic)
if os.path.exists(full_save_path):
	random_csv_df.to_csv(f'{full_save_path}{random_save_name}.csv', index=False)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#### moving images to be in same folder (bloody nightmare)
base_img_pth = '/gws/nopw/j04/dcmex/users/ezriab/processed_images/'
hvps_img_folders = os.listdir(base_img_pth+'hvps/')
ch0_img_folders = os.listdir(base_img_pth+'2ds/ch_0/')
ch1_img_folders = os.listdir(base_img_pth+'2ds/ch_1/')

## make dictionary containing file path to every single image
#dictionaries containing list of all file names
probe_file_names = ['hvps','ch0','ch1']
probe_file_extensions = ['hvps/','2ds/ch_0/','2ds/ch_1/']
probe_file_list = [hvps_img_folders, ch0_img_folders, ch1_img_folders]
full_file_dic = dict.fromkeys(probe_file_names)
## making dictionary each probe + each folder of h5 file + path to each png image
for i in range(3):
    full_file_dic[probe_file_names[i]] = {}
    for date in probe_file_list[i]:
        date_path = base_img_pth+probe_file_extensions[i]+date+'/'
        file_list = glob(date_path+'*.png')
        full_file_dic[probe_file_names[i]][date] = file_list

## looping though to get corresponding paths to images extracted
### there are multiple folders of same date, so may have to loop through each one to find the image
full_loc_file_list = []
for name in r_sample_list:
    slice_n, date_probe = name.split('_')
    day_n = date_probe[:2]
    probe = date_probe[2:]
    potential_dates = [f'2207{day_n}',f'2208{day_n}']

    folders_contain_date = []
    all_dates = list(full_file_dic[probe].keys())
    for folder_name in all_dates:
        if potential_dates[0] in folder_name or potential_dates[1] in folder_name:
            folders_contain_date.append(folder_name)
    
    for f in folders_contain_date:
        files_in_f = full_file_dic[probe][f]
        #print(files_in_f[0])
        #if name in files_in_f:
        if any(name in file_path for file_path in files_in_f):
            for individual_file in files_in_f:
                if name in individual_file:
                    full_loc_file_list.append(individual_file)
                    
            #og_image_loc = base_img_pth+probe+'/'+f +'/'+ name +'.png'
                #full_loc_file_list.append(og_image_loc)
    
## finally move that file 
for file_path in full_loc_file_list:
    if os.path.exists(file_path) and os.path.exists(full_save_path):
        shutil.copy(file_path, full_save_path)	

