## code for Alan to view hvps images - h5 files
# very simple tbh, but will have lot of small particles

import xarray as xr
import matplotlib
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


## location of hvps files - on jasmin
path = '/gws/nopw/j04/dcmex/users/ezriab/raw_h5/hvps/'
# e.g. one file 
file = 'Export_base220730152928.h5'
# open file
f2ds = h5py.File(path+file,'r')


# structure of the file
#list(f2ds.keys()) ## ['ImageData', 'ImageTimes']

ds_image = f2ds['ImageData'] # shape (128, 200000) --> 128 pixels width, 200000 length
ds_time = f2ds['ImageTimes'] # shape (100000, 3) --> (rows, columns) related to time

# make time variable -> 3 seperate columns
og_t_xr = xr.Dataset({'ImageTimes': (('data', 'time_vars'),ds_time)})
sec_since = og_t_xr['ImageTimes'][:,0] # seconds since midnight UTC
pixel_slice = og_t_xr['ImageTimes'][:,1] # number of slices of pixel per image (contain -1, to fill)
# we can use pixel slice to correcly divide up data ^ the index to call f2ds['ImageData']
# these slices have been pre-determined by Jonny + algorithm he has written
### basically pixel slice gives a rough idea of particle location

bit_time = og_t_xr['ImageTimes'][:,2] # 32 bit (instrument things) - not useful

# do cumulative sum of pixel slices -> can use this as index to slice + extract single crystals
pix_sum = pixel_slice.cumsum(dim='data', dtype ='int')

# shove together into a useful xarray 
time_xr = xr.Dataset({
    'sec_since': sec_since,
    'pixel_slice': pixel_slice,
    'bit_time': bit_time,
    'pix_sum': pix_sum})
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## do some quick cleaning - get rid of slices that have pixel slice of < 4
diff = np.diff(time_xr['pix_sum'][:].values) # this is finding the difference between the elements of pix_sum
selected_values = time_xr['pix_sum'][:-1][diff > 4] # this is selecting the adjacent files in which have pixels > 4 length

## as the image is a 2D array, we can view it in matplotlib <3
i = 10 # random crystal

im_s_idx = int(time_xr['pix_sum'][selected_values[i]])
im_e_idx = int(time_xr['pix_sum'][selected_values[i+1]]) ## may want to add more - to look at a bigger slice

one_crystal = f2ds['ImageData'][:,im_s_idx:im_e_idx] # extract 1 crystal

## this will plot the crystal
plt.imshow(one_crystal, cmap='gray')
plt.axis('off')  # Turn off axis labels
## (I'm not sure how to display images directly from scripts in linux)
# but you can save the plot, then view the png like normal?
'''
plt.savefig("hvps_crystal.png")
'''


