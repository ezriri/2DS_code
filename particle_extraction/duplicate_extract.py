## hello this is a script to make a new df with the duplicated stats (but not images) make new stats / image file
## ive already got a csv with measured stats + start indexs etc, this is a case of using the start / end indexes to make new image associated with it. 

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
import csv



## where is the duplicate csv
dup_csv_pth = '/gws/nopw/j04/dcmex/users/ezriab/processed_stats/silly_duplicates/duplicates_df.csv'
base_h5_pth = '/gws/nopw/j04/dcmex/users/ezriab/raw_h5/2ds/'

