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

from scipy.ndimage import convolve, label
from skimage.measure import regionprops, find_contours
from scipy.spatial import ConvexHull, distance_matrix
from scipy.spatial.distance import pdist
from skimage.morphology import remove_small_holes ## remove holes <3
from scipy.ndimage import binary_fill_holes
from skimage import measure
import tensorflow as tf

# Check if the function exists
if hasattr(tf.image, 'resize_with_crop_or_pad'):
    print("The function 'tf.image.resize_with_crop_or_pad' is available.")
else:
    print("The function 'tf.image.resize_with_crop_or_pad' is not available.")

print('loaded all!')