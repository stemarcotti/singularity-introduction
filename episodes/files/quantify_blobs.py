# CC-BY-SA-4.0 license
# please see https://github.com/RMS-DAIM/introduction-to-image-analysis for details

# usage
# to use this script call from the command line the following: `python quantify_blobs.py path_to_images`

# import libraries
import os
import sys
import numpy as np
from skimage import io
from skimage import filters
from skimage import measure
import matplotlib.pyplot as plt
import pandas as pd

# read all .tif files in folder
path = sys.argv[1]
dir_list = os.listdir(path)
im_read = io.imread(path+'*.tif')

# transpose image dimensions
im_all = np.transpose(im_read, (0, 3, 1, 2))

# initialise output
obj_count = []
props_df = pd.DataFrame()

# set up for loop for all the images available in the folder
for im_idx in range(im_all.shape[0]):
    # open image (only ch0)
    im = im_all[im_idx,0,]
    # filter with Gaussian
    im_gauss = filters.gaussian(im, sigma=5)
    # threshold with Otsu
    thresh = filters.threshold_otsu(im_gauss)
    im_thresh = im_gauss >= thresh
    # label mask
    labels = measure.label(im_thresh)
    # count objects
    obj_count = np.append(obj_count,labels.max())
    # measure properties
    props = measure.regionprops_table(labels, im, properties=['area', 'eccentricity'])
    props = pd.DataFrame(props)
    # add image ID and object ID
    props['image_ID'] = dir_list[im_idx]
    props['object_ID'] = props.index+1
    # add to output dataframe
    props_df = pd.concat([props_df, pd.DataFrame(props)], ignore_index=True)

# save dataframe as excel file
props_df.to_excel(path+'results.xlsx')

# save label images
for im_idx in range(im_all.shape[0]):
    # open image (only ch0)
    im = im_all[im_idx,0,]
    # filter with Gaussian
    im_gauss = filters.gaussian(im, sigma=5)
    # threshold with Otsu
    thresh = filters.threshold_otsu(im_gauss)
    im_thresh = im_gauss >= thresh
    # label mask
    labels = measure.label(im_thresh)
    # save
    file_name = path+dir_list[im_idx][0:-4]+'_labels.tif'
    io.imsave(file_name, labels)
