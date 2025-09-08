from tifffile import imsave
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import time as Time
import scipy.stats
from scipy import stats
from scipy.ndimage import label
from skimage.filters import threshold_otsu
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
from scipy import stats
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr
import random as rd 
from scipy.signal import fftconvolve
from scipy import ndimage
from rasterio.transform import from_origin
from pyproj import Proj, transform
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
from scipy.ndimage import label, distance_transform_edt, sobel
from scipy.special import expit  # Pour la fonction logistique
from scipy.ndimage import binary_dilation
from scipy.spatial import ConvexHull
from skimage.draw import polygon
import matplotlib.pyplot as plt
from collections import Counter
import random


def filter4prevalence(binary_list):

    """
    permits to select randomly inside a validation dataset, or a training dataset, the same number of positives and negatives.
    
    Input:

    binary_list: (list) list of zeros and ones, symbolizing the succesion of positives "1" and negatives "0".

    Output:

    position_list: (list) list of indexes to keep inside the input list.
    """
    #print("imput binary list =",binary_list)
    position_list = list(np.arange(0,len(binary_list),1)) 
    # Count the number of ones and zeros
    num_ones = binary_list.count(1)
    num_zeros = binary_list.count(0)

    # Calculate how many zeros need to be removed
    zeros_to_remove = num_zeros - num_ones

    # If zeros need to be removed
    if zeros_to_remove > 0:
        # Find all the indices of zeros in the binary list
        zero_indices = [i for i, x in enumerate(binary_list) if x == 0]

        # Randomly choose indices to remove from the zero_indices list
        indices_to_remove = random.sample(zero_indices, zeros_to_remove)
        indices_to_remove.sort(reverse=True)

        # Remove the elements at the selected indices from both lists
        for index in indices_to_remove:
            del binary_list[index]
            del position_list[index]
    
    if zeros_to_remove <=0:
        ones_to_remove = -zeros_to_remove
        # Find all the indices of ones in the binary list
        ones_indices = [i for i, x in enumerate(binary_list) if x == 1]

        # Randomly choose indices to remove from the zero_indices list
        indices_to_remove = random.sample(ones_indices, ones_to_remove)
        indices_to_remove.sort(reverse=True)

        # Remove the elements at the selected indices from both lists
        for index in indices_to_remove:
            del binary_list[index]
            del position_list[index]

    # Output the modified lists
    #print("Modified binary list:", binary_list)
    #print("Modified position list:", position_list)
    
    return position_list

print("validation tools imported")