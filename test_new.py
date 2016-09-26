# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 19:55:47 2016

@author: archie
"""

import numpy as np
import math
import csv
from cvxpy import *
import scipy as scipy
import gc
def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

a = np.arange(36)
print a
a2d = np.reshape(a, (6,6) )
a = np.copy( np.reshape(blockshaped(a2d, k, k),(1,36)) )
print a

feature = 36
multiply_factor = 4
print "d type"
D = np.zeros((feature,feature))

for row_index in range(feature):
    offset = multiply_factor * row_index
    for column_index in xrange(multiply_factor):
        D[row_index][offset + column_index]

