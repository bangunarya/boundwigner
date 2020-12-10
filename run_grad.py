#########################################################################
### This file is used to generate Figure 4 in the paper
###
### Tight bounds on the mutual coherence of sensing matrices
### for Wigner D-functions on regular grid - Arya Bangun, Arash Behboodi
### and Rudolf Mathar 
###
### Created by Arya Bangun
#########################################################################
import matplotlib.pyplot as plt
import numpy as np
import os
from grad_algorithms import GradientAlgorithms
from metric import BoundCoherence, params
from matrix import Matrix
from samplingsphere import SamplingPoints
from storedata import StoreData
types = ['snf','all']
sampling = ['gradient', types[1]]

B = 16
N, col_comb = params(types, B)
m = np.arange(50,N,35).astype(np.int64) # Samples
 
## Parameters Gradient
eps = 1e-4
max_iter = 100

## Store data
storepath = os.path.join(os.getcwd(), types[0] + types[1])
savedata = StoreData(m = m, N = N, path = storepath )

## Run Gradient

for ii in range(len(m)):
    
    ## Generate samples
    angles = SamplingPoints(m = m[ii],types = sampling).angles
    
    ## Generate Matrix
    matrix = Matrix(B = B,
                    types = types,
                    angles = angles)
    ## Gradient
    gr_algo = GradientAlgorithms(matrix = matrix, eps = eps,
                                 max_iter = max_iter, col_comb = col_comb)
    
    ## Generate bound
    Bound = BoundCoherence(m[ii],N,B) # Generate the Bound in the main theorem
    
    
    ## Store Data
    savedata.coh_angle_handle(ii = ii, gr_algo = gr_algo, bound = Bound)
    