from scipy.special import lpmv as asleg
from scipy.special import sph_harm as SH
from scipy.special import eval_jacobi as Plkn
import numpy as np
import math
from numpy import linalg as LA
from itertools import combinations

import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matrix import WignerDfunctions
from samplingsphere import Spiralsampling
from metric import Maxprod
########################################################
## Main
########################################################

B = 6
N = B*(2*B-1)*(2*B+1)//3
m = np.arange(2,N,1).astype(np.int64)

MaxProd = np.zeros(len(m))
LegendreBound = np.zeros(len(m))
for ii in range(len(m)):
    
    theta = Spiralsampling(m[ii])[0] #Spiral sampling implement equispaced sampling points
    phi_0 = np.zeros(m[ii])
    chi_0 = np.zeros(m[ii])

    normA = WignerDfunctions(theta,phi_0,chi_0,B)[1]
    lkn = WignerDfunctions(theta,phi_0,chi_0,B)[2]
    
    MaxProd[ii] = Maxprod(normA,lkn)

    ##################################################
    PB1 = asleg(0,B-1,np.cos(theta))
    PB3 = asleg(0,B-3,np.cos(theta))
    LegendreBound[ii] = abs(np.dot(PB1/LA.norm(PB1),PB3/LA.norm(PB3)))
    print("Samples %s, Maximum Product  %s,  and Legendre Bound %s" %(m[ii], MaxProd[ii], LegendreBound[ii]))


Bound = np.ceil(((B + 2)**2)/10 + 1)

plt.figure(1)
plt.axvline(Bound, 0, 1, label='m = ((B+2)**2)/10 + 1')
plt.legend()
plt.plot(m,LegendreBound,color ='b')
plt.plot(m,MaxProd,color = 'r')
plt.xlim(m[0],m[len(m)-1])
plt.ylim(0,1)
plt.xlabel('Samples (m)')
plt.ylabel('Product')
plt.show()




