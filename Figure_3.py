#########################################################################
### This file is used to generate Figure 3 in the paper
###
### Tight bounds on the mutual coherence of sensing matrices
### for Wigner D-functions on regular grid - Arya Bangun, Arash Behboodi
### and Rudolf Mathar 
###
### Created by Arya Bangun
#########################################################################

import matplotlib.pyplot as plt
import numpy as np
from gradmethod import gradientdescent, ADAM, AdaDelta, AdaGrad
from metric import BoundCoherence

update = 'partial' # We want to update only angle on azimuth
type_matrix = 'Wigner' # Matrix Spherical Harmonics or Wigner D-functions
B = 4 # Bandwidth of Wigner D-functions
N = B*(2*B-1)*(2*B+1)//3 # Column dimension of the matrix

m = np.arange(17,N,8).astype(np.int64) #samples
idx = 3 # index to get output variable from gradient method


## Preallocation
graddes = np.zeros(len(m), dtype = np.float64);
adam = np.copy(graddes)
adadelta = np.copy(graddes)
adagrad = np.copy(graddes)
legbound = np.copy(graddes)
welchbound = np.copy(graddes)


for ii in range(len(m)):

    Bound = BoundCoherence(m[ii],N,B)
    ang_all_graddes = gradientdescent(type_matrix, m[ii], B, Bound, update)
    print("GrDe     -- Samples %s, Coherence  %s,  Welch Bound %s, and Legendre Bound %s" %(m[ii],ang_all_graddes[idx],Bound[0], Bound[1])) 
    ang_all_adam = ADAM(type_matrix, m[ii], B, Bound, update)
    print("ADAM     -- Samples %s, Coherence  %s,  Welch Bound %s, and Legendre Bound %s" %(m[ii],ang_all_adam[idx],Bound[0], Bound[1]))
    ang_all_adadelta = AdaDelta(type_matrix, m[ii], B, Bound, update)
    print("AdaDelta -- Samples %s, Coherence  %s,  Welch Bound %s, and Legendre Bound %s" %(m[ii],ang_all_adadelta[idx],Bound[0], Bound[1])) 
    ang_all_adagrad = AdaGrad(type_matrix, m[ii], B, Bound, update)
    print("AdaGrad  -- Samples %s, Coherence  %s,  Welch Bound %s, and Legendre Bound %s" %(m[ii],ang_all_adagrad[idx],Bound[0], Bound[1]))

    ### File
    graddes[ii] = ang_all_graddes[idx]
    adam[ii] = ang_all_adam[idx]
    adadelta[ii] = ang_all_adadelta[idx]
    adagrad[ii] = ang_all_adagrad[idx]
    legbound[ii] = Bound[1]
    welchbound[ii] = Bound[0]



plt.figure(1)
plt.plot(m,graddes, color = 'r', marker = 'o', linestyle = '--',linewidth = 2, label = 'Gradient Descent in Algorithm ')
plt.grid(True)
plt.plot(m,adadelta, color = 'k', marker = 'x', linewidth = 2, linestyle = '--', label = 'Ada Delta')
plt.plot(m,adagrad, color = 'c', marker = 's', linestyle = '--', linewidth = 2, label = 'Ada Grad')
plt.plot(m,adam, color = 'g', marker = '*', linewidth = 2, linestyle = '--', label = 'Adam')
plt.plot(m,legbound, color = 'b', marker = 's', linewidth = 2, linestyle = '--', label = 'Bound in Theorem 1')
plt.plot(m,welchbound, color = 'tab:brown', marker = '^', linewidth = 2, linestyle = '--', label = 'Welchbound ')
plt.xlim(m[0],m[len(m)-1])
plt.ylim(0,1)
plt.legend()
plt.xlabel('Samples (m)')
plt.ylabel('Coherence')
plt.show()

#np.savetxt('MaxProd2D_4042.txt', MaxProd2D)
#np.savetxt('LegendreBound2D_4042.txt', LegendreBound2D)
#np.savetxt('Ph_trans_4042.txt', Ph_trans)

