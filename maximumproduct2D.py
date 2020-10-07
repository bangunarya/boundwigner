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

Blist = np.arange(4,14,2)
B = np.max(Blist)
N = 50#B*(2*B-1)*(2*B+1)//3
m = np.arange(7,N,4).astype(np.int64)

MaxProd2D = np.zeros((len(Blist),len(m)))
LegendreBound2D = np.zeros((len(Blist),len(m)))
#minsample = np.zeros(len(Blist))
Ph_trans = np.zeros((len(Blist),len(m)))
for jj in range(len(Blist)):
    for ii in range(len(m)):
    
        theta = Spiralsampling(m[ii])[0] #Spiral sampling implement equispaced sampling points
        phi_0 = np.zeros(m[ii])
        chi_0 = np.zeros(m[ii])

        normA = WignerDfunctions(theta,phi_0,chi_0,Blist[jj])[1]
        lkn = WignerDfunctions(theta,phi_0,chi_0,Blist[jj])[2]
    
        MaxProd2D[jj,ii] = Maxprod(normA,lkn)

        ##################################################
        PB1 = asleg(0,Blist[jj]-1,np.cos(theta))
        PB3 = asleg(0,Blist[jj]-3,np.cos(theta))
        LegendreBound2D[jj,ii] = abs(np.dot(PB1/LA.norm(PB1),PB3/LA.norm(PB3)))
        
        Ph_trans[jj,ii] = 1 - abs(LegendreBound2D[jj,ii] - MaxProd2D[jj,ii])
        if Ph_trans[jj,ii] < 0.99:
            Ph_trans[jj,ii] = 0
        
        print("Bandlimited %s, Samples %s, Maximum Product  %s,  and Legendre Bound %s" %(Blist[jj], m[ii], MaxProd2D[jj,ii], LegendreBound2D[jj,ii]))


minsample = np.ceil(((Blist + 3)**2)/10 + 1)

plt.figure(1)
plt.imshow(Ph_trans.T, aspect = 'auto', cmap = cm.jet, origin ='lower', extent = [Blist[0],Blist[-1],m[0],m[-1]],vmax=np.amax(Ph_trans),vmin=np.amin(Ph_trans))
plt.colorbar(orientation='vertical')
plt.plot(Blist, minsample, linewidth = 2, color ='k',label = '$m = (B + 2)^2/10 + 1$')
plt.legend()
plt.xlim(Blist[0],Blist[-1])
plt.ylim(m[0],m[-1])
plt.xlabel('Bandwidth (B)')
plt.ylabel('Samples (m)')
plt.show()



