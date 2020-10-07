#########################################################################
### This file is used to generate Figure 1 in the paper
###
### Tight bounds on the mutual coherence of sensing matrices
### for Wigner D-functions on regular grid - Arya Bangun, Arash Behboodi
### and Rudolf Mathar 
###
### Created by Arya Bangun
#########################################################################

from scipy.special import lpmv as asleg
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm

m = np.arange(3,3003,4)
l = np.arange(2,202,2)
R = np.zeros((len(m),len(l)))
idx_row = range(len(m))
idx_col = range(len(l))
plt.rcParams.update({'font.size': 14})

for ii in idx_row:
    for jj in idx_col:
        x  = np.linspace(-1,1,m[ii])
        Pl = asleg(0,l[jj],x)
        Pl_est  = 1 + (l[jj]*(l[jj]+1))/(6.0*(m[ii]-1))
        #R[ii,jj] = sum(Pl) - Pl_est
        if (sum(Pl) - Pl_est) < -0.5:
            R[ii,jj] = -0.5
        else:
            R[ii,jj] = sum(Pl) - Pl_est

 #   print('Samples  ' +  str(m[ii]))

min_sample = (l**2)/10.0 + 1

plt.figure(1)
plt.imshow(R, aspect = 'auto', cmap = cm.jet, origin ='lower', extent = [0,200,0,3001],vmax=np.amax(R),vmin=np.amin(R))
#plt.colorbar(orientation='vertical')
plt.colorbar(orientation='vertical') #.set_label('Residual', rotation=270)
plt.plot(l,min_sample,linewidth = 2, color ='k',label = 'm = (l + 1)^2/10 + 1')
plt.legend()
plt.xlim(2,200)
plt.ylim(3,3003)
plt.xlabel('degree (i)')
plt.ylabel('sample (m)')
plt.show()
