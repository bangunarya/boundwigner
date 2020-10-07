import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib as numat
 
import matplotlib.cm as cm


#######################################################################################
### Plot maximum product and Legendre bound
### 
#######################################################################################
Blist = np.arange(4,48,2)
B = np.max(Blist)
N = 300
m = np.arange(7,N,2).astype(np.int64)



minsample = ((Blist + 2)**2/10 + 1)
Ph_trans = np.loadtxt("PhTransTotal.txt", delimiter=" ")
LegBound = np.loadtxt("LegendreBound2DTotal.txt", delimiter = " ")
MaxProd = np.loadtxt("MaxProdTotal.txt", delimiter = " ")
#print(LegBound.shape)
#print(MaxProd.shape)
#print(Ph_trans.shape)
Diff_Mat = np.abs(LegBound-MaxProd)# np.zeros((MaxProd.T.shape))
#idx = Ph_trans < 0.9
Ph_trans_thres = 1- Diff_Mat#np.zeros(Ph_trans.shape)
idx = Ph_trans < 0.99
Ph_trans_thres[idx] = 0
plt.figure(1)
plt.imshow(Ph_trans_thres.T, aspect = 'auto', cmap = cm.jet, origin ='lower', extent = [Blist[0],Blist[len(Blist)-1],m[0],m[len(m)-1]],vmax = 1,vmin=0)
plt.colorbar(orientation='vertical') #.set_label('Difference', rotation = 270, labelpad = 1)
plt.plot(Blist,minsample,linewidth = 4, color ='k',label = '$(m = (B + 2)^2/10 + 1$')
#plt.legend()
plt.xlim(Blist[0],Blist[len(Blist)-1])
plt.ylim(m[0],m[len(m)-1])
plt.xlabel('Bandwidth (B)')
plt.ylabel('Samples (m)')

