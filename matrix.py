from scipy.special import sph_harm as SH
from scipy.special import eval_jacobi as Plkn
import numpy as np
import math
from numpy import linalg as LA
##########################################################################################
## This file is used to generate spherical harmonics and Wigner D-functions sensing matrix
## as well as derivative of associated Legendre and Wigner d-functions
##
##
##########################################################################################


#####################################################################################################
## Spherical Harmonics Matrix
## Input :  m dimension theta and phi, bandlimited B 
## Output : Spherical harmonics matrix (A), normalization spherical harmonics matrix (normA)
##          Combination degree and order (lk), derivative of associated Legendre functions (dPlk)
#####################################################################################################

def SphericalHarmonics(theta,phi,B):
    N = B**2
    lk = np.zeros((N,2))
    m = len(theta)
    A = np.zeros((m,N), dtype = np.complex128)
    normA = np.zeros((m,N), dtype = np.complex128)
    dPlk = np.zeros((m,N))
    ###################################################################
    ## Generating combination degree and order for spherical harmonics
    ###################################################################
    idx_beg = 0
    for l in range(B):
        k = range(-l,l+1)
        idx = len(k)
        idx_end = idx_beg + idx-1
        lk[idx_beg:idx_end+1,1] = k
        lk[idx_beg:idx_end+1,0] = np.full((1,idx),l)
        idx_beg = idx_beg + idx

    #####################################################################################################
    ## Generating Spherical Harmonics Matrix and their derivative
    ## with respect to theta (Derivative of associated Legendre polynomials)
    ## d/dtheta ) = m/tan(theta) Ylk(theta,phi) + sqrt((l-k)(l+k+1))*Yl(k+1)(theta,phi)
    ##
    ## The spherical harmonics when k+1 > l will be zero because of multiplication with sqrt((l-k)(l+k+1)
    ######################################################################################################

    for ii in range(N):
        ### Generate spherical harmonics and their (column) normalization w.r.t sampling points
        A[:,ii] = SH(lk[ii,1],lk[ii,0],phi,theta)
        normA[:,ii] = A[:,ii]/LA.norm(A[:,ii])
        
        ### Spherical harmonics order > degree, just assign arbitrary since will be zero multiply with the coefficients
        if lk[ii,1] + 1 > lk[ii,0]:
            SH_lastterm = np.ones(m)
        else:
            SH_lastterm = np.exp(-1j*phi)*SH(lk[ii,1]+1,lk[ii,0],phi,theta)/LA.norm(SH(lk[ii,1]+1,lk[ii,0],phi,theta))
       
        ### Derivative of spherical harmonics w.r.t theta, or derivative of associated Legendre functions
        Plk_deriv = (lk[ii,1]/np.tan(theta))*normA[:,ii] + np.sqrt((lk[ii,0] - lk[ii,1])*(lk[ii,0]+lk[ii,1]+1))*\
                    SH_lastterm
        
        dPlk[:,ii] = np.real(Plk_deriv)
        
    
    return A, normA, lk, dPlk




#####################################################################
## Wigner D-functions Matrix
##
## Input :  m dimension theta, phi and chi, bandlimited B
## Output : Wigner D-function  matrix (A), normalization Wigner D-function matrix (normA)
##          Combination degree and orders (lkn), derivative of Wigner d-functions (dWignerd)
#####################################################################################################

def WignerDfunctions(theta,phi,chi,B):
    m = int(len(theta))
    N = B*(2*B-1)*(2*B+1)//3
    A = np.zeros((m,N),dtype = np.complex128)
    normA = np.zeros((m,N), dtype = np.complex128)
    lkn = np.zeros((N,3))
    ###################################################################
    ## Generating combination degree and orders
    ###################################################################
    idx_beg = 0
    for l in range(B):
        n = range(-l,l+1)
        k = range(-l,l+1)
        mesh_k_n = np.array(np.meshgrid(k, n))
        k_n = mesh_k_n.T.reshape(-1, 2)
        idx = len(n)**2
        idx_end = idx_beg + idx-1
        lkn[idx_beg:idx_end + 1,0] = np.full((1,idx),l)
        lkn[idx_beg:idx_end + 1,1:] = k_n 
        idx_beg = idx_beg + idx


    ############################################################################
    ## Generating Wigner D-functions matrix and derivative of Wigner d-functions
    ############################################################################
    A = np.zeros((m,N), dtype = np.complex128)
    normA = np.zeros((m,N), dtype = np.complex128)
    dWignerd = np.zeros((m,N))
    

    for ii in range(N):
        

        #########################################################################
        ## Set initial parameters
        #########################################################################
        if lkn[ii,2] >= lkn[ii,1]:
            eta = 1
        else:
            eta = (-1)**(lkn[ii,2] - lkn[ii,1])

        #########################################################################
        ## Set Normalization
        #########################################################################
        Normalization = math.sqrt((2.0*lkn[ii,0]+1)/(8.0*math.pi**2))
        mu_plus = abs(lkn[ii,1] - lkn[ii,2])
        vu_plus = abs(lkn[ii,1] + lkn[ii,2])
        s_plus = lkn[ii,0] - (mu_plus + vu_plus)/2.0
        
        Norm_Gamma = math.sqrt((math.factorial(s_plus)*math.factorial(s_plus+mu_plus+vu_plus))/\
                    (math.factorial(s_plus+mu_plus)*(math.factorial(s_plus+vu_plus))))
        ###########################################################################
        ## Generate Wigner d-functions
        ###########################################################################

        Wignerd =  Normalization*eta*Norm_Gamma*\
                    np.array(np.power(np.sin(theta/2.0),mu_plus))*np.array(np.power(np.cos(theta/2.0),vu_plus))*\
                    np.array(Plkn(s_plus,mu_plus,vu_plus,np.cos(theta)))
        #################################################################################################
        ## Generate Wigner D-functions sensing matrix and their (column) normalization
        ################################################################################################

        A[:,ii] = np.array(np.exp(-1j*lkn[ii,1]*phi))*np.array(Wignerd)*np.array(np.exp(-1j*lkn[ii,2]*chi))
        normA[:,ii] = A[:,ii]/LA.norm(A[:,ii])
        
        ###################################################################################################
        ## Calculate derivative of Wigner d-functions
        ###################################################################################################
        
        Jacobi_last = (Plkn(s_plus-1,mu_plus+1,vu_plus+1,np.cos(theta)))
        Wignerd_deriv =(-(mu_plus*np.sin(theta)**2)/(2.0*(1- np.cos(theta))) + (vu_plus*np.sin(theta)**2)/(2.0*(1 + np.cos(theta))))*normA[:,ii] - Normalization*eta*Norm_Gamma*(mu_plus + vu_plus + s_plus + 1)*0.5*np.array(np.sin(theta))*(np.array(np.power(np.sin(theta/2.0),mu_plus)))*np.array(np.power(np.cos(theta/2.0),vu_plus))\
                *(Jacobi_last/LA.norm(A[:,ii]))
        dWignerd[:,ii] = np.real(Wignerd_deriv)
        

    return A, normA, lkn , dWignerd


