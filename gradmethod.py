#############################################################################################################################
### This file contains the implementation of several gradient descent algorithms, simple gradient descent and ADAM
###
###
#############################################################################################################################

from matrix import SphericalHarmonics, WignerDfunctions
from metric import Coherence, BoundCoherence
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from itertools import combinations
import numpy.matlib as npmat
from gradient import DerivProdSH, DerivProdWigner



def gradientdescent(type_matrix, m, B, Bound, update):
    #################################################################################################
    ## This is simple implementation of gradient descent algorithm
    ## input : type_matrix 'SH' for spherical harmonics or 'Wigner' for Wigner D-functions
    ##         m number of sampling points, bandlimited B, Coherence bound defined in the paper
    ##         update for phi and chi
    ## output : sampling points theta, phi with respect to the choice of type matrix and update
    #################################################################################################
    eps = 1e-8 #error tolerance
    gamma = 0.5 # stepsize
    max_iter = 500 #iteration
   
    #################################################################################################
    ## This condition if we choose matrix spherical harmonics and update  phi
    ## 
    ################################################################################################
    if type_matrix == 'SH':
        LB = Bound[1]
        iterate = 0
        Coh = 1
        theta = np.arccos(np.linspace(-1,1,m)) #We fix theta equispaced
        phi = np.random.rand(m)*2.0*np.pi #Initial guess random sampling points on phi
        
        ##### Gradient SH
        gr_phi = DerivProdSH(theta,phi,B,update)

        while iterate < max_iter and abs(Coh - LB) > eps:
            
            iterate += 1
            ##### Update
            phi = phi - gamma*(gr_phi)

            ##### Update Gradient SH
            gr_phi = DerivProdSH(theta,phi,B,update)
            
            ### Calculate the coherence and compare with the Legendre bound
           
            normA = SphericalHarmonics(theta,phi,B)[1]
            Coh = Coherence(normA)
        
            
           # print("Number of samples %s with Coherence %s and Bound %s" % (m, Coh, LB))
        return theta, phi, Coh


    #################################################################################################
    ## This condition if we choose matrix Wigner and update  phi and chi
    ##
    #################################################################################################
    else:
        LB = Bound[1]
        iterate = 0
        Coh = 1
        theta = np.arccos(np.linspace(-1,1,m)) # Fix theta variable with equispaced sampling
        phi = np.random.rand(m)*2.0*np.pi # Initialize phi, since we want to update partial phi and chi
        chi = np.random.rand(m)*2.0*np.pi # Initialize chi, since we want to update partial phi and chi

        ##### Gradient Wigner
        gr_phi = DerivProdWigner(theta,phi,chi,B,update)[0]
        gr_chi = DerivProdWigner(theta,phi,chi,B,update)[1]

        while iterate < max_iter and abs(Coh - LB) > eps:    
                                                                                   
            ##### Update
            phi = phi - gamma*gr_phi
            chi = chi - gamma*gr_chi
            
            ##### Update Gradient Wigner
            gr_phi = DerivProdWigner(theta,phi,chi,B,update)[0]
            gr_chi = DerivProdWigner(theta,phi,chi,B,update)[1]

            ### Calculate the coherence and compare with the Legendre bound
            normA = WignerDfunctions(theta,phi,chi,B)[1]
            Coh = Coherence(normA)
            iterate += 1
           # print("Number of samples %s with Coherence %s and Bound %s" % (m, Coh, LB))
            
        return theta, phi, chi, Coh




######################################################################################
### Implementation of ADAM; A METHOD FOR STOCHASTIC OPTIMIZATION
### https://arxiv.org/pdf/1412.6980.pdf
###
######################################################################################
## input : type_matrix 'SH' for spherical harmonics or 'Wigner' for Wigner D-functions
##         m number of sampling points, bandlimited B, Coherence bound defined in the paper
##         update  for phi
## output : sampling points theta, phi with respect to the choice of type matrix and update
#################################################################################################


def ADAM(type_matrix, m, B, Bound, update):
    gamma = 0.05 ## stepsize
    beta1 = 0.9 ## ADAM parameter
    beta2 = 0.999 ## ADAM parameter
    max_iter = 500 ## Maximum Iteration
    eps = 1e-8 ## Error tolerance
    #theta_all = np.zeros((m,max_iter))
    #phi_all = np.zeros((m,max_iter))
    #chi_all = np-zeros((m,max_iter))
    
   
    #################################################################################################
    ## This condition if we choose matrix spherical harmonics and update  phi
    ## 
    ################################################################################################

    if type_matrix == 'SH':
        LB = Bound[1]
        iterate = 0
        Coh = 1
        theta = np.arccos(np.linspace(-1,1,m)) ## Fix sampling points on theta, equispaced
        phi = np.random.rand(m)*2.0*np.pi ## Initialize random guess
       
        ##### Update Gradient SH for phi
        gr_phi = DerivProdSH(theta,phi,B,update)


        mm_phi = np.zeros(m)
        v_phi = np.zeros(m)
        
        while iterate < max_iter and abs(Coh - LB) > eps:
            
            ###########################################################
            ### Update for phi
            ##########################################################

            iterate += 1
            
            ## Update biased 1st moment estimate
            mm_phi = beta1*mm_phi + (1.0 - beta1)*gr_phi
            ## Update biased 2nd raw moment estimate
            v_phi = beta2*v_phi + (1.0 - beta2)*np.power(gr_phi,2)

            ## Compute bias-corrected 1st moment estimate
            mHat_phi = mm_phi/(1.0 - np.power(beta1,iterate))
            ## Compute bias-corrected 2nd raw moment estimate
            vHat_phi = v_phi/(1.0 - np.power(beta2,iterate))
             
            ## Update decision variables
            phi = phi - gamma*mHat_phi/(np.sqrt(vHat_phi) + eps)
            
            ##### Update Gradient SH for phi
            gr_phi = DerivProdSH(theta,phi,B,update)
            
            ### Calculate the coherence and compare with the Legendre bound

            normA = SphericalHarmonics(theta,phi,B)[1]
            Coh = Coherence(normA)

            

           # print("Number of samples %s with Coherence %s and Bound %s" % (m, Coh, LB))
        return theta, phi, Coh

    #################################################################################################
    ## This condition if we choose matrix Wigner D-functions  and update  phi,chi
    ## 
    ################################################################################################

    else:
        LB = Bound[1]
        iterate = 0
        Coh = 1
        theta = np.arccos(np.linspace(-1,1,m)) ## Fix sampling on theta with equispaced
        phi = np.random.rand(m)*2.0*np.pi ## Initialize random guess on phi
        chi = np.random.rand(m)*2.0*np.pi ## Initialize random guess on chi
       
        ##### Gradient Wigner phi and chi 
        gr_phi = DerivProdWigner(theta,phi,chi,B,update)[0]
        gr_chi = DerivProdWigner(theta,phi,chi,B,update)[1]
        
        mm_phi = np.zeros(m)
        v_phi = np.zeros(m)
        mm_chi = np.zeros(m)
        v_chi = np.zeros(m)

        while iterate < max_iter and abs(Coh - LB) > eps:
            
            ###########################################################
            ### Update for phi and chi
            ##########################################################

            iterate += 1
            
            ## Update biased 1st moment estimate
            mm_phi = beta1*mm_phi + (1.0 - beta1)*gr_phi
            mm_chi = beta1*mm_chi + (1.0 - beta1)*gr_chi
            ## Update biased 2nd raw moment estimate
            v_phi = beta2*v_phi + (1.0 - beta2)*np.power(gr_phi,2)
            v_chi = beta2*v_chi + (1.0 - beta2)*np.power(gr_chi,2)
            ## Compute bias-corrected 1st moment estimate
            mHat_phi = mm_phi/(1.0 - np.power(beta1,iterate))
            mHat_chi = mm_chi/(1.0 - np.power(beta1,iterate))
            ## Compute bias-corrected 2nd raw moment estimate
            vHat_phi = v_phi/(1.0 - np.power(beta2,iterate))
            vHat_chi = v_chi/(1.0 - np.power(beta2,iterate))
            ## Update decision variables
            phi = phi - gamma*mHat_phi/(np.sqrt(vHat_phi) + eps)
            chi = chi - gamma*mHat_chi/(np.sqrt(vHat_chi) + eps)
            
            ##### Update Gradient Wigner phi and chi
            gr_phi = DerivProdWigner(theta,phi,chi,B,update)[0]
            gr_chi = DerivProdWigner(theta,phi,chi,B,update)[1]
            
            normA = WignerDfunctions(theta,phi,chi,B)[1]
            Coh = Coherence(normA)
            
           # print("Number of samples %s with Coherence %s and Bound %s" % (m, Coh, LB))

        return theta, phi, chi, Coh



########################################################
##  
## ADADELTA: An Adaptive Learning Rate Method
#########################################################





def AdaDelta(type_matrix, m, B, Bound, update):

    beta = 0.95 ## AdaDelta parameter
    max_iter = 500 ## Maximum Iteration
    eps = 1e-8 ## Error tolerance
    #theta_all = np.zeros((m,max_iter))
    #phi_all = np.zeros((m,max_iter))
    #chi_all = np-zeros((m,max_iter))
    
   
    #################################################################################################
    ## This condition if we choose matrix spherical harmonics and update  phi
    ## 
    ################################################################################################

    if type_matrix == 'SH':
        LB = Bound[1]
        iterate = 0
        Coh = 1
        theta = np.arccos(np.linspace(-1,1,m)) ## Fix sampling points on theta, equispaced
        phi = np.random.rand(m)*2.0*np.pi ## Initialize random guess
       
        ##### Update Gradient SH for phi
        gr_phi = DerivProdSH(theta,phi,B,update)

 
        acculGrad = np.zeros(m) # accumulated gradients
        acculDelta = np.zeros(m) # accumulated updates
        
        while iterate < max_iter and abs(Coh - LB) > eps:
            
            ###########################################################
            ### Update for phi
            ##########################################################

            iterate += 1
            
            ## Update accumulated gradients
            acculGrad = beta*acculGrad + (1.0 - beta)*gr_phi**2
            ## Calculate update
            dCurrent = -(np.sqrt(acculDelta + eps)/np.sqrt(acculGrad + eps))*gr_phi

            ## Update accumulated updates
            acculDelta = beta*acculDelta + (1.0 - beta)*dCurrent**2
            
            ## Update decision variables
            phi = phi  + dCurrent
            
            ##### Update Gradient SH for phi
            gr_phi = DerivProdSH(theta,phi,B,update)
            
            ### Calculate the coherence and compare with the Legendre bound

            normA = SphericalHarmonics(theta,phi,B)[1]
            Coh = Coherence(normA)

            

           # print("Number of samples %s with Coherence %s and Bound %s" % (m, Coh, LB))
        return theta, phi, Coh

    #################################################################################################
    ## This condition if we choose matrix Wigner D-functions  and update  phi,chi
    ## 
    ################################################################################################

    else:
        LB = Bound[1]
        iterate = 0
        Coh = 1
        theta = np.arccos(np.linspace(-1,1,m)) ## Fix sampling on theta with equispaced
        phi = np.random.rand(m)*2.0*np.pi ## Initialize random guess on phi
        chi = np.random.rand(m)*2.0*np.pi ## Initialize random guess on chi
       
        ##### Gradient Wigner phi and chi 
        gr_phi = DerivProdWigner(theta,phi,chi,B,update)[0]
        gr_chi = DerivProdWigner(theta,phi,chi,B,update)[1]
        


 
        acculGradphi = np.zeros(m) # accumulated gradients
        acculDeltaphi = np.zeros(m) # accumulated updates
        acculGradchi = np.zeros(m) # accumulated gradients
        acculDeltachi = np.zeros(m) # accumulated updates
       
        
        while iterate < max_iter and abs(Coh - LB) > eps:
            
            ###########################################################
            ### Update for phi
            ##########################################################

            iterate += 1
            
            ## Update accumulated gradients
            acculGradphi = beta*acculGradphi + (1.0 - beta)*gr_phi**2
            acculGradchi = beta*acculGradchi + (1.0 - beta)*gr_chi**2
            ## Calculate update
            dCurrentphi = -(np.sqrt(acculDeltaphi + eps)/np.sqrt(acculGradphi + eps))*gr_phi
            dCurrentchi = -(np.sqrt(acculDeltachi + eps)/np.sqrt(acculGradchi + eps))*gr_chi

            ## Update accumulated updates
            acculDeltaphi = beta*acculDeltaphi + (1.0 - beta)*dCurrentphi**2
            acculDeltachi = beta*acculDeltachi + (1.0 - beta)*dCurrentchi**2
            
            ## Update decision variables
            phi = phi  + dCurrentphi
            chi = chi + dCurrentchi
            ##### Update Gradient SH for phi
            gr_phi = DerivProdWigner(theta,phi,chi,B,update)[0]
            gr_chi = DerivProdWigner(theta,phi,chi,B,update)[1]
            ### Calculate the coherence and compare with the Legendre bound

            normA = WignerDfunctions(theta,phi,chi,B)[1]
            Coh = Coherence(normA)

            

           # print("Number of samples %s with Coherence %s and Bound %s" % (m, Coh, LB))
        return theta, phi, chi, Coh
      




####################################################################################
## AdaGrad
## Adaptive Subgradient Methods for Online Learning and Stochastic Optimization 
####################################################################################





def AdaGrad(type_matrix, m, B, Bound, update):
    gamma = 0.05 ## stepsize
    max_iter = 500 ## Maximum Iteration
    eps = 1e-8 ## Error tolerance
    #theta_all = np.zeros((m,max_iter))
    #phi_all = np.zeros((m,max_iter))
    #chi_all = np-zeros((m,max_iter))
    
   
    #################################################################################################
    ## This condition if we choose matrix spherical harmonics and update  phi
    ## 
    ################################################################################################

    if type_matrix == 'SH':
        LB = Bound[1]
        iterate = 0
        Coh = 1
        theta = np.arccos(np.linspace(-1,1,m)) ## Fix sampling points on theta, equispaced
        phi = np.random.rand(m)*2.0*np.pi ## Initialize random guess
       
        ##### Update Gradient SH for phi
        gr_phi = DerivProdSH(theta,phi,B,update)

 
        histograd = np.zeros(m) # historical gradients
        
        while iterate < max_iter and abs(Coh - LB) > eps:
            
            ###########################################################
            ### Update for phi
            ##########################################################

            iterate += 1
            
            ## Update historical gradients
            histograd = histograd + gr_phi**2
            
            ## Update decision variables
            phi = phi - gamma*gr_phi/(np.sqrt(histograd) + eps)
            
            ##### Update Gradient SH for phi
            gr_phi = DerivProdSH(theta,phi,B,update)
            
            ### Calculate the coherence and compare with the Legendre bound

            normA = SphericalHarmonics(theta,phi,B)[1]
            Coh = Coherence(normA)

            

           # print("Number of samples %s with Coherence %s and Bound %s" % (m, Coh, LB))
        return theta, phi, Coh

    #################################################################################################
    ## This condition if we choose matrix Wigner D-functions  and update  phi,chi
    ## 
    ################################################################################################

    else:
        LB = Bound[1]
        iterate = 0
        Coh = 1
        theta = np.arccos(np.linspace(-1,1,m)) ## Fix sampling on theta with equispaced
        phi = np.random.rand(m)*2.0*np.pi ## Initialize random guess on phi
        chi = np.random.rand(m)*2.0*np.pi ## Initialize random guess on chi
       
        ##### Gradient Wigner phi and chi 
        gr_phi = np.array(DerivProdWigner(theta,phi,chi,B,update)[0])
        gr_chi = np.array(DerivProdWigner(theta,phi,chi,B,update)[1])
        
        histogradphi = np.zeros(m) # historical gradients
        histogradchi = np.zeros(m)
         
        while iterate < max_iter and abs(Coh - LB) > eps:
            
            ###########################################################
            ### Update for phi
            ##########################################################

            iterate += 1
            
            ## Update historical gradients
            histogradphi = histogradphi + np.array(gr_phi)**2
            histogradchi = histogradchi + np.array(gr_chi)**2
            
            ## Update decision variables
            phi = phi - gamma*np.array(gr_phi)/(np.sqrt(histogradphi) + eps)
            chi = chi - gamma*np.array(gr_chi)/(np.sqrt(histogradchi) + eps)
            
            ##### Update Gradient SH for phi
            gr_phi = np.array(DerivProdWigner(theta,phi,chi,B,update)[0])
            gr_chi = np.array(DerivProdWigner(theta,phi,chi,B,update)[1])
            
            ### Calculate the coherence and compare with the Legendre bound

            normA = WignerDfunctions(theta,phi,chi,B)[1]
            Coh = Coherence(normA)
            
            

           # print("Number of samples %s with Coherence %s and Bound %s" % (m, Coh, LB))
        return theta, phi,chi, Coh
 
      





