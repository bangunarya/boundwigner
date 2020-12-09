#############################################################################################################################
### This file contains the implementation of several gradient descent algorithms, simple gradient descent and ADAM
###
###
#############################################################################################################################

from matrix import Matrix#SphericalHarmonics, WignerDfunctions
from metric import Coherence, BoundCoherence
import numpy as np
#np.seterr(divide='ignore', invalid='ignore')
from itertools import combinations
#import numpy.matlib as npmat
#from gradient import DerivProdSH, DerivProdWigner
from gradient import Gradient

class GradientAlgorithms:

            
    def __init__(self, matrix, eps, max_iter, col_comb):
        self.matrix = matrix
        self.eps = eps ## error_tolerance
        self.col_comb = col_comb
        self.max_iter = max_iter ## Maximum iteration
        
        self.graddescent()
        self.adam()
        self.adadelta()
        self.adagrad()
     
    def graddescent(self): 
        #################################################################################################
        ## This is simple implementation of gradient descent algorithm
        ## input : type_matrix 'SH' for spherical harmonics or 'Wigner' for Wigner D-functions
        ##         m number of sampling points, bandlimited B, Coherence bound defined in the paper
        ##         update for phi and chi
        ## output : sampling points theta, phi with respect to the choice of type matrix and update
        #################################################################################################
        gamma = 0.5 # stepsize
        Coh = 1 #Initialization of coherence
        iterate = 0
        
        
        if self.matrix.types[0] == 'Wigner':
            
            ## Preallocation of the angles
            angles = np.zeros((self.matrix.m,3), dtype = np.float64)
            graddes_ang = np.copy(angles)



            ## Get angles
            theta = self.matrix.angles[:,0]
            phi = self.matrix.angles[:,1]
            chi = self.matrix.angles[:,2]

            ################################################################################################
            ## This condition if we choose matrix spherical harmonics
            ## 
            ################################################################################################
            if self.matrix.types[1] == 'partial':      
                ##### Gradient 
                gr_phi = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_phi
                gr_chi = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_chi


                ## Legendre bound
                LowerBound = BoundCoherence(self.matrix.m, 
                                            self.matrix.N, 
                                            self.matrix.B)[1]


                #########
                while iterate < self.max_iter and np.abs(Coh - LowerBound) > self.eps:

                    iterate += 1
                    ##### Update
                    phi = phi - gamma*gr_phi
                    chi = chi - gamma*gr_chi


                    #### Update Matrix
                    angles[:,0] = theta
                    angles[:,1] = phi
                    angles[:,2] = chi


                    matrix = Matrix(B = self.matrix.B,
                                    types = self.matrix.types,
                                    angles = angles)

                    ##### Update Gradient SH
                    gr_phi = Gradient(matrix = matrix, col_comb = self.col_comb).gr_phi  
                    gr_chi = Gradient(matrix = matrix, col_comb = self.col_comb).gr_chi

                    ### Calculate the coherence and compare with the Legendre bound
                    if Coherence(matrix.normA) < Coh:
                        Coh = Coherence(matrix.normA)
                        graddes_ang = angles




            #################################################################################################
            ## This condition if we choose matrix Wigner and update  phi and chi
            ##
            #################################################################################################
            else:

                ##### Gradient
                gr_theta = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_theta
                gr_phi = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_phi
                gr_chi = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_chi


                ### Welch bound for universal bound coherence
                WelchBound = BoundCoherence(self.matrix.m, 
                                            self.matrix.N, 
                                            self.matrix.B)[0]

                #########
                while iterate < self.max_iter and np.abs(Coh - WelchBound) > self.eps:

                    iterate += 1
                    ##### Update
                    theta = theta - gamma*gr_theta
                    phi = phi - gamma*gr_phi 
                    chi = chi - gamma*gr_chi


                    #### Update Matrix
                    angles[:,0] = theta
                    angles[:,1] = phi
                    angles[:,2] = chi

                    matrix = Matrix(B = self.matrix.B,
                                    types = self.matrix.types,
                                    angles = angles)

                    ##### Update Gradient 
                    gr_theta = Gradient(matrix = matrix, col_comb = self.col_comb).gr_theta
                    gr_phi = Gradient(matrix = matrix, col_comb = self.col_comb).gr_phi  
                    gr_chi = Gradient(matrix = matrix, col_comb = self.col_comb).gr_chi

                    ### Calculate the coherence and compare with the Legendre bound
                    if Coherence(matrix.normA) < Coh:
                        Coh = Coherence(matrix.normA)
                        graddes_ang = angles 
                        

            self.graddes_coh = Coh
            self.graddes_ang = graddes_ang
    
        elif self.matrix.types[0] == 'SH':
            
            ## Preallocation of the angles
            angles = np.zeros((self.matrix.m,2), dtype = np.float64)
            graddes_ang = np.copy(angles)



            ## Get angles
            theta = self.matrix.angles[:,0]
            phi = self.matrix.angles[:,1]
  

            ################################################################################################
            ## This condition if we choose matrix spherical harmonics
            ## 
            ################################################################################################
            if self.matrix.types[1] == 'partial':      
                ##### Gradient 
                gr_phi = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_phi
                


                ## Legendre bound
                LowerBound = BoundCoherence(self.matrix.m, 
                                            self.matrix.N, 
                                            self.matrix.B)[1]


                #########
                while iterate < self.max_iter and np.abs(Coh - LowerBound) > self.eps:

                    iterate += 1
                    ##### Update
                    phi = phi - gamma*gr_phi
                    


                    #### Update Matrix
                    angles[:,0] = theta
                    angles[:,1] = phi
                


                    matrix = Matrix(B = self.matrix.B,
                                    types = self.matrix.types,
                                    angles = angles)

                    ##### Update Gradient SH
                    gr_phi = Gradient(matrix = matrix, col_comb = self.col_comb).gr_phi  
                    

                    ### Calculate the coherence and compare with the Legendre bound
                    if Coherence(matrix.normA) < Coh:
                        Coh = Coherence(matrix.normA)
                        graddes_ang = angles




            #################################################################################################
            ## This condition if we choose matrix Wigner and update  phi and chi
            ##
            #################################################################################################
            else:

                ##### Gradient
                gr_theta = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_theta
                gr_phi = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_phi
                


                ### Welch bound for universal bound coherence
                WelchBound = BoundCoherence(self.matrix.m, 
                                            self.matrix.N, 
                                            self.matrix.B)[0]

                #########
                while iterate < self.max_iter and np.abs(Coh - WelchBound) > self.eps:

                    iterate += 1
                    ##### Update
                    theta = theta - gamma*gr_theta
                    phi = phi - gamma*gr_phi 
                   

                    #### Update Matrix
                    angles[:,0] = theta
                    angles[:,1] = phi
               
                    matrix = Matrix(B = self.matrix.B,
                                    types = self.matrix.types,
                                    angles = angles)

                    ##### Update Gradient 
                    gr_theta = Gradient(matrix = matrix, col_comb = self.col_comb).gr_theta
                    gr_phi = Gradient(matrix = matrix, col_comb = self.col_comb).gr_phi  
                  

                    ### Calculate the coherence and compare with the Legendre bound
                    if Coherence(matrix.normA) < Coh:
                        Coh = Coherence(matrix.normA)
                        graddes_ang = angles 

            self.graddes_coh = Coh
            self.graddes_ang = graddes_ang
            
        
        else:
            
            ## Preallocation of the angles
            angles = np.zeros((self.matrix.m,3), dtype = np.float64)
            graddes_ang = np.copy(angles)



            ## Get angles
            theta = self.matrix.angles[:,0]
            phi = self.matrix.angles[:,1]
            chi = self.matrix.angles[:,2]

            ################################################################################################
            ## This condition if we choose matrix spherical harmonics
            ## 
            ################################################################################################
            if self.matrix.types[1] == 'partial':      
                ##### Gradient 
                gr_phi = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_phi
                 


                ## Legendre bound
                LowerBound = BoundCoherence(self.matrix.m, 
                                            self.matrix.N, 
                                            self.matrix.B)[1]


                #########
                while iterate < self.max_iter and np.abs(Coh - LowerBound) > self.eps:

                    iterate += 1
                    ##### Update
                    phi = phi - gamma*gr_phi
                    


                    #### Update Matrix
                    angles[:,0] = theta
                    angles[:,1] = phi
                    angles[:,2] = chi


                    matrix = Matrix(B = self.matrix.B,
                                    types = self.matrix.types,
                                    angles = angles)

                    ##### Update Gradient SH
                    gr_phi = Gradient(matrix = matrix, col_comb = self.col_comb).gr_phi  
                   
                    ### Calculate the coherence and compare with the Legendre bound
                    if Coherence(matrix.normA) < Coh:
                        Coh = Coherence(matrix.normA)
                        graddes_ang = angles




            #################################################################################################
            ## This condition if we choose matrix Wigner and update  phi and chi
            ##
            #################################################################################################
            else:

                ##### Gradient
                gr_theta = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_theta
                gr_phi = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_phi
                


                ### Welch bound for universal bound coherence
                WelchBound = BoundCoherence(self.matrix.m, 
                                            self.matrix.N, 
                                            self.matrix.B)[0]

                #########
                while iterate < self.max_iter and np.abs(Coh - WelchBound) > self.eps:

                    iterate += 1
                    ##### Update
                    theta = theta - gamma*gr_theta
                    phi = phi - gamma*gr_phi 
                    


                    #### Update Matrix
                    angles[:,0] = theta
                    angles[:,1] = phi
                    angles[:,2] = chi

                    matrix = Matrix(B = self.matrix.B,
                                    types = self.matrix.types,
                                    angles = angles)

                    ##### Update Gradient 
                    gr_theta = Gradient(matrix = matrix, col_comb = self.col_comb).gr_theta
                    gr_phi = Gradient(matrix = matrix, col_comb = self.col_comb).gr_phi  
                     

                    ### Calculate the coherence and compare with the Legendre bound
                    if Coherence(matrix.normA) < Coh:
                        Coh = Coherence(matrix.normA)
                        graddes_ang = angles 
                 
            self.graddes_coh = Coh
            self.graddes_ang = graddes_ang
            
            
            
            
            
            



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


    def adam(self):
        gamma = 0.05 ## stepsize
        beta1 = 0.9 ## ADAM parameter
        beta2 = 0.999 ## ADAM parameter
      
        Coh = 1
        iterate = 0
        
        
        if self.matrix.types[0] == 'Wigner':
            ## Preallocation of the angles
            angles = np.zeros((self.matrix.m,3), dtype = np.float64)
            adam_ang = np.copy(angles)

            ## Get angles
            theta = self.matrix.angles[:,0]
            phi = self.matrix.angles[:,1]
            chi = self.matrix.angles[:,2]
            #################################################################################################
            ## This condition if we choose matrix spherical harmonics and update  phi
            ## 
            ################################################################################################

            if self.matrix.types[1] == 'partial':

                ##### Gradient 
                gr_phi = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_phi
                gr_chi = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_chi


                ## Legendre bound
                LowerBound = BoundCoherence(self.matrix.m, 
                                            self.matrix.N, 
                                            self.matrix.B)[1]
                mm_phi = np.zeros(self.matrix.m)
                v_phi = np.zeros(self.matrix.m)

                mm_chi = np.zeros(self.matrix.m)
                v_chi = np.zeros(self.matrix.m)

                while iterate < self.max_iter and np.abs(Coh - LowerBound) > self.eps:

                    ###########################################################
                    ### Update for phi
                    ##########################################################

                    iterate += 1

                    ## Update biased 1st moment estimate
                    mm_phi = beta1*mm_phi + (1.0 - beta1)*gr_phi
                    mm_chi = beta1*mm_chi + (1.0 - beta1)*gr_chi

                    ## Update biased 2nd raw moment estimate
                    v_phi = beta2*v_phi + (1.0 - beta2)*(gr_phi**2)
                    v_chi = beta2*v_chi + (1.0 - beta2)*(gr_chi**2)

                    ## Compute bias-corrected 1st moment estimate
                    mHat_phi = mm_phi/(1.0 - (beta1**iterate))
                    mHat_chi = mm_chi/(1.0 - (beta1**iterate))

                    ## Compute bias-corrected 2nd raw moment estimate
                    vHat_phi = v_phi/(1.0 - (beta2**iterate))
                    vHat_chi = v_chi/(1.0 - (beta2**iterate))

                    ## Update decision variables
                    phi = phi - gamma*mHat_phi/(np.sqrt(vHat_phi) + self.eps)
                    chi = chi - gamma*mHat_chi/(np.sqrt(vHat_chi) + self.eps)

                    #### Update Matrix
                    angles[:,0] = theta
                    angles[:,1] = phi
                    angles[:,2] = chi

                    matrix = Matrix(B = self.matrix.B,
                                    types = self.matrix.types,
                                    angles = angles)


                    ##### Update Gradient 

                    gr_phi = Gradient(matrix = matrix, col_comb = self.col_comb).gr_phi  
                    gr_chi = Gradient(matrix = matrix, col_comb = self.col_comb).gr_chi

                    ### Calculate the coherence and compare with the Legendre bound
                    if Coherence(matrix.normA) < Coh:
                        Coh = Coherence(matrix.normA)
                        adam_ang = angles 



            #################################################################################################
            ## This condition if we choose matrix Wigner D-functions  and update  phi,chi
            ## 
            ################################################################################################

            else:


                ##### Gradient
                gr_theta = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_theta
                gr_phi = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_phi
                gr_chi = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_chi


                ### Welch bound for universal bound coherence
                WelchBound = BoundCoherence(self.matrix.m, 
                                            self.matrix.N, 
                                            self.matrix.B)[0]



                mm_theta = np.zeros(self.matrix.m)
                v_theta = np.zeros(self.matrix.m)

                mm_phi = np.zeros(self.matrix.m)
                v_phi = np.zeros(self.matrix.m)

                mm_chi = np.zeros(self.matrix.m)
                v_chi = np.zeros(self.matrix.m)

                while iterate < self.max_iter and np.abs(Coh - WelchBound) > self.eps:

                    ###########################################################
                    ### Update for phi and chi
                    ##########################################################

                    iterate += 1

                    ## Update biased 1st moment estimate
                    mm_theta = beta1*mm_theta + (1.0 - beta1)*gr_theta
                    mm_phi = beta1*mm_phi + (1.0 - beta1)*gr_phi
                    mm_chi = beta1*mm_chi + (1.0 - beta1)*gr_chi

                    ## Update biased 2nd raw moment estimate
                    v_theta = beta2*v_theta + (1.0 - beta2)*(gr_theta**2)
                    v_phi = beta2*v_phi + (1.0 - beta2)*(gr_phi**2)
                    v_chi = beta2*v_chi + (1.0 - beta2)*(gr_chi**2)

                    ## Compute bias-corrected 1st moment estimate
                    mHat_theta = mm_theta/(1.0 - beta1**iterate)
                    mHat_phi = mm_phi/(1.0 - beta1**iterate)
                    mHat_chi = mm_chi/(1.0 - beta1*+iterate)

                    ## Compute bias-corrected 2nd raw moment estimate
                    vHat_theta = v_theta /(1.0 - beta2**iterate)
                    vHat_phi = v_phi/(1.0 - beta2**iterate)
                    vHat_chi = v_chi/(1.0 - beta2**iterate)

                    ## Update decision variables
                    theta = theta - gamma*mHat_theta/(np.sqrt(vHat_theta) + self.eps)
                    phi = phi - gamma*mHat_phi/(np.sqrt(vHat_phi) + self.eps)
                    chi = chi - gamma*mHat_chi/(np.sqrt(vHat_chi) + self.eps)


                    ### Update Matrix
                    angles[:,0] = theta
                    angles[:,1] = phi
                    angles[:,2] = chi

                    matrix = Matrix(B = self.matrix.B,
                                    types = self.matrix.types,
                                    angles = angles)

                    ##### Update Gradient 
                    gr_theta = Gradient(matrix = matrix, col_comb = self.col_comb).gr_theta
                    gr_phi = Gradient(matrix = matrix, col_comb = self.col_comb).gr_phi  
                    gr_chi = Gradient(matrix = matrix, col_comb = self.col_comb).gr_chi

                    ### Calculate the coherence and compare with the Legendre bound
                    if Coherence(matrix.normA) < Coh:
                        Coh = Coherence(matrix.normA)
                        adam_ang = angles 

            self.adam_coh = Coh
            self.adam_ang = adam_ang
            
        elif self.matrix.types[0] == 'SH':
            ## Preallocation of the angles
            angles = np.zeros((self.matrix.m,2), dtype = np.float64)
            adam_ang = np.copy(angles)
          
            ## Get angles
            theta = self.matrix.angles[:,0]
            phi = self.matrix.angles[:,1]
            
            #################################################################################################
            ## This condition if we choose matrix spherical harmonics and update  phi
            ## 
            ################################################################################################

            if self.matrix.types[1] == 'partial':

                ##### Gradient 
                gr_phi = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_phi
               


                ## Legendre bound
                LowerBound = BoundCoherence(self.matrix.m, 
                                            self.matrix.N, 
                                            self.matrix.B)[1]
                mm_phi = np.zeros(self.matrix.m)
                v_phi = np.zeros(self.matrix.m)

              

                while iterate < self.max_iter and np.abs(Coh - LowerBound) > self.eps:

                    ###########################################################
                    ### Update for phi
                    ##########################################################

                    iterate += 1

                    ## Update biased 1st moment estimate
                    mm_phi = beta1*mm_phi + (1.0 - beta1)*gr_phi
                    

                    ## Update biased 2nd raw moment estimate
                    v_phi = beta2*v_phi + (1.0 - beta2)*(gr_phi**2)
                

                    ## Compute bias-corrected 1st moment estimate
                    mHat_phi = mm_phi/(1.0 - (beta1**iterate))
                    

                    ## Compute bias-corrected 2nd raw moment estimate
                    vHat_phi = v_phi/(1.0 - (beta2**iterate))
             

                    ## Update decision variables
                    phi = phi - gamma*mHat_phi/(np.sqrt(vHat_phi) + self.eps)
                    

                    #### Update Matrix
                    angles[:,0] = theta
                    angles[:,1] = phi
                   

                    matrix = Matrix(B = self.matrix.B,
                                    types = self.matrix.types,
                                    angles = angles)


                    ##### Update Gradient 

                    gr_phi = Gradient(matrix = matrix, col_comb = self.col_comb).gr_phi  
                    

                    ### Calculate the coherence and compare with the Legendre bound
                    if Coherence(matrix.normA) < Coh:
                        Coh = Coherence(matrix.normA)
                        adam_ang = angles 



            #################################################################################################
            ## This condition if we choose matrix Wigner D-functions  and update  phi,chi
            ## 
            ################################################################################################

            else:


                ##### Gradient
                gr_theta = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_theta
                gr_phi = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_phi
            


                ### Welch bound for universal bound coherence
                WelchBound = BoundCoherence(self.matrix.m, 
                                            self.matrix.N, 
                                            self.matrix.B)[0]



                mm_theta = np.zeros(self.matrix.m)
                v_theta = np.zeros(self.matrix.m)

                mm_phi = np.zeros(self.matrix.m)
                v_phi = np.zeros(self.matrix.m)

                 
                while iterate < self.max_iter and np.abs(Coh - WelchBound) > self.eps:

                    ###########################################################
                    ### Update for phi and chi
                    ##########################################################

                    iterate += 1

                    ## Update biased 1st moment estimate
                    mm_theta = beta1*mm_theta + (1.0 - beta1)*gr_theta
                    mm_phi = beta1*mm_phi + (1.0 - beta1)*gr_phi
               

                    ## Update biased 2nd raw moment estimate
                    v_theta = beta2*v_theta + (1.0 - beta2)*(gr_theta**2)
                    v_phi = beta2*v_phi + (1.0 - beta2)*(gr_phi**2)
                    

                    ## Compute bias-corrected 1st moment estimate
                    mHat_theta = mm_theta/(1.0 - beta1**iterate)
                    mHat_phi = mm_phi/(1.0 - beta1**iterate)
                    

                    ## Compute bias-corrected 2nd raw moment estimate
                    vHat_theta = v_theta /(1.0 - beta2**iterate)
                    vHat_phi = v_phi/(1.0 - beta2**iterate)
              

                    ## Update decision variables
                    theta = theta - gamma*mHat_theta/(np.sqrt(vHat_theta) + self.eps)
                    phi = phi - gamma*mHat_phi/(np.sqrt(vHat_phi) + self.eps)
                     


                    ### Update Matrix
                    angles[:,0] = theta
                    angles[:,1] = phi
                   

                    matrix = Matrix(B = self.matrix.B,
                                    types = self.matrix.types,
                                    angles = angles)

                    ##### Update Gradient 
                    gr_theta = Gradient(matrix = matrix, col_comb = self.col_comb).gr_theta
                    gr_phi = Gradient(matrix = matrix, col_comb = self.col_comb).gr_phi  
                    

                    ### Calculate the coherence and compare with the Legendre bound
                    if Coherence(matrix.normA) < Coh:
                        Coh = Coherence(matrix.normA)
                        adam_ang = angles 

            self.adam_coh = Coh
            self.adam_ang = adam_ang
            
        else:
            
                ## Preallocation of the angles
            angles = np.zeros((self.matrix.m,3), dtype = np.float64)
            adam_ang = np.copy(angles)

            ## Get angles
            theta = self.matrix.angles[:,0]
            phi = self.matrix.angles[:,1]
            chi = self.matrix.angles[:,2]
            #################################################################################################
            ## This condition if we choose matrix spherical harmonics and update  phi
            ## 
            ################################################################################################

            if self.matrix.types[1] == 'partial':

                ##### Gradient 
                gr_phi = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_phi
                

                ## Legendre bound
                LowerBound = BoundCoherence(self.matrix.m, 
                                            self.matrix.N, 
                                            self.matrix.B)[1]
                mm_phi = np.zeros(self.matrix.m)
                v_phi = np.zeros(self.matrix.m)
 
                while iterate < self.max_iter and np.abs(Coh - LowerBound) > self.eps:

                    ###########################################################
                    ### Update for phi
                    ##########################################################

                    iterate += 1

                    ## Update biased 1st moment estimate
                    mm_phi = beta1*mm_phi + (1.0 - beta1)*gr_phi
                   
                    ## Update biased 2nd raw moment estimate
                    v_phi = beta2*v_phi + (1.0 - beta2)*(gr_phi**2)
                    

                    ## Compute bias-corrected 1st moment estimate
                    mHat_phi = mm_phi/(1.0 - (beta1**iterate))
                    

                    ## Compute bias-corrected 2nd raw moment estimate
                    vHat_phi = v_phi/(1.0 - (beta2**iterate))
                   

                    ## Update decision variables
                    phi = phi - gamma*mHat_phi/(np.sqrt(vHat_phi) + self.eps) 
                    
                    #### Update Matrix
                    angles[:,0] = theta
                    angles[:,1] = phi
                    angles[:,2] = chi

                    matrix = Matrix(B = self.matrix.B,
                                    types = self.matrix.types,
                                    angles = angles)


                    ##### Update Gradient 

                    gr_phi = Gradient(matrix = matrix, col_comb = self.col_comb).gr_phi  
                     
                    ### Calculate the coherence and compare with the Legendre bound
                    if Coherence(matrix.normA) < Coh:
                        Coh = Coherence(matrix.normA)
                        adam_ang = angles 



            #################################################################################################
            ## This condition if we choose matrix Wigner D-functions  and update  phi,chi
            ## 
            ################################################################################################

            else:


                ##### Gradient
                gr_theta = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_theta
                gr_phi = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_phi
                 


                ### Welch bound for universal bound coherence
                WelchBound = BoundCoherence(self.matrix.m, 
                                            self.matrix.N, 
                                            self.matrix.B)[0]



                mm_theta = np.zeros(self.matrix.m)
                v_theta = np.zeros(self.matrix.m)

                mm_phi = np.zeros(self.matrix.m)
                v_phi = np.zeros(self.matrix.m)

              

                while iterate < self.max_iter and np.abs(Coh - WelchBound) > self.eps:

                    ###########################################################
                    ### Update for phi and chi
                    ##########################################################

                    iterate += 1

                    ## Update biased 1st moment estimate
                    mm_theta = beta1*mm_theta + (1.0 - beta1)*gr_theta
                    mm_phi = beta1*mm_phi + (1.0 - beta1)*gr_phi
                  

                    ## Update biased 2nd raw moment estimate
                    v_theta = beta2*v_theta + (1.0 - beta2)*(gr_theta**2)
                    v_phi = beta2*v_phi + (1.0 - beta2)*(gr_phi**2)
                 

                    ## Compute bias-corrected 1st moment estimate
                    mHat_theta = mm_theta/(1.0 - beta1**iterate)
                    mHat_phi = mm_phi/(1.0 - beta1**iterate)
                     

                    ## Compute bias-corrected 2nd raw moment estimate
                    vHat_theta = v_theta /(1.0 - beta2**iterate)
                    vHat_phi = v_phi/(1.0 - beta2**iterate)
                

                    ## Update decision variables
                    theta = theta - gamma*mHat_theta/(np.sqrt(vHat_theta) + self.eps)
                    phi = phi - gamma*mHat_phi/(np.sqrt(vHat_phi) + self.eps)
                    


                    ### Update Matrix
                    angles[:,0] = theta
                    angles[:,1] = phi
                    angles[:,2] = chi

                    matrix = Matrix(B = self.matrix.B,
                                    types = self.matrix.types,
                                    angles = angles)

                    ##### Update Gradient 
                    gr_theta = Gradient(matrix = matrix, col_comb = self.col_comb).gr_theta
                    gr_phi = Gradient(matrix = matrix, col_comb = self.col_comb).gr_phi   
                    
                    ### Calculate the coherence and compare with the Legendre bound
                    if Coherence(matrix.normA) < Coh:
                        Coh = Coherence(matrix.normA)
                        adam_ang = angles 

            self.adam_coh = Coh
            self.adam_ang = adam_ang

            
    ########################################################
    ##  
    ## ADADELTA: An Adaptive Learning Rate Method
    #########################################################





    def adadelta(self):
       
        beta = 0.95 ## AdaDelta parameter
         
        Coh = 1
        iterate = 0
        
        if self.matrix.types[0] == 'Wigner':
            ## Preallocation of the angles
            angles = np.zeros((self.matrix.m,3), dtype = np.float64)
            adadelta_ang = np.copy(angles)


            ## Get angles
            theta = self.matrix.angles[:,0]
            phi = self.matrix.angles[:,1]
            chi = self.matrix.angles[:,2]
            #################################################################################################
            ## This condition if we choose matrix spherical harmonics and update  phi
            ## 
            ################################################################################################

            if self.matrix.types[1] == 'partial':

                ##### Gradient 
                gr_phi = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_phi
                gr_chi = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_chi


                ## Legendre bound
                LowerBound = BoundCoherence(self.matrix.m, 
                                            self.matrix.N, 
                                            self.matrix.B)[1]


                acculGradphi = np.zeros(self.matrix.m) # accumulated gradients
                acculDeltaphi = np.zeros(self.matrix.m) # accumulated updates

                acculGradchi = np.zeros(self.matrix.m) # accumulated gradients
                acculDeltachi = np.zeros(self.matrix.m) # accumulated updates


                while iterate < self.max_iter and np.abs(Coh - LowerBound) > self.eps:

                    ###########################################################
                    ### Update for phi
                    ##########################################################

                    iterate += 1

                    ## Update accumulated gradients
                    acculGradphi = beta*acculGradphi + (1.0 - beta)*gr_phi**2
                    acculGradchi = beta*acculGradchi + (1.0 - beta)*gr_chi**2

                    ## Calculate update
                    dCurrentphi = -(np.sqrt(acculDeltaphi + self.eps)/np.sqrt(acculGradphi + self.eps))*gr_phi
                    dCurrentchi = -(np.sqrt(acculDeltachi + self.eps)/np.sqrt(acculGradchi + self.eps))*gr_chi

                    ## Update accumulated updates
                    acculDeltaphi = beta*acculDeltaphi + (1.0 - beta)*dCurrentphi**2
                    acculDeltachi = beta*acculDeltachi + (1.0 - beta)*dCurrentchi**2

                    ## Update decision variables
                    phi = phi  + dCurrentphi
                    chi = chi + dCurrentchi

                    ### Update Matrix
                    angles[:,0] = theta
                    angles[:,1] = phi
                    angles[:,2] = chi

                    matrix = Matrix(B = self.matrix.B,
                                    types = self.matrix.types,
                                    angles = angles)

                    ##### Update Gradient 

                    gr_phi = Gradient(matrix = matrix, col_comb = self.col_comb).gr_phi  
                    gr_chi = Gradient(matrix = matrix, col_comb = self.col_comb).gr_chi

                    ### Calculate the coherence and compare with the Legendre bound
                    if Coherence(matrix.normA) < Coh:
                        Coh = Coherence(matrix.normA)
                        adadelta_ang = angles 



            #################################################################################################
            ## This condition if we choose 
            ## 
            ################################################################################################

            else:
               ##### Gradient
                gr_theta = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_theta
                gr_phi = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_phi
                gr_chi = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_chi


                ### Welch bound for universal bound coherence
                WelchBound = BoundCoherence(self.matrix.m, 
                                            self.matrix.N, 
                                            self.matrix.B)[0]




                acculGradtheta = np.zeros(self.matrix.m) # accumulated gradients
                acculDeltatheta = np.zeros(self.matrix.m) # accumulated updates


                acculGradphi = np.zeros(self.matrix.m) # accumulated gradients
                acculDeltaphi = np.zeros(self.matrix.m) # accumulated updates


                acculGradchi = np.zeros(self.matrix.m) # accumulated gradients
                acculDeltachi = np.zeros(self.matrix.m) # accumulated updates


                while iterate < self.max_iter and np.abs(Coh - WelchBound) > self.eps:

                    ###########################################################
                    ### Update for phi
                    ##########################################################

                    iterate += 1

                    ## Update accumulated gradients
                    acculGradtheta = beta*acculGradtheta + (1.0 - beta)*gr_theta**2
                    acculGradphi = beta*acculGradphi + (1.0 - beta)*gr_phi**2
                    acculGradchi = beta*acculGradchi + (1.0 - beta)*gr_chi**2

                    ## Calculate update
                    dCurrenttheta = -(np.sqrt(acculDeltatheta + self.eps)/np.sqrt(acculGradtheta + self.eps))*gr_theta 
                    dCurrentphi = -(np.sqrt(acculDeltaphi + self.eps)/np.sqrt(acculGradphi + self.eps))*gr_phi
                    dCurrentchi = -(np.sqrt(acculDeltachi + self.eps)/np.sqrt(acculGradchi + self.eps))*gr_chi

                    ## Update accumulated updates
                    acculDeltatheta = beta*acculDeltatheta + (1.0 - beta)*dCurrenttheta**2
                    acculDeltaphi = beta*acculDeltaphi + (1.0 - beta)*dCurrentphi**2
                    acculDeltachi = beta*acculDeltachi + (1.0 - beta)*dCurrentchi**2

                    ## Update decision variables
                    theta = theta + dCurrenttheta
                    phi = phi  + dCurrentphi
                    chi = chi + dCurrentchi

                    ### Update Matrix
                    angles[:,0] = theta
                    angles[:,1] = phi
                    angles[:,2] = chi

                    matrix = Matrix(B = self.matrix.B,
                                    types = self.matrix.types,
                                    angles = angles)

                    ##### Update Gradient 
                    gr_theta = Gradient(matrix = matrix, col_comb = self.col_comb).gr_theta
                    gr_phi = Gradient(matrix = matrix, col_comb = self.col_comb).gr_phi  
                    gr_chi = Gradient(matrix = matrix, col_comb = self.col_comb).gr_chi

                    ### Calculate the coherence and compare with the Legendre bound
                    if Coherence(matrix.normA) < Coh:
                        Coh = Coherence(matrix.normA)
                        adadelta_ang = angles 

            self.adadelta_coh = Coh
            self.adadelta_ang = adadelta_ang
            
            
        elif self.matrix.types[1] == 'SH':
            
                ## Preallocation of the angles
            angles = np.zeros((self.matrix.m,2), dtype = np.float64)
            adadelta_ang = np.copy(angles)


            ## Get angles
            theta = self.matrix.angles[:,0]
            phi = self.matrix.angles[:,1]
    
            #################################################################################################
            ## This condition if we choose matrix spherical harmonics and update  phi
            ## 
            ################################################################################################

            if self.matrix.types[1] == 'partial':

                ##### Gradient 
                gr_phi = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_phi
                

                ## Legendre bound
                LowerBound = BoundCoherence(self.matrix.m, 
                                            self.matrix.N, 
                                            self.matrix.B)[1]


                acculGradphi = np.zeros(self.matrix.m) # accumulated gradients
                acculDeltaphi = np.zeros(self.matrix.m) # accumulated updates

                 

                while iterate < self.max_iter and np.abs(Coh - LowerBound) > self.eps:

                    ###########################################################
                    ### Update for phi
                    ##########################################################

                    iterate += 1

                    ## Update accumulated gradients
                    acculGradphi = beta*acculGradphi + (1.0 - beta)*gr_phi**2
                    

                    ## Calculate update
                    dCurrentphi = -(np.sqrt(acculDeltaphi + self.eps)/np.sqrt(acculGradphi + self.eps))*gr_phi
                    

                    ## Update accumulated updates
                    acculDeltaphi = beta*acculDeltaphi + (1.0 - beta)*dCurrentphi**2
                    

                    ## Update decision variables
                    phi = phi  + dCurrentphi
                    
                    ### Update Matrix
                    angles[:,0] = theta
                    angles[:,1] = phi
                     

                    matrix = Matrix(B = self.matrix.B,
                                    types = self.matrix.types,
                                    angles = angles)

                    ##### Update Gradient 

                    gr_phi = Gradient(matrix = matrix, col_comb = self.col_comb).gr_phi  
                   

                    ### Calculate the coherence and compare with the Legendre bound
                    if Coherence(matrix.normA) < Coh:
                        Coh = Coherence(matrix.normA)
                        adadelta_ang = angles 



            #################################################################################################
            ## This condition if we choose 
            ## 
            ################################################################################################

            else:
               ##### Gradient
                gr_theta = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_theta
                gr_phi = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_phi
               


                ### Welch bound for universal bound coherence
                WelchBound = BoundCoherence(self.matrix.m, 
                                            self.matrix.N, 
                                            self.matrix.B)[0]




                acculGradtheta = np.zeros(self.matrix.m) # accumulated gradients
                acculDeltatheta = np.zeros(self.matrix.m) # accumulated updates


                acculGradphi = np.zeros(self.matrix.m) # accumulated gradients
                acculDeltaphi = np.zeros(self.matrix.m) # accumulated updates

 


                while iterate < self.max_iter and np.abs(Coh - WelchBound) > self.eps:

                    ###########################################################
                    ### Update for phi
                    ##########################################################

                    iterate += 1

                    ## Update accumulated gradients
                    acculGradtheta = beta*acculGradtheta + (1.0 - beta)*gr_theta**2
                    acculGradphi = beta*acculGradphi + (1.0 - beta)*gr_phi**2
                    

                    ## Calculate update
                    dCurrenttheta = -(np.sqrt(acculDeltatheta + self.eps)/np.sqrt(acculGradtheta + self.eps))*gr_theta 
                    dCurrentphi = -(np.sqrt(acculDeltaphi + self.eps)/np.sqrt(acculGradphi + self.eps))*gr_phi
                     

                    ## Update accumulated updates
                    acculDeltatheta = beta*acculDeltatheta + (1.0 - beta)*dCurrenttheta**2
                    acculDeltaphi = beta*acculDeltaphi + (1.0 - beta)*dCurrentphi**2
                     
                    ## Update decision variables
                    theta = theta + dCurrenttheta
                    phi = phi  + dCurrentphi
                     

                    ### Update Matrix
                    angles[:,0] = theta
                    angles[:,1] = phi
                   

                    matrix = Matrix(B = self.matrix.B,
                                    types = self.matrix.types,
                                    angles = angles)

                    ##### Update Gradient 
                    gr_theta = Gradient(matrix = matrix, col_comb = self.col_comb).gr_theta
                    gr_phi = Gradient(matrix = matrix, col_comb = self.col_comb).gr_phi  
                     

                    ### Calculate the coherence and compare with the Legendre bound
                    if Coherence(matrix.normA) < Coh:
                        Coh = Coherence(matrix.normA)
                        adadelta_ang = angles 

            self.adadelta_coh = Coh
            self.adadelta_ang = adadelta_ang

            
            
        else:
            
                ## Preallocation of the angles
            angles = np.zeros((self.matrix.m,3), dtype = np.float64)
            adadelta_ang = np.copy(angles)


            ## Get angles
            theta = self.matrix.angles[:,0]
            phi = self.matrix.angles[:,1]
            chi = self.matrix.angles[:,2]
            #################################################################################################
            ## This condition if we choose matrix spherical harmonics and update  phi
            ## 
            ################################################################################################

            if self.matrix.types[1] == 'partial':

                ##### Gradient 
                gr_phi = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_phi
                


                ## Legendre bound
                LowerBound = BoundCoherence(self.matrix.m, 
                                            self.matrix.N, 
                                            self.matrix.B)[1]


                acculGradphi = np.zeros(self.matrix.m) # accumulated gradients
                acculDeltaphi = np.zeros(self.matrix.m) # accumulated updates

              

                while iterate < self.max_iter and np.abs(Coh - LowerBound) > self.eps:

                    ###########################################################
                    ### Update for phi
                    ##########################################################

                    iterate += 1

                    ## Update accumulated gradients
                    acculGradphi = beta*acculGradphi + (1.0 - beta)*gr_phi**2
                    
                    ## Calculate update
                    dCurrentphi = -(np.sqrt(acculDeltaphi + self.eps)/np.sqrt(acculGradphi + self.eps))*gr_phi
                     

                    ## Update accumulated updates
                    acculDeltaphi = beta*acculDeltaphi + (1.0 - beta)*dCurrentphi**2
                     

                    ## Update decision variables
                    phi = phi  + dCurrentphi
                     

                    ### Update Matrix
                    angles[:,0] = theta
                    angles[:,1] = phi
                    angles[:,2] = chi

                    matrix = Matrix(B = self.matrix.B,
                                    types = self.matrix.types,
                                    angles = angles)

                    ##### Update Gradient 

                    gr_phi = Gradient(matrix = matrix, col_comb = self.col_comb).gr_phi  
                     
                    ### Calculate the coherence and compare with the Legendre bound
                    if Coherence(matrix.normA) < Coh:
                        Coh = Coherence(matrix.normA)
                        adadelta_ang = angles 



            #################################################################################################
            ## This condition if we choose 
            ## 
            ################################################################################################

            else:
               ##### Gradient
                gr_theta = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_theta
                gr_phi = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_phi
                


                ### Welch bound for universal bound coherence
                WelchBound = BoundCoherence(self.matrix.m, 
                                            self.matrix.N, 
                                            self.matrix.B)[0]




                acculGradtheta = np.zeros(self.matrix.m) # accumulated gradients
                acculDeltatheta = np.zeros(self.matrix.m) # accumulated updates


                acculGradphi = np.zeros(self.matrix.m) # accumulated gradients
                acculDeltaphi = np.zeros(self.matrix.m) # accumulated updates


                 


                while iterate < self.max_iter and np.abs(Coh - WelchBound) > self.eps:

                    ###########################################################
                    ### Update for phi
                    ##########################################################

                    iterate += 1

                    ## Update accumulated gradients
                    acculGradtheta = beta*acculGradtheta + (1.0 - beta)*gr_theta**2
                    acculGradphi = beta*acculGradphi + (1.0 - beta)*gr_phi**2
                     
                    ## Calculate update
                    dCurrenttheta = -(np.sqrt(acculDeltatheta + self.eps)/np.sqrt(acculGradtheta + self.eps))*gr_theta 
                    dCurrentphi = -(np.sqrt(acculDeltaphi + self.eps)/np.sqrt(acculGradphi + self.eps))*gr_phi
                   
                    ## Update accumulated updates
                    acculDeltatheta = beta*acculDeltatheta + (1.0 - beta)*dCurrenttheta**2
                    acculDeltaphi = beta*acculDeltaphi + (1.0 - beta)*dCurrentphi**2
                    

                    ## Update decision variables
                    theta = theta + dCurrenttheta
                    phi = phi  + dCurrentphi
                   

                    ### Update Matrix
                    angles[:,0] = theta
                    angles[:,1] = phi
                    angles[:,2] = chi

                    matrix = Matrix(B = self.matrix.B,
                                    types = self.matrix.types,
                                    angles = angles)

                    ##### Update Gradient 
                    gr_theta = Gradient(matrix = matrix, col_comb = self.col_comb).gr_theta
                    gr_phi = Gradient(matrix = matrix, col_comb = self.col_comb).gr_phi  
                    

                    ### Calculate the coherence and compare with the Legendre bound
                    if Coherence(matrix.normA) < Coh:
                        Coh = Coherence(matrix.normA)
                        adadelta_ang = angles 

            self.adadelta_coh = Coh
            self.adadelta_ang = adadelta_ang

            

    ####################################################################################
    ## AdaGrad
    ## Adaptive Subgradient Methods for Online Learning and Stochastic Optimization 
    ###################################################################################


    def adagrad(self):
        gamma = 0.05 ## stepsize
        
        Coh = 1
        iterate = 0
        
        if self.matrix.types[0] == 'Wigner':
            ## Preallocation of the angles
            angles = np.zeros((self.matrix.m,3), dtype = np.float64)
            adagrad_ang = np.copy(angles)

            ## Get angles
            theta = self.matrix.angles[:,0]
            phi = self.matrix.angles[:,1]
            chi = self.matrix.angles[:,2]
            #################################################################################################
            ## This condition if we choose matrix spherical harmonics and update  phi
            ## 
            ################################################################################################

            if self.matrix.types[1] == 'partial':

                ##### Gradient 
                gr_phi = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_phi
                gr_chi = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_chi


                ## Legendre bound
                LowerBound = BoundCoherence(self.matrix.m, 
                                            self.matrix.N, 
                                            self.matrix.B)[1]

                ## Get fix theta
                theta = self.matrix.angles[:,0]


                histogradphi = np.zeros(self.matrix.m) # historical gradients
                histogradchi = np.zeros(self.matrix.m)


                while iterate < self.max_iter and np.abs(Coh - LowerBound) > self.eps:

                    ###########################################################
                    ### Update for phi
                    ##########################################################

                    iterate += 1

                    ## Update historical gradients
                    histogradphi = histogradphi + gr_phi**2
                    histogradchi = histogradchi + gr_chi**2

                    ## Update decision variables
                    phi = phi - gamma*gr_phi/(np.sqrt(histogradphi) + self.eps)
                    chi = chi - gamma*gr_chi/(np.sqrt(histogradchi) + self.eps)

                    ### Update Matrix
                    angles[:,0] = theta
                    angles[:,1] = phi
                    angles[:,2] = chi

                    matrix = Matrix(B = self.matrix.B,
                                    types = self.matrix.types,
                                    angles = angles)

                    ##### Update Gradient 

                    gr_phi = Gradient(matrix = matrix, col_comb = self.col_comb).gr_phi  
                    gr_chi = Gradient(matrix = matrix, col_comb = self.col_comb).gr_chi

                    ### Calculate the coherence and compare with the Legendre bound
                    if Coherence(matrix.normA) < Coh:
                        Coh = Coherence(matrix.normA)
                        adagrad_ang = angles 


            ########################################################################################
            ## This condition if we choose matrix Wigner D-functions  and update  phi,chi
            ## 
            ################################################################################################

            else:


                ##### Gradient 
                gr_theta = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_theta
                gr_phi = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_phi
                gr_chi = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_chi


                ## Legendre bound
                WelchBound = BoundCoherence(self.matrix.m, 
                                            self.matrix.N, 
                                            self.matrix.B)[0]



                histogradtheta = np.zeros(self.matrix.m)
                histogradphi = np.zeros(self.matrix.m) # historical gradients
                histogradchi = np.zeros(self.matrix.m)


                while iterate < self.max_iter and abs(Coh - WelchBound) > self.eps:

                    ###########################################################
                    ### Update for phi
                    ##########################################################

                    iterate += 1

                    ## Update historical gradients
                    histogradtheta = histogradtheta + np.array(gr_theta)**2
                    histogradphi = histogradphi + np.array(gr_phi)**2
                    histogradchi = histogradchi + np.array(gr_chi)**2

                    ## Update decision variables
                    theta = theta - gamma*np.array(gr_theta)/(np.sqrt(histogradtheta) + self.eps)
                    phi = phi - gamma*np.array(gr_phi)/(np.sqrt(histogradphi) + self.eps)
                    chi = chi - gamma*np.array(gr_chi)/(np.sqrt(histogradchi) + self.eps)

                    ### Update Matrix
                    angles[:,0] = theta
                    angles[:,1] = phi
                    angles[:,2] = chi

                    matrix = Matrix(B = self.matrix.B,
                                    types = self.matrix.types,
                                    angles = angles)

                    ##### Update Gradient 
                    gr_theta = Gradient(matrix = matrix, col_comb = self.col_comb).gr_theta
                    gr_phi = Gradient(matrix = matrix, col_comb = self.col_comb).gr_phi  
                    gr_chi = Gradient(matrix = matrix, col_comb = self.col_comb).gr_chi

                    ### Calculate the coherence and compare with the Legendre bound
                    if Coherence(matrix.normA) < Coh:
                        Coh = Coherence(matrix.normA)
                        adagrad_ang = angles 

            self.adagrad_coh = Coh
            self.adagrad_ang = adagrad_ang
        
        elif self.matrix.types[0] == 'SH':
            ## Preallocation of the angles
            angles = np.zeros((self.matrix.m,2), dtype = np.float64)
            adagrad_ang = np.copy(angles)

            ## Get angles
            theta = self.matrix.angles[:,0]
            phi = self.matrix.angles[:,1]
            
            #################################################################################################
            ## This condition if we choose matrix spherical harmonics and update  phi
            ## 
            ################################################################################################

            if self.matrix.types[1] == 'partial':

                ##### Gradient 
                gr_phi = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_phi
                


                ## Legendre bound
                LowerBound = BoundCoherence(self.matrix.m, 
                                            self.matrix.N, 
                                            self.matrix.B)[1]

                ## Get fix theta
                theta = self.matrix.angles[:,0]


                histogradphi = np.zeros(self.matrix.m) # historical gradients
                

                while iterate < self.max_iter and np.abs(Coh - LowerBound) > self.eps:

                    ###########################################################
                    ### Update for phi
                    ##########################################################

                    iterate += 1

                    ## Update historical gradients
                    histogradphi = histogradphi + gr_phi**2
                     
                    ## Update decision variables
                    phi = phi - gamma*gr_phi/(np.sqrt(histogradphi) + self.eps)
                    
                    ### Update Matrix
                    angles[:,0] = theta
                    angles[:,1] = phi
                     

                    matrix = Matrix(B = self.matrix.B,
                                    types = self.matrix.types,
                                    angles = angles)

                    ##### Update Gradient 

                    gr_phi = Gradient(matrix = matrix, col_comb = self.col_comb).gr_phi  
                    

                    ### Calculate the coherence and compare with the Legendre bound
                    if Coherence(matrix.normA) < Coh:
                        Coh = Coherence(matrix.normA)
                        adagrad_ang = angles 


            ########################################################################################
            ## This condition if we choose matrix Wigner D-functions  and update  phi,chi
            ## 
            ################################################################################################

            else:


                ##### Gradient 
                gr_theta = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_theta
                gr_phi = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_phi
               

                ## Legendre bound
                WelchBound = BoundCoherence(self.matrix.m, 
                                            self.matrix.N, 
                                            self.matrix.B)[0]



                histogradtheta = np.zeros(self.matrix.m)
                histogradphi = np.zeros(self.matrix.m) # historical gradients
                


                while iterate < self.max_iter and abs(Coh - WelchBound) > self.eps:

                    ###########################################################
                    ### Update for phi
                    ##########################################################

                    iterate += 1

                    ## Update historical gradients
                    histogradtheta = histogradtheta + np.array(gr_theta)**2
                    histogradphi = histogradphi + np.array(gr_phi)**2
                    

                    ## Update decision variables
                    theta = theta - gamma*np.array(gr_theta)/(np.sqrt(histogradtheta) + self.eps)
                    phi = phi - gamma*np.array(gr_phi)/(np.sqrt(histogradphi) + self.eps)
                    

                    ### Update Matrix
                    angles[:,0] = theta
                    angles[:,1] = phi
                 

                    matrix = Matrix(B = self.matrix.B,
                                    types = self.matrix.types,
                                    angles = angles)

                    ##### Update Gradient 
                    gr_theta = Gradient(matrix = matrix, col_comb = self.col_comb).gr_theta
                    gr_phi = Gradient(matrix = matrix, col_comb = self.col_comb).gr_phi  
                   
                    ### Calculate the coherence and compare with the Legendre bound
                    if Coherence(matrix.normA) < Coh:
                        Coh = Coherence(matrix.normA)
                        adagrad_ang = angles 

            self.adagrad_coh = Coh
            self.adagrad_ang = adagrad_ang
            
        else:
            
            ## Preallocation of the angles
            angles = np.zeros((self.matrix.m,3), dtype = np.float64)
            adagrad_ang = np.copy(angles)

            ## Get angles
            theta = self.matrix.angles[:,0]
            phi = self.matrix.angles[:,1]
            chi = self.matrix.angles[:,2]
            #################################################################################################
            ## This condition if we choose matrix spherical harmonics and update  phi
            ## 
            ################################################################################################

            if self.matrix.types[1] == 'partial':

                ##### Gradient 
                gr_phi = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_phi
           


                ## Legendre bound
                LowerBound = BoundCoherence(self.matrix.m, 
                                            self.matrix.N, 
                                            self.matrix.B)[1]

                ## Get fix theta
                theta = self.matrix.angles[:,0]


                histogradphi = np.zeros(self.matrix.m) # historical gradients
                 


                while iterate < self.max_iter and np.abs(Coh - LowerBound) > self.eps:

                    ###########################################################
                    ### Update for phi
                    ##########################################################

                    iterate += 1

                    ## Update historical gradients
                    histogradphi = histogradphi + gr_phi**2
                   
                    ## Update decision variables
                    phi = phi - gamma*gr_phi/(np.sqrt(histogradphi) + self.eps)
                    

                    ### Update Matrix
                    angles[:,0] = theta
                    angles[:,1] = phi
                    angles[:,2] = chi

                    matrix = Matrix(B = self.matrix.B,
                                    types = self.matrix.types,
                                    angles = angles)

                    ##### Update Gradient 

                    gr_phi = Gradient(matrix = matrix, col_comb = self.col_comb).gr_phi  
                     

                    ### Calculate the coherence and compare with the Legendre bound
                    if Coherence(matrix.normA) < Coh:
                        Coh = Coherence(matrix.normA)
                        adagrad_ang = angles 


            ########################################################################################
            ## This condition if we choose matrix Wigner D-functions  and update  phi,chi
            ## 
            ################################################################################################

            else:


                ##### Gradient 
                gr_theta = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_theta
                gr_phi = Gradient(matrix = self.matrix, col_comb = self.col_comb).gr_phi
                


                ## Legendre bound
                WelchBound = BoundCoherence(self.matrix.m, 
                                            self.matrix.N, 
                                            self.matrix.B)[0]



                histogradtheta = np.zeros(self.matrix.m)
                histogradphi = np.zeros(self.matrix.m) # historical gradients
               


                while iterate < self.max_iter and abs(Coh - WelchBound) > self.eps:

                    ###########################################################
                    ### Update for phi
                    ##########################################################

                    iterate += 1

                    ## Update historical gradients
                    histogradtheta = histogradtheta + np.array(gr_theta)**2
                    histogradphi = histogradphi + np.array(gr_phi)**2
                   

                    ## Update decision variables
                    theta = theta - gamma*np.array(gr_theta)/(np.sqrt(histogradtheta) + self.eps)
                    phi = phi - gamma*np.array(gr_phi)/(np.sqrt(histogradphi) + self.eps)
                     
                    ### Update Matrix
                    angles[:,0] = theta
                    angles[:,1] = phi
                    angles[:,2] = chi

                    matrix = Matrix(B = self.matrix.B,
                                    types = self.matrix.types,
                                    angles = angles)

                    ##### Update Gradient 
                    gr_theta = Gradient(matrix = matrix, col_comb = self.col_comb).gr_theta
                    gr_phi = Gradient(matrix = matrix, col_comb = self.col_comb).gr_phi  
                    
                    ### Calculate the coherence and compare with the Legendre bound
                    if Coherence(matrix.normA) < Coh:
                        Coh = Coherence(matrix.normA)
                        adagrad_ang = angles 

            self.adagrad_coh = Coh
            self.adagrad_ang = adagrad_ang



