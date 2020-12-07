##############################################################################################################
## This file contains functions to calculate derivative of the product spherical harmonics and derivative 
## of the product Wigner D-functions.
##
##############################################################################################################
from itertools import combinations
import numpy as np
import numpy.matlib as npmat


class Gradient:

    def __init__(self, matrix):
        self.matrix = matrix
        self.get_gradient()
    
    def get_gradient(self):
        
    
        #######################
        if self.matrix.types[0] == 'SH':     
            self.grad_sh()

        elif self.matrix.types[0] == 'Wigner':
            self.grad_wigner()

        else:
            self.grad_wigner_snf()



    def grad_sh(self):
        #########################################################################################################
        ## Function to calculate derivative of product spherical harmonics
        ## input : m dimensions of theta and phi, bandlimited B, update "all" mean we derive the product spherical 
        ## harmonics with respect to theta and phi (elevation and azimuth). Update "partial" only calculate
        ## derivative of the product spherical harmonics w.r.t phi and fix the theta with equispaced samples.
        ## 
        ## Partial is used to verify the lower bound coherence of spherical harmonics matrix if we fix sampling on
        ## theta
        ##
        ##
        ## Since we want to optimize the coherence of the matrix (inner product of difference columns), we have
        ## to calculate the gradient w.r.t theta, or phi
        ## 
        ## input : theta, phi m dimension samples on sphere, bandlimited B, and update
        ## output : gr_theta,gr_phi -- gradient with respect to (theta,phi) for update all,
        ##          gr_phi -- gradient with respect to (phi) for partial
        ##########################################################################################################
        
        ## Allocation
        theta = self.matrix.angles[:,0]
        phi = self.matrix.angles[:,1]
        m = self.matrix.m
        N = self.matrix.N
        lk = self.matrix.deg_order
        Plk = self.matrix.Plk
        dPlk = self.matrix.dPlk
        
        ##################################
        col_comb = np.array(list(combinations(range(N),2)))
        comb_lk = [lk[col_comb[:,0],:],lk[col_comb[:,1],:]]
        
        ## Product of combination of degree and order associated Legendre 
        ProductasLeg = Plk[:,col_comb[:,0]]*Plk[:,col_comb[:,1]] 
        
        ## diferences order
        k = comb_lk[0][:,1] - comb_lk[1][:,1]
    

        ##########################################################################################################
        ## Derivative of q-norm
        ##########################################################################################################
        q = 8.0
        
        ##########################################################################################################
        ## Matrix-based derivative w.r.t phi
        ##########################################################################################################
        mat_cos = np.cos(np.outer(phi,k))
        mat_sin = np.sin(np.outer(phi,k))
    
        Qnorm = (q/2.0)*np.sqrt(np.abs(np.sum(ProductasLeg*mat_cos,0))**2 +
                                np.abs(np.sum(ProductasLeg*mat_sin,0))**2)**(q-2)
    
        Qnorm1 = (1/q)*np.sum(np.sqrt(np.abs(np.sum(ProductasLeg*mat_cos,0))**2 + 
                                      np.abs(np.sum(ProductasLeg*mat_sin,0))**2)**q)**((1/q)-1)


        if self.matrix.types[1] == 'all':
        ########################################################################################################
        ## Direct calculation of derivativei with respect to theta
        #######################################################################################################

            dPlktotal = (Plk[:,col_comb[:,0]]*dPlk[:,col_comb[:,1]] +
                         dPlk[:,col_comb[:,0]]*Plk[:,col_comb[:,1]])

    
        ##################################################################################################
        ## Matrix-based derivative w.r.t theta
        ##################################################################################################
            gr_temp_theta1 = 2.0*npmat.repmat(np.sum(ProductasLeg*mat_cos,0),m,1)*(dPlktotal*mat_cos)
            gr_temp_theta2 = 2.0*npmat.repmat(np.sum(ProductasLeg*mat_sin,0),m,1)*(dPlktotal*mat_sin)
            gr_temp_theta = gr_temp_theta1 + gr_temp_theta2
            gr_theta = Qnorm1*np.sum(npmat.repmat(Qnorm,m,1)*gr_temp_theta, 1)

        ####################################################################################################
        ## Matrix-based derivative w.r.t phi
        ###################################################################################################
            grad_temp1 = 2.0*npmat.repmat(np.sum(ProductasLeg*mat_cos,0),m,1)
            grad_temp2 = ProductasLeg*mat_sin*npmat.repmat(-k,m,1)
            grad1 = grad_temp1*grad_temp2

            grad_temp3 = 2.0*npmat.repmat(np.sum(ProductasLeg*mat_sin,0),m,1)
            grad_temp4 = ProductasLeg*mat_cos*npmat.repmat(k,m,1)
            grad2 = grad_temp3*grad_temp4
            grad_temp = grad1 + grad2
            
            ## Gradient phi
            gr_phi = Qnorm1*np.sum(npmat.repmat(Qnorm,m,1)*grad_temp,1)

            self.gr_theta = gr_theta
            self.gr_phi = gr_phi

        else:

            grad_temp1 = 2.0*npmat.repmat(np.sum(ProductasLeg*mat_cos,0),m,1)
            grad_temp2 = ProductasLeg*mat_sin*npmat.repmat(-k,m,1)
            grad1 = grad_temp1*grad_temp2

            grad_temp3 = 2.0*npmat.repmat(np.sum(ProductasLeg*mat_sin,0),m,1)
            grad_temp4 = ProductasLeg*mat_cos*npmat.repmat(k,m,1)
            grad2 = grad_temp3*grad_temp4
            grad_temp = grad1 + grad2
            
            ## Gradient phi
            gr_phi = Qnorm1*np.sum(npmat.repmat(Qnorm,m,1)*grad_temp,1)


            self.gr_phi = gr_phi
    

    def grad_wigner(self):
    #########################################################################################################
    ## Function to calculate derivative of product Wigner D-functions
    ## input : m dimensions of theta,phi, and chi, bandlimited B, update "all" mean we derive the product
    ## Wigner D-functions  with respect to theta, phi, chi (elevation, azimuth and polarization).
    ## Update "partial" only calculate derivative of the product Wigner D-functions  w.r.t phi and chi
    ## 
    ## Partial is used to verify the lower bound coherence of Wigner D-functions if we fix sampling on theta
    ##
    ## Since we want to optimize the coherence of the matrix (inner product of difference columns), we have
    ## to calculate the gradient w.r.t theta, phi and chi
    ##
    ## input : theta, phi, chi in m-dimension samples on the rotation group, bandlimited B, and update
    ## output : gr_theta,gr_phi, gr_chi -- gradient with respect to (theta,phi,chi) for update all,
    ##          gr_phi,gr_chi -- gradient with respect to (phi,chi) for partial
    ##########################################################################################################
        
        
        ### Allocation
        theta = self.matrix.angles[:,0]
        phi = self.matrix.angles[:,1]
        chi = self.matrix.angles[:,2]
        m = self.matrix.m
        N = self.matrix.N
        lkn = self.matrix.deg_order
        wigner_d = self.matrix.Wignerd
        d_wigner_d = self.matrix.dWignerd
        
        
        
        ### Combination for coherence
        col_comb = np.array(list(combinations(range(N),2)))
        comb_lkn = [lkn[col_comb[:,0],:],lkn[col_comb[:,1],:]]

        ## Product of combination degree and orders Wigner d-functions
        ProductWignerd = wigner_d[:,col_comb[:,0]]*wigner_d[:,col_comb[:,1]]

        ########################################################################
        ## Matrix-based phi and chi
        #######################################################################
        k = comb_lkn[0][:,1] - comb_lkn[1][:,1]
        n = comb_lkn[0][:,2] - comb_lkn[1][:,2]

        ## Derivative of q norm
        q = 8.0

        mat_cos = np.cos(np.outer(phi,k) + np.outer(chi,n))
        mat_sin = np.sin(np.outer(phi,k) + np.outer(chi,n))

        Qnorm = (q/2.0)*np.sqrt(np.abs(np.sum(ProductWignerd*mat_cos,0))**2 +
                                np.abs(np.sum(ProductWignerd*mat_sin,0))**2)**(q-2)
        Qnorm1 = (1/q)*np.sum(np.sqrt(np.abs(np.sum(ProductWignerd*mat_cos,0))**2 +
                                      np.abs(np.sum(ProductWignerd*mat_sin,0))**2)**q)**((1/q)-1)
 

        if self.matrix.types[1] == 'all':

        ##################################################################################
        ## Direct derivative with respect to theta
        ##################################################################################
            dWignerd = (wigner_d[:,col_comb[:,0]]*d_wigner_d[:,col_comb[:,1]] + 
                        d_wigner_d[:,col_comb[:,0]]*wigner_d[:,col_comb[:,1]])
        
        ##################################################################################
        ## Matrix-based derivative w.r.t to the q-norm
        ###################################################################################
        
            gr_temp_theta1 = 2.0*npmat.repmat(np.sum(ProductWignerd*mat_cos,0),m,1)*(dWignerd*mat_cos)
            gr_temp_theta2 = 2.0*npmat.repmat(np.sum(ProductWignerd*mat_sin,0),m,1)*(dWignerd*mat_sin)
            gr_temp_theta = gr_temp_theta1 + gr_temp_theta2
            gr_theta = Qnorm1*np.sum(npmat.repmat(Qnorm,m,1)*gr_temp_theta,axis = 1)
        
        
        
        ######################################################################
        ## Matrix-based phi
        #####################################################################
            grad_temp1_phi = 2.0*npmat.repmat(np.sum(ProductWignerd*mat_cos,0),m,1)
            grad_temp2_phi = (ProductWignerd*mat_sin)*npmat.repmat(-k,m,1)
            grad1_phi = (grad_temp1_phi*grad_temp2_phi)

            grad_temp3_phi = 2.0*npmat.repmat(np.sum(ProductWignerd*mat_sin,0),m,1)
            grad_temp4_phi = (ProductWignerd*mat_cos)*npmat.repmat(k,m,1)
            grad2_phi = (grad_temp3_phi*grad_temp4_phi)
            grad_temp_phi = grad1_phi + grad2_phi

        ######################################################################
        ## Matrix-based chi
        ######################################################################

            grad_temp1_chi = 2.0*npmat.repmat(np.sum(ProductWignerd*mat_cos,0),m,1)
            grad_temp2_chi = (ProductWignerd*mat_sin)*npmat.repmat(-n,m,1)
            grad1_chi = (grad_temp1_chi*grad_temp2_chi)

            grad_temp3_chi = 2.0*npmat.repmat(np.sum(ProductWignerd*mat_sin,0),m,1)
            grad_temp4_chi = (ProductWignerd*mat_cos)*npmat.repmat(n,m,1)
            grad2_chi = (grad_temp3_chi*grad_temp4_chi)
            grad_temp_chi = grad1_chi + grad2_chi

            gr_phi = Qnorm1*np.sum((npmat.repmat(Qnorm,m,1)*grad_temp_phi), 1)
            gr_chi = Qnorm1*np.sum((npmat.repmat(Qnorm,m,1)*grad_temp_chi), 1)
        
            self.gr_theta = gr_theta
            self.gr_phi = gr_phi
            self.gr_chi = gr_chi


        else:
        
        ######################################################################
        ######################################################################
        ## Matrix-based phi
        #####################################################################
            grad_temp1_phi = 2.0*npmat.repmat(np.sum(ProductWignerd*mat_cos,0),m,1)
            grad_temp2_phi = (ProductWignerd*mat_sin)*npmat.repmat(-k,m,1)
            grad1_phi = grad_temp1_phi*grad_temp2_phi

            grad_temp3_phi = 2.0*npmat.repmat(np.sum(ProductWignerd*mat_sin,0),m,1)
            grad_temp4_phi = (ProductWignerd*mat_cos)*npmat.repmat(k,m,1)
            grad2_phi = grad_temp3_phi*grad_temp4_phi
            grad_temp_phi = grad1_phi + grad2_phi

        ######################################################################
        ## Matrix-based chi
        ######################################################################

            grad_temp1_chi = 2.0*npmat.repmat(np.sum(ProductWignerd*mat_cos,0),m,1)
            grad_temp2_chi = (ProductWignerd*mat_sin)*npmat.repmat(-n,m,1)
            grad1_chi = grad_temp1_chi*grad_temp2_chi

            grad_temp3_chi = 2.0*npmat.repmat(np.sum(ProductWignerd*mat_sin,0),m,1)
            grad_temp4_chi = (ProductWignerd*mat_cos)*npmat.repmat(n,m,1)
            grad2_chi = grad_temp3_chi*grad_temp4_chi
            grad_temp_chi = grad1_chi + grad2_chi

            gr_phi = Qnorm1*np.sum(npmat.repmat(Qnorm,m,1)*grad_temp_phi, 1)
            gr_chi = Qnorm1*np.sum(npmat.repmat(Qnorm,m,1)*grad_temp_chi, 1)
    
            self.gr_phi = gr_phi
            self.gr_chi = gr_chi
    

    def grad_wigner_snf(self):
        
        
        ### Allocation
        theta = self.matrix.angles[:,0]
        phi = self.matrix.angles[:,1]
        chi = self.matrix.angles[:,2]
        m = self.matrix.m
        N = self.matrix.N
        lk = self.matrix.deg_order
        dmm_plus = self.matrix.dmm_plus
        dmm_min = self.matrix.dmm_min
        d_dmm_plus = self.matrix.d_dmm_plus
        d_dmm_min  = self.matrix.d_dmm_min
        norm_A = self.matrix.normA
        
        
        #### Combination    
        col_comb = np.array(list(combinations(range(2*N),2)))
        ProductCoh = norm_A[:,col_comb[:,0]]*np.conj(norm_A[:,col_comb[:,1]])
        normalization = self.matrix.normalize[col_comb[:,0]]*self.matrix.normalize[col_comb[:,1]]
       
        
        ## Derivative of q norm
        q = 8.0

        Qnorm = (q/2.0)*np.abs(np.sum(ProductCoh,0))**(q-2)


        Qnorm1 = (1/q)*np.sum(np.abs(np.sum(ProductCoh,0))**q)**((1/q)-1)


    

        ## Case 3 (All possible_combination of two bases)
        col_comb3 = np.array(np.meshgrid(range(N), range(N))).T.reshape(-1,2)
        comb_lk3 = [lk[col_comb3[:,0],:],lk[col_comb3[:,1],:]]
        ## Combination product Wigner small d
        d1 = dmm_plus[:,col_comb3[:,0]]*dmm_plus[:,col_comb3[:,1]]
        d2 = dmm_plus[:,col_comb3[:,0]]*dmm_min[:,col_comb3[:,1]]
        d3 = dmm_min[:,col_comb3[:,0]]*dmm_plus[:,col_comb3[:,1]]
        d4 = dmm_min[:,col_comb3[:,0]]*dmm_min[:,col_comb3[:,1]]

        ## Derivatives d1,d2,d3,d4 (w.r.t theta)
        d_d1 = (d_dmm_plus[:,col_comb3[:,0]]*dmm_plus[:,col_comb3[:,1]] + 
                dmm_plus[:,col_comb3[:,0]]*d_dmm_plus[:, col_comb3[:,1]]) 
        d_d2 = (d_dmm_plus[:,col_comb3[:,0]]*dmm_min[:,col_comb3[:,1]] +
                dmm_plus[:,col_comb3[:,0]]*d_dmm_min[:,col_comb3[:,1]])
        d_d3 = (d_dmm_min[:,col_comb3[:,0]]*dmm_plus[:,col_comb3[:,1]] +
                dmm_min[:,col_comb3[:,0]]*d_dmm_plus[:,col_comb3[:,1]])
        d_d4 = (d_dmm_min[:,col_comb3[:,0]]*dmm_min[:,col_comb3[:,1]] +
                dmm_min[:,col_comb3[:,0]]*d_dmm_min[:,col_comb3[:,1]])

        ## diferences order
        k = comb_lk3[0][:,1] - comb_lk3[1][:,1]

        ## Combination sine and cosine 
        c1 = np.cos(np.outer(phi,k))
        c2 = np.cos(np.outer(phi,k) + 2*chi[:,np.newaxis])
        c3 = np.cos(np.outer(phi,k) - 2*chi[:,np.newaxis])
        s1 = np.sin(np.outer(phi,k))
        s2 = np.sin(np.outer(phi,k) + 2*chi[:,np.newaxis])
        s3 = np.sin(np.outer(phi,k) - 2*chi[:,np.newaxis])


        ## Derivatives c1,c2,c3 and s1,s2,s3 (w.r.t phi)
        d_c1 = npmat.repmat(-k,m,1)*s1
        d_c2 = npmat.repmat(-k,m,1)*s2
        d_c3 = npmat.repmat(-k,m,1)*s3

        d_s1 = npmat.repmat(k,m,1)*c1
        d_s2 = npmat.repmat(k,m,1)*c2
        d_s3 = npmat.repmat(k,m,1)*c3
        
        ### Condition for all or partial
        
        if self.matrix.types[1] == 'all':

            ## Real
            case3_real = c1*d1 - c1*d4 - c2*d2 + c3*d3
                
            ## Deriv Real w.r.t theta
            d_case3_real_theta = c1*d_d1 - c1*d_d4 - c2*d_d2 + c3*d_d3

            ## Deriv Real w.r.t phi
            d_case3_real_phi = d_c1*d1 - d_c1*d4 - d_c2*d2 + d_c3*d3

            ## Imag
            case3_imag = s1*d1 - s1*d4 - s2*d2 + s3*d3


            ## Deriv Imag  w.r.t theta
            d_case3_imag_theta =  s1*d_d1 - s1*d_d4 - s2*d_d2 + s3*d_d3

            ## Deriv Imag w.r.t phi
            d_case3_imag_phi =  d_s1*d1 - d_s1*d4 - d_s2*d2 + d_s3*d3


            ##Total derivation 
            gr_theta_case3 = case3_real*d_case3_real_theta + case3_imag*d_case3_imag_theta
            gr_phi_case3 = case3_real*d_case3_real_phi + case3_imag*d_case3_imag_phi
            #############################################################################
            ## Case2 (coherence inside second basis, negative and negative)
            ############################################################################

            idx_12 = np.nonzero(col_comb3[:,1] > col_comb3[:,0])[0]

            ## Real
            case2_real = (c1[:,idx_12]*d1[:,idx_12] + c1[:,idx_12]*d4[:,idx_12] -
                          c2[:,idx_12]*d2[:,idx_12] - c3[:,idx_12]*d3[:,idx_12])

            ## Deriv Real w.r.t theta
            d_case2_real_theta = (c1[:,idx_12]*d_d1[:,idx_12] + 
                                  c1[:,idx_12]*d_d4[:,idx_12] - 
                                  c2[:,idx_12]*d_d2[:,idx_12] - 
                                  c3[:,idx_12]*d_d3[:,idx_12])

            ## Deriv Real w.r.t phi
            d_case2_real_phi = (d_c1[:,idx_12]*d1[:,idx_12] + 
                                d_c1[:,idx_12]*d4[:,idx_12] - 
                                d_c2[:,idx_12]*d2[:,idx_12] - 
                                d_c3[:,idx_12]*d3[:,idx_12])

            ## Imag
            case2_imag = (s1[:,idx_12]*d1[:,idx_12] + 
                          s1[:,idx_12]*d4[:,idx_12] - 
                          s2[:,idx_12]*d2[:,idx_12] - 
                          s3[:,idx_12]*d3[:,idx_12])


            ## Deriv Imag  w.r.t theta
            d_case2_imag_theta =  (s1[:,idx_12]*d_d1[:,idx_12] +
                                   s1[:,idx_12]*d_d4[:,idx_12] - 
                                   s2[:,idx_12]*d_d2[:,idx_12] - 
                                   s3[:,idx_12]*d_d3[:,idx_12])

            ## Deriv Imag w.r.t phi
            d_case2_imag_phi =  (d_s1[:,idx_12]*d1[:,idx_12] +
                                 d_s1[:,idx_12]*d4[:,idx_12] - 
                                 d_s2[:,idx_12]*d2[:,idx_12] - 
                                 d_s3[:,idx_12]*d3[:,idx_12])


            ##Total derivation(DONT FORGET NORMALIZATION)
            gr_theta_case2 = case2_real*d_case2_real_theta + case2_imag*d_case2_imag_theta
            gr_phi_case2   = case2_real*d_case2_real_phi + case2_imag*d_case2_imag_phi

            ###############################################################################
            ## Case 1 (Coherence between basis 1, positive and positive)
            ###############################################################################

            ## Real
            case1_real = (c1[:,idx_12]*d1[:,idx_12] + c1[:,idx_12]*d4[:,idx_12] +
                          c2[:,idx_12]*d2[:,idx_12] + c3[:,idx_12]*d3[:,idx_12])

            ## Deriv Real w.r.t theta
            d_case1_real_theta = (c1[:,idx_12]*d_d1[:,idx_12] +
                                  c1[:,idx_12]*d_d4[:,idx_12] +
                                  c2[:,idx_12]*d_d2[:,idx_12] +
                                  c3[:,idx_12]*d_d3[:,idx_12])

            ## Deriv Real w.r.t phi
            d_case1_real_phi = (d_c1[:,idx_12]*d1[:,idx_12] +
                                d_c1[:,idx_12]*d4[:,idx_12] +
                                d_c2[:,idx_12]*d2[:,idx_12] +
                                d_c3[:,idx_12]*d3[:,idx_12])

            ## Imag
            case1_imag = (s1[:,idx_12]*d1[:,idx_12] +
                          s1[:,idx_12]*d4[:,idx_12] +
                          s2[:,idx_12]*d2[:,idx_12] +
                          s3[:,idx_12]*d3[:,idx_12])


            ## Deriv Imag  w.r.t theta
            d_case1_imag_theta =  (s1[:,idx_12]*d_d1[:,idx_12] +
                                   s1[:,idx_12]*d_d4[:,idx_12] +
                                   s2[:,idx_12]*d_d2[:,idx_12] +
                                   s3[:,idx_12]*d_d3[:,idx_12])

            ## Deriv Imag w.r.t phi
            d_case1_imag_phi =  (d_s1[:,idx_12]*d1[:,idx_12] +
                                 d_s1[:,idx_12]*d4[:,idx_12] +
                                 d_s2[:,idx_12]*d2[:,idx_12] +
                                 d_s3[:,idx_12]*d3[:,idx_12])


            ##Total derivation(DONT FORGET NORMALIZATION)
            gr_theta_case1 = case1_real*d_case1_real_theta + case1_imag*d_case1_imag_theta
            gr_phi_case1   = case1_real*d_case1_real_phi + case1_imag*d_case1_imag_phi



            #################################################################################
            ## Total all cases (DONT FORGET NORMALIZATION)
            #################################################################################

            gr_theta_total = (np.concatenate((gr_theta_case1,gr_theta_case3, gr_theta_case2),axis = 1)/
                              normalization[np.newaxis,:])
             
            
            gr_phi_total = (np.concatenate((gr_phi_case1, gr_phi_case3, gr_phi_case2), axis = 1)/
                            normalization[np.newaxis,:])
            
           
            ################################################################################
            ########### Gradient
            self.gr_theta = Qnorm1*np.sum(npmat.repmat(Qnorm,m,1)*gr_theta_total, 1)
            self.gr_phi   = Qnorm1*np.sum(npmat.repmat(Qnorm,m,1)*gr_phi_total, 1)
            
        else:
            ##################### Partial
            ## Real
            case3_real = c1*d1 - c1*d4 - c2*d2 + c3*d3
                
            ## Deriv Real w.r.t phi
            d_case3_real_phi = d_c1*d1 - d_c1*d4 - d_c2*d2 + d_c3*d3

            ## Imag
            case3_imag = s1*d1 - s1*d4 - s2*d2 + s3*d3

            ## Deriv Imag w.r.t phi
            d_case3_imag_phi =  d_s1*d1 - d_s1*d4 - d_s2*d2 + d_s3*d3


            ##Total derivation 
            gr_phi_case3 = case3_real*d_case3_real_phi + case3_imag*d_case3_imag_phi
            #############################################################################
            ## Case2 (coherence inside second basis, negative and negative)
            ############################################################################

            idx_12 = np.nonzero(col_comb3[:,1] > col_comb3[:,0])[0]

            ## Real
            case2_real = (c1[:,idx_12]*d1[:,idx_12] + c1[:,idx_12]*d4[:,idx_12] -
                          c2[:,idx_12]*d2[:,idx_12] - c3[:,idx_12]*d3[:,idx_12])


            ## Deriv Real w.r.t phi
            d_case2_real_phi = (d_c1[:,idx_12]*d1[:,idx_12] + 
                                d_c1[:,idx_12]*d4[:,idx_12] - 
                                d_c2[:,idx_12]*d2[:,idx_12] - 
                                d_c3[:,idx_12]*d3[:,idx_12])

            ## Imag
            case2_imag = (s1[:,idx_12]*d1[:,idx_12] + 
                          s1[:,idx_12]*d4[:,idx_12] - 
                          s2[:,idx_12]*d2[:,idx_12] - 
                          s3[:,idx_12]*d3[:,idx_12])



            ## Deriv Imag w.r.t phi
            d_case2_imag_phi =  (d_s1[:,idx_12]*d1[:,idx_12] +
                                 d_s1[:,idx_12]*d4[:,idx_12] - 
                                 d_s2[:,idx_12]*d2[:,idx_12] - 
                                 d_s3[:,idx_12]*d3[:,idx_12])


            ##Total derivation(DONT FORGET NORMALIZATION)
            gr_phi_case2   = case2_real*d_case2_real_phi + case2_imag*d_case2_imag_phi

            ###############################################################################
            ## Case 1 (Coherence between basis 1, positive and positive)
            ###############################################################################

            ## Real
            case1_real = (c1[:,idx_12]*d1[:,idx_12] + c1[:,idx_12]*d4[:,idx_12] +
                          c2[:,idx_12]*d2[:,idx_12] + c3[:,idx_12]*d3[:,idx_12])

         
            ## Deriv Real w.r.t phi
            d_case1_real_phi = (d_c1[:,idx_12]*d1[:,idx_12] +
                                d_c1[:,idx_12]*d4[:,idx_12] +
                                d_c2[:,idx_12]*d2[:,idx_12] +
                                d_c3[:,idx_12]*d3[:,idx_12])

            ## Imag
            case1_imag = (s1[:,idx_12]*d1[:,idx_12] +
                          s1[:,idx_12]*d4[:,idx_12] +
                          s2[:,idx_12]*d2[:,idx_12] +
                          s3[:,idx_12]*d3[:,idx_12])


            ## Deriv Imag w.r.t phi
            d_case1_imag_phi =  (d_s1[:,idx_12]*d1[:,idx_12] +
                                 d_s1[:,idx_12]*d4[:,idx_12] +
                                 d_s2[:,idx_12]*d2[:,idx_12] +
                                 d_s3[:,idx_12]*d3[:,idx_12])


            ##Total derivation
            gr_phi_case1   = case1_real*d_case1_real_phi + case1_imag*d_case1_imag_phi



            #################################################################################
            ## Total all cases 
            #################################################################################

            gr_phi_total = (np.concatenate((gr_phi_case1, gr_phi_case3, gr_phi_case2), axis = 1)/
                            normalization[np.newaxis,:])
            
            
            ################################################################################
            ########### Gradient
 
            self.gr_phi   = Qnorm1*np.sum(npmat.repmat(Qnorm,m,1)*gr_phi_total, 1)
            
            
                
        






