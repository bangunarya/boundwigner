##############################################################################################################
## This file contains functions to calculate derivative of the product spherical harmonics and derivative 
## of the product Wigner D-functions.
##
##############################################################################################################
from matrix import SphericalHarmonics
from matrix import WignerDfunctions
from itertools import combinations
import numpy as np
import numpy.matlib as npmat

def DerivProdSH(theta,phi,B,update):
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
    m = len(theta)
    N=B**2 
    phi_0 = np.zeros(m) #Generate only matrix associated Legendre by assigning phi = 0 
    A = np.real(SphericalHarmonics(theta,phi_0,B)[1])
    lk = SphericalHarmonics(theta,phi_0,B)[2]
    Aderiv = np.real(SphericalHarmonics(theta,phi_0,B)[3])
    col_comb = np.array(list(combinations(range(N),2)))
    comb_lk = [lk[col_comb[:,0],:],lk[col_comb[:,1],:]]
    ## Product of combination of degree and order associated Legendre 
    ProductasLeg = np.multiply(A[:,col_comb[:,0]],A[:,col_comb[:,1]]) 

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
    
    Qnorm = (q/2.0)*np.power(np.sqrt( np.power(abs(sum(np.multiply(ProductasLeg,mat_cos))),2)\
            + np.power( abs(sum(np.multiply(ProductasLeg,mat_sin))),2  )),(q-2))
    
    Qnorm1 = (1/q)*np.power(sum(np.power(np.sqrt( np.power(abs(sum(np.multiply(ProductasLeg,mat_cos))),2) \
            +  np.power(abs(sum(np.multiply(ProductasLeg,mat_sin))),2  )),q)),((1/q)-1))


    if update == 'all':
        ########################################################################################################
        ## Direct calculation of derivativei with respect to theta
        #######################################################################################################

        dPlktotal = np.multiply(A[:,col_comb[:,0]],Aderiv[:,col_comb[:,1]]) + np.multiply(Aderiv[:,col_comb[:,0]],A[:,col_comb[:,1]])

    
        ############################################################################################################
        ## Matrix-based derivative w.r.t theta
        ############################################################################################################
        gr_temp_theta1 = np.multiply((2.0*npmat.repmat(sum(np.multiply(ProductasLeg,mat_cos)),m,1)),(np.multiply(dPlktotal,mat_cos)))
        gr_temp_theta2 = np.multiply((2.0*npmat.repmat(sum(np.multiply(ProductasLeg,mat_sin)),m,1)),(np.multiply(dPlktotal,mat_sin)))
        gr_temp_theta = gr_temp_theta1 + gr_temp_theta2
        gr_theta = Qnorm1*np.sum(np.multiply(npmat.repmat(Qnorm,m,1),gr_temp_theta),axis = 1)

        ####################################################################################################
        ## Matrix-based derivative w.r.t phi
        ###################################################################################################
        Grad_temp1 =  2.0*npmat.repmat(sum(np.multiply(ProductasLeg,mat_cos)),m,1)
        Grad_temp2 = np.multiply(np.multiply(ProductasLeg,mat_sin),npmat.repmat(-k,m,1))
        Grad1 = np.multiply(Grad_temp1,Grad_temp2)

        Grad_temp3 = 2.0*npmat.repmat(sum(np.multiply(ProductasLeg,mat_sin)),m,1)
        Grad_temp4 = np.multiply(np.multiply(ProductasLeg,mat_cos),npmat.repmat(k,m,1))
        Grad2 = np.multiply(Grad_temp3,Grad_temp4)
        Grad_temp = Grad1 + Grad2

        gr_phi = Qnorm1*np.sum(np.multiply(npmat.repmat(Qnorm,m,1),Grad_temp),axis =1)

        return gr_theta, gr_phi

    else:

        Grad_temp1 =  2.0*npmat.repmat(sum(np.multiply(ProductasLeg,mat_cos)),m,1)
        Grad_temp2 = np.multiply(np.multiply(ProductasLeg,mat_sin),npmat.repmat(-k,m,1))
        Grad1 = np.multiply(Grad_temp1,Grad_temp2)

        Grad_temp3 = 2.0*npmat.repmat(sum(np.multiply(ProductasLeg,mat_sin)),m,1)
        Grad_temp4 = np.multiply(np.multiply(ProductasLeg,mat_cos),npmat.repmat(k,m,1))
        Grad2 = np.multiply(Grad_temp3,Grad_temp4)
        Grad_temp = Grad1 + Grad2
        
        gr_phi = Qnorm1*np.sum(np.multiply(npmat.repmat(Qnorm,m,1),Grad_temp),axis =1)


        return gr_phi
    

def DerivProdWigner(theta,phi,chi,B,update):
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


    m = len(theta)
    N = B*(2*B-1)*(2*B+1)//3
    
    phi_0 = np.zeros(m) #Generate only matrix Wigner d-functions by setting azimuth (phi) = 0
    chi_0 = np.zeros(m) #Generate only matrix Wigner d-functions by setting polarization (chi) = 0
    A = np.real(WignerDfunctions(theta,phi_0,chi_0,B)[1])
    lkn = WignerDfunctions(theta,phi_0,chi_0,B)[2]
    Aderiv = np.real(WignerDfunctions(theta,phi_0,chi_0,B)[3])
    col_comb = np.array(list(combinations(range(N),2)))
    comb_lkn = [lkn[col_comb[:,0],:],lkn[col_comb[:,1],:]]

    ## Product of combination degree and orders Wigner d-functions
    ProductWignerd = np.multiply(A[:,col_comb[:,0]],A[:,col_comb[:,1]])

    ########################################################################
    ## Matrix-based phi and chi
    #######################################################################
    k = comb_lkn[0][:,1] - comb_lkn[1][:,1]
    n = comb_lkn[0][:,2] - comb_lkn[1][:,2]

    ## Derivative of q norm
    q = 8.0

    mat_cos = np.cos(np.outer(phi,k) + np.outer(chi,n))
    mat_sin = np.sin(np.outer(phi,k) + np.outer(chi,n))

    Qnorm = (q/2.0)*np.power(np.sqrt( np.power(abs(sum(np.multiply(ProductWignerd,mat_cos))),2)\
                + np.power( abs(sum(np.multiply(ProductWignerd,mat_sin))),2  )),(q-2))
    Qnorm1 = (1/q)*np.power(sum(np.power(np.sqrt( np.power(abs(sum(np.multiply(ProductWignerd,mat_cos))),2) \
                 +  np.power(abs(sum(np.multiply(ProductWignerd,mat_sin))),2  )),q)),((1/q)-1))




    if update == 'all':

        ##################################################################################
        ## Direct derivative with respect to theta
        ##################################################################################
        dWignerd = np.multiply(A[:,col_comb[:,0]],Aderiv[:,col_comb[:,1]]) + np.multiply(Aderiv[:,col_comb[:,0]],A[:,col_comb[:,1]])
        
        ############################################################################################################
        ## Matrix-based derivative w.r.t to the q-norm
        ############################################################################################################
        gr_temp_theta1 = np.multiply((2.0*npmat.repmat(sum(np.multiply(ProductWignerd,mat_cos)),m,1)),(np.multiply(dWignerd,mat_cos)))
        gr_temp_theta2 = np.multiply((2.0*npmat.repmat(sum(np.multiply(ProductWignerd,mat_sin)),m,1)),(np.multiply(dWignerd,mat_sin)))
        gr_temp_theta = gr_temp_theta1 + gr_temp_theta2
        gr_theta = Qnorm1*np.sum(np.multiply(npmat.repmat(Qnorm,m,1),gr_temp_theta),axis = 1)
        
        
        
        ######################################################################
        ## Matrix-based phi
        #####################################################################
        Grad_temp1_phi =  2.0*npmat.repmat(sum(np.multiply(ProductWignerd,mat_cos)),m,1)
        Grad_temp2_phi =  np.multiply(np.multiply(ProductWignerd,mat_sin),npmat.repmat(-k,m,1))
        Grad1_phi = np.multiply(Grad_temp1_phi,Grad_temp2_phi)

        Grad_temp3_phi =  2.0*npmat.repmat(sum(np.multiply(ProductWignerd,mat_sin)),m,1)
        Grad_temp4_phi = np.multiply(np.multiply(ProductWignerd,mat_cos),npmat.repmat(k,m,1))
        Grad2_phi = np.multiply(Grad_temp3_phi,Grad_temp4_phi)
        Grad_temp_phi = Grad1_phi + Grad2_phi

        ######################################################################
        ## Matrix-based chi
        ######################################################################

        Grad_temp1_chi =  2.0*npmat.repmat(sum(np.multiply(ProductWignerd,mat_cos)),m,1)
        Grad_temp2_chi =  np.multiply(np.multiply(ProductWignerd,mat_sin),npmat.repmat(-n,m,1))
        Grad1_chi = np.multiply(Grad_temp1_chi,Grad_temp2_chi)

        Grad_temp3_chi =  2.0*npmat.repmat(sum(np.multiply(ProductWignerd,mat_sin)),m,1)
        Grad_temp4_chi = np.multiply(np.multiply(ProductWignerd,mat_cos),npmat.repmat(n,m,1))
        Grad2_chi = np.multiply(Grad_temp3_chi,Grad_temp4_chi)
        Grad_temp_chi = Grad1_chi + Grad2_chi

        gr_phi = Qnorm1*np.sum(np.multiply(npmat.repmat(Qnorm,m,1),Grad_temp_phi),axis = 1)
        gr_chi = Qnorm1*np.sum(np.multiply(npmat.repmat(Qnorm,m,1),Grad_temp_chi),axis = 1)
        
        return gr_theta, gr_phi, gr_chi
    else:
        
        ######################################################################
        ## Matrix-based phi
        #####################################################################
        Grad_temp1_phi =  2.0*npmat.repmat(sum(np.multiply(ProductWignerd,mat_cos)),m,1)
        Grad_temp2_phi =  np.multiply(np.multiply(ProductWignerd,mat_sin),npmat.repmat(-k,m,1))
        Grad1_phi = np.multiply(Grad_temp1_phi,Grad_temp2_phi)

        Grad_temp3_phi =  2.0*npmat.repmat(sum(np.multiply(ProductWignerd,mat_sin)),m,1)
        Grad_temp4_phi = np.multiply(np.multiply(ProductWignerd,mat_cos),npmat.repmat(k,m,1))
        Grad2_phi = np.multiply(Grad_temp3_phi,Grad_temp4_phi)
        Grad_temp_phi = Grad1_phi + Grad2_phi

        ######################################################################
        ## Matrix-based chi
        ######################################################################

        Grad_temp1_chi =  2.0*npmat.repmat(sum(np.multiply(ProductWignerd,mat_cos)),m,1)
        Grad_temp2_chi =  np.multiply(np.multiply(ProductWignerd,mat_sin),npmat.repmat(-n,m,1))
        Grad1_chi = np.multiply(Grad_temp1_chi,Grad_temp2_chi)

        Grad_temp3_chi =  2.0*npmat.repmat(sum(np.multiply(ProductWignerd,mat_sin)),m,1)
        Grad_temp4_chi = np.multiply(np.multiply(ProductWignerd,mat_cos),npmat.repmat(n,m,1))
        Grad2_chi = np.multiply(Grad_temp3_chi,Grad_temp4_chi)
        Grad_temp_chi = Grad1_chi + Grad2_chi

        gr_phi = Qnorm1*np.sum(np.multiply(npmat.repmat(Qnorm,m,1),Grad_temp_phi),axis = 1)
        gr_chi = Qnorm1*np.sum(np.multiply(npmat.repmat(Qnorm,m,1),Grad_temp_chi),axis = 1)

        
        
        
        
        return gr_phi,gr_chi




