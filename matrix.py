from scipy.special import sph_harm as SH
from scipy.special import eval_jacobi as Plkn
import numpy as np
import math
from numpy import linalg as LA



class Matrix:
    '''
    '''
    def __init__(self, B, types, angles):
        self.B = B
        self.types = types 
        self.angles = angles
        self.degree_orders()
        
    def degree_orders(self):
        '''
        Generating combination of degree and orders for spherical
        harmonics and Wigner D-functions
        '''
        
        self.m = self.angles.shape[0]
        if self.types[0] == 'SH':
            self.N = self.B**2
            lk = np.zeros((self.N,2), dtype = np.float64)
            idx_beg = 0
            for l in range(self.B):
                k = range(-l,l+1)
                idx = len(k)
                idx_end = idx_beg + idx-1
                lk[idx_beg:idx_end+1,1] = k
                lk[idx_beg:idx_end+1,0] = np.full((1,idx),l)
                idx_beg = idx_beg + idx
            self.deg_order = lk

            self.sphericalharmonics()

        elif self.types[0] == 'Wigner':
            self.N = self.B*(2*self.B-1)*(2*self.B+1)//3
            lkn = np.zeros((self.N,3), dtype = np.float64)
            idx_beg = 0
            for l in range(self.B):
                n = range(-l,l+1)
                k = range(-l,l+1)
                mesh_k_n = np.array(np.meshgrid(k, n))
                k_n = mesh_k_n.T.reshape(-1, 2)
                idx = len(n)**2
                idx_end = idx_beg + idx-1
                lkn[idx_beg:idx_end + 1,0] = np.full((1,idx),l)
                lkn[idx_beg:idx_end + 1,1:] = k_n 
                idx_beg = idx_beg + idx
            self.deg_order = lkn           
        
            self.wignerDfunctions()

        else:
            self.N = self.B**2 + 2*self.B
            lk = np.zeros((self.N,2), dtype = np.float64)
            idx_beg = 0
            for l in np.arange(1,self.B + 1):
                k = range(-l,l+1)
                idx = len(k)
                idx_end = idx_beg + idx-1
                lk[idx_beg:idx_end+1,1] = k
                lk[idx_beg:idx_end+1,0] = np.full((1,idx),l)
                idx_beg = idx_beg + idx
            self.deg_order = lk
             
            self.wigner_snf()
                
    def sphericalharmonics(self):
    
        A = np.zeros((self.m,self.N), dtype = np.complex)
        normA = np.copy(A)
        dPlk = np.zeros((self.m,self.N), dtype = np.float64)
        
        theta = self.angles[:,0]
        phi = self.angles[:,1]
        
        ## Asign phi = 0 to get associated Legendre
        phi_0 = np.zeros(self.m, dtype = np.float64)
        Plk = np.copy(dPlk)
        
        #####################################################################################################
        ## Generating Spherical Harmonics Matrix and their derivative
        ## with respect to theta (Derivative of associated Legendre polynomials)
        ## d/dtheta ) = k/tan(theta) Ylk(theta,phi) + sqrt((l-k)(l+k+1))*Yl(k+1)(theta,phi)
        ##
        ## The spherical harmonics when k+1 > l will be zero because of multiplication with sqrt((l-k)(l+k+1)
        ######################################################################################################

        for ii in range(self.N):
            
            ### Generate spherical harmonics and their (column) normalization w.r.t sampling points
            A[:,ii] = SH(self.deg_order[ii,1],self.deg_order[ii,0],phi,theta)
            normA[:,ii] = A[:,ii]/LA.norm(A[:,ii])
            
            ## Get Associated Legendre (get real since we have phi_0 = 0)
            Plk[:,ii] = np.real(SH(self.deg_order[ii,1], 
                           self.deg_order[ii,0], 
                           phi_0, theta))/LA.norm(A[:,ii]) #note that absolute value 
                                                          #spherical harmonics = absolute value associated Legendre
                 
       
            
            
            ## Calculate derivative w.r.t theta if choose all
            ### Spherical harmonics order > degree, 
            ## just assign arbitrary since will be zero multiply with the coefficients
            
            if self.types[1] == 'all':
                if self.deg_order[ii,1] + 1 > self.deg_order[ii,0]:
                    Plk_lastterm = np.ones(self.m)
                else:
                ## (get real since we have phi_0 = 0)
                    Plk_lastterm = (np.real(np.exp(-1j*phi_0)*SH(self.deg_order[ii,1] + 1,self.deg_order[ii,0],phi_0,theta))/
                               LA.norm(SH(self.deg_order[ii,1] + 1 , self.deg_order[ii,0], phi_0, theta)))
            
            
            
                ### Derivative of spherical harmonics w.r.t theta, or derivative of associated Legendre functions
                Plk_deriv = ((self.deg_order[ii,1]/np.tan(theta))*Plk[:,ii] + 
                             np.sqrt((self.deg_order[ii,0] - self.deg_order[ii,1])*
                                 (self.deg_order[ii,0] + self.deg_order[ii,1] + 1))*
                             Plk_lastterm)       
                dPlk[:,ii] = Plk_deriv
        
        self.A = A
        self.normA = normA
        self.dPlk = dPlk
        self.Plk = Plk
    

    
    def wignerDfunctions(self):
        ############################################################################
        ## Generating Wigner D-functions matrix and derivative of Wigner d-functions
        ############################################################################
        A = np.zeros((self.m, self.N), dtype = np.complex64)
        normA = np.copy(A)
        dWignerd = np.zeros((self.m, self.N), dtype = np.float64)
        Wignerd = np.copy(dWignerd)
    
        theta = self.angles[:,0]
        phi = self.angles[:,1]
        chi = self.angles[:,2]
        
        ## Asign phi, chi = 0 to get Wigner (small) d functions
        phi_0 = np.zeros(self.m, dtype = np.float64)
        chi_0 = np.copy(phi_0)
        
        
        for ii in range(self.N):
        

            #########################################################################
            ## Set initial parameters
            #########################################################################
            if self.deg_order[ii,2] >= self.deg_order[ii,1]:
                eta = 1
            else:
                eta = (-1)**(self.deg_order[ii,2] - self.deg_order[ii,1])

            #########################################################################
            ## Set Normalization
            #########################################################################
            Normalization = np.sqrt((2.0*self.deg_order[ii,0]+1)/(8.0*np.pi**2))
            mu_plus = np.abs(self.deg_order[ii,1] - self.deg_order[ii,2])
            vu_plus = np.abs(self.deg_order[ii,1] + self.deg_order[ii,2])
            s_plus = self.deg_order[ii,0] - (mu_plus + vu_plus)/2.0
        
            Norm_Gamma = np.sqrt((math.factorial(s_plus)*math.factorial(s_plus+mu_plus+vu_plus))/
                                 (math.factorial(s_plus+mu_plus)*(math.factorial(s_plus+vu_plus))))
            ###########################################################################
            ## Generate Wigner d-functions
            ###########################################################################

            Wignerd[:,ii] =  (Normalization*eta*Norm_Gamma*
                       (np.sin(theta/2.0)**mu_plus)*(np.cos(theta/2.0)**vu_plus)*
                       (Plkn(s_plus,mu_plus,vu_plus,np.cos(theta))))

            #################################################################################################
            ## Generate Wigner D-functions sensing matrix and their (column) normalization
            ################################################################################################

            A[:,ii] = np.exp(-1j*self.deg_order[ii,1]*phi)*Wignerd[:,ii]*np.exp(-1j*self.deg_order[ii,2]*chi)
            normA[:,ii] = A[:,ii]/LA.norm(A[:,ii])
            
            Wignerd[:,ii] =  Wignerd[:,ii]/LA.norm(A[:,ii]) 
            
            if self.types[1] == 'all':
            
            ###################################################################################################
            ## Calculate derivative of Wigner d-functions
            ###################################################################################################
            
                ## Jacobi polynomials
                Jacobi_last = (Plkn(s_plus - 1, mu_plus + 1, vu_plus + 1,np.cos(theta)))
            
                ## derivative
                Wignerd_deriv =((-(mu_plus*np.sin(theta)**2)/(2.0*(1 - np.cos(theta))) + 
                                  (vu_plus*np.sin(theta)**2)/(2.0*(1 + np.cos(theta))))*Wignerd[:,ii] - 
                                  (np.sin(theta)*Normalization*eta*Norm_Gamma*(mu_plus + vu_plus + s_plus + 1)*0.5*
                                  (np.sin(theta/2.0)**mu_plus)*(np.cos(theta/2.0)**vu_plus)*
                                  (Jacobi_last/LA.norm(A[:,ii]))))
                dWignerd[:,ii] = Wignerd_deriv
        

        self.A = A
        self.normA = normA
        self.Wignerd = Wignerd
        self.dWignerd = dWignerd


    def wigner_snf(self):
        
        ############################################################################
        ## Generating Wigner D-functions matrix for SNF and its derivative on theta
        ############################################################################
 
        ## Normalization
        normalize1 = np.zeros((self.N), dtype = np.float64)
        normalize2 = np.zeros((self.N), dtype = np.float64)

        ### Alocate the matrix
        Basis_1 = np.zeros((self.m, self.N), dtype = np.complex64)
        Basis_2 = np.copy(Basis_1)
        
        norm_Basis_1 = np.copy(Basis_1)
        norm_Basis_2 = np.copy(Basis_2)


        dmm_plus = np.zeros((self.m, self.N), dtype = np.float64)
        dmm_min  = np.zeros((self.m, self.N), dtype = np.float64)

        ### allocate derivative
        d_dmm_plus = np.zeros((self.m, self.N), dtype = np.float64)
        d_dmm_min = np.copy(d_dmm_plus)
        
        ## Angles
        theta = self.angles[:,0]
        phi = self.angles[:,1]
        chi = self.angles[:,2]

        #####
        def eta_val(m,mu):
            if mu >= m:
                eta = 1
            else:
                eta = (-1)**(mu-m)
            return eta


        for ii in range(self.N):
            Normalization = math.sqrt((2.0*self.deg_order[ii,0]+1)/(8.0*np.pi**2))
            
            #########################################################################
            ## Set Wigner d for positive
            #########################################################################
            mu_plus = np.abs(self.deg_order[ii,1] - 1)
            vu_plus = np.abs(self.deg_order[ii,1] + 1)
            s_plus = self.deg_order[ii,0] - (mu_plus + vu_plus)/2.0
            

            Norm_Gamma = np.sqrt((math.factorial(s_plus)*math.factorial(s_plus + mu_plus + vu_plus))/
                                 (math.factorial(s_plus + mu_plus)*(math.factorial(s_plus + vu_plus))))

            dmm_plus[:,ii] =(eta_val(self.deg_order[ii,1],1)*Norm_Gamma*(np.sin(theta/2)**mu_plus)*(np.cos(theta/2)**vu_plus)*
                             Plkn(s_plus,mu_plus,vu_plus,np.cos(theta)))

            
            ## Derivative w.r.t theta
            if self.types[1] == 'all':
                ## Jacobi polynomials
                Jacobi_last_plus = (Plkn(s_plus - 1, mu_plus + 1, vu_plus + 1,np.cos(theta)))

                ## derivative
                Wignerd_deriv_plus = ((-(mu_plus*np.sin(theta)**2)/(2.0*(1 - np.cos(theta))) +
                                     (vu_plus*np.sin(theta)**2)/(2.0*(1 + np.cos(theta))))*dmm_plus[:,ii] -
                                     (np.sin(theta)*Normalization*eta_val(self.deg_order[ii,1],1)*Norm_Gamma*
                                      (mu_plus + vu_plus + s_plus + 1)*0.5*
                                     (np.sin(theta/2.0)**mu_plus)*(np.cos(theta/2.0)**vu_plus)*
                                     (Jacobi_last_plus)))
                ## Store derivatives
                d_dmm_plus[:,ii] = Wignerd_deriv_plus


            

            #########################################################################
            ## Set Wigner d for negative
            ########################################################################
            
            mu_min = np.abs(self.deg_order[ii,1] + 1)
            vu_min = np.abs(self.deg_order[ii,1] - 1)
            s_min = self.deg_order[ii,0] - (mu_min + vu_min)/2.0

            Norm_Gamma = np.sqrt((math.factorial(s_min)*math.factorial(s_min + mu_min + vu_min))/
                                 (math.factorial(s_min + mu_min)*math.factorial(s_min + vu_min)))

            dmm_min[:,ii] = (eta_val(self.deg_order[ii,1],-1)*Norm_Gamma*(np.sin(theta/2)**mu_min)*(np.cos(theta/2)**vu_min)*
                             Plkn(s_min,mu_min,vu_min,np.cos(theta)))
            

            if self.types[1] == 'all':
                ## Jacobi polynomials
                Jacobi_last_min = (Plkn(s_min - 1, mu_min + 1, vu_min + 1,np.cos(theta)))

                ## derivative
                Wignerd_deriv_min = ((-(mu_min*np.sin(theta)**2)/(2.0*(1 - np.cos(theta))) +
                                     (vu_min*np.sin(theta)**2)/(2.0*(1 + np.cos(theta))))*dmm_min[:,ii] -
                                     (np.sin(theta)*Normalization*eta_val(self.deg_order[ii,1],-1)*
                                     Norm_Gamma*(mu_min + vu_min + s_min + 1)*0.5*
                                     (np.sin(theta/2.0)**mu_min)*(np.cos(theta/2.0)**vu_min)*
                                     (Jacobi_last_min)))

                ### Store derivatives

                d_dmm_min[:,ii] = Wignerd_deriv_min
            

            ## Generate Wigner D for s = 1 (TE) and s = 2 (TM)
            Basis_1[:,ii] = (np.exp(1j*chi)*dmm_plus[:,ii] + np.exp(-1j*chi)*
                             dmm_min[:,ii])*np.exp(1j*self.deg_order[ii,1]*phi)
    
            norm_Basis_1[:,ii] = Basis_1[:,ii]/LA.norm(Basis_1[:,ii])


            Basis_2[:,ii] = (np.exp(1j*chi)*dmm_plus[:,ii] - np.exp(-1j*chi)*
                             dmm_min[:,ii])*np.exp(1j*self.deg_order[ii,1]*phi)

            norm_Basis_2[:,ii] = Basis_2[:,ii]/LA.norm(Basis_2[:,ii])

            ### Store normalization
            normalize1[ii] = LA.norm(Basis_1[:,ii])
            normalize2[ii] = LA.norm(Basis_2[:,ii])
        

        ### Concatenate to get matrix for SNF
        A =  np.concatenate((Basis_1,Basis_2),axis = 1)
        normA = np.concatenate((norm_Basis_1, norm_Basis_2), axis = 1)

        normalize = np.concatenate((normalize1, normalize2), axis = 0)

        
        self.A = A
        self.normA = normA
        self.d_dmm_plus = d_dmm_plus
        self.d_dmm_min = d_dmm_min  
        self.dmm_plus = dmm_plus
        self.dmm_min = dmm_min
        self.normalize = normalize
