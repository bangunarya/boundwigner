from scipy.io import savemat
import numpy as np
import os
class StoreData: 
    
    
    def __init__(self, m, N, path):
        self.m = m
        self.N = N
        self.path = path
        self.grad_algo = ['graddes','adam','adadelta',
                          'adagrad','legendre','welch']
        
        self.coh = np.zeros((len(self.grad_algo),len(self.m)), dtype = np.float64)
        self.angles = np.zeros((len(self.grad_algo),len(self.m),self.N,3), dtype = np.float64) 
          
    def coh_angle_handle(self,ii, gr_algo, bound):
        ## Calculate gradient
        self.coh[0,ii] = gr_algo.graddes_coh # use simple gradient descent
        self.angles[0,ii,0:self.m[ii], 0:gr_algo.graddes_ang.shape[1]] = gr_algo.graddes_ang
        print("GrDe     -- Samples %s, Coherence  %s,  Welch Bound %s, and Legendre Bound %s"
              %(self.m[ii], gr_algo.graddes_coh, bound[0], bound[1])) 
        
        
        self.coh[1,ii] = gr_algo.adam_coh # use adam 
        self.angles[1,ii,0:self.m[ii], 0:gr_algo.adam_ang.shape[1]] = gr_algo.adam_ang
        print("ADAM     -- Samples %s, Coherence  %s,  Welch Bound %s, and Legendre Bound %s" 
              %(self.m[ii], gr_algo.adam_coh, bound[0], bound[1]))
    
  
        self.coh[2,ii] = gr_algo.adadelta_coh # use adadelta
        self.angles[2,ii,0:self.m[ii], 0:gr_algo.adadelta_ang.shape[1]] = gr_algo.adadelta_ang
        print("AdaDelta -- Samples %s, Coherence  %s,  Welch Bound %s, and Legendre Bound %s" 
              %(self.m[ii], gr_algo.adadelta_coh, bound[0], bound[1]))   
    
    
        self.coh[3,ii] = gr_algo.adagrad_coh # use adagrad
        self.angles[3,ii,0:self.m[ii], 0:gr_algo.adagrad_ang.shape[1]] = gr_algo.adagrad_ang
        print("AdaGrad  -- Samples %s, Coherence  %s,  Welch Bound %s, and Legendre Bound %s"
              %(self.m[ii], gr_algo.adagrad_coh, bound[0], bound[1]))
                    
        self.coh[4,ii] = bound[1]
        self.coh[5,ii] = bound[0]
        
        ## Store data
        if ii == len(self.m) - 1:
            os.makedirs(self.path, exist_ok = True)
            path_data = os.path.join(self.path, 'data.mat')
            data = {"m":self.m,
                    "N": self.N,
                    "Coherence": self.coh,
                    "GradientDescent": self.angles[0,:,:,:],
                    "ADAM":  self.angles[1,:,:,:],
                    "AdaDelta":  self.angles[2,:,:,:],
                    "AdaGrad":  self.angles[3,:,:,:]}
            
            
            savemat(path_data, data)