#############################################################################################################
## This file contains several implementations of sampling points on the sphere
## Equiangular, Spiral, Fibonacci, Hammersley, Random on pole, Uniform random
##
##
############################################################################################################
import numpy as np

class SamplingPoints:
    
    def __init__(self, m, types):
        self.m = m
        self.types = types
        self.choosesampling()
        
    def choosesampling(self):
        self.angles = np.zeros((self.m,3),dtype = np.float64)
        
        if self.types[0] == 'equiangular':
            self.equiangularsampling()
            
        elif self.types[0] == 'spiral':
            self.spiralsampling()
            
        elif self.types[0] == 'fibonacci':
            self.fibonaccisampling()
            
        elif self.types[0] == 'hammersley':
            self.hammersleysampling()
            
        elif self.types[0] == 'polerandom':
            self.polerandom()
        
        elif self.types[0] == 'uniformrandom':
            self.uniformrandom()
        
        else:
            self.initialsnf()
            
        
    
    def equiangularsampling(self):
        theta = np.linspace(0,np.pi,self.m)
        phi = np.linspace(0,2.0*np.pi,self.m)
        
        if self.types[1] == 'random':
            chi = np.random.rand(self.m)*2.0*np.pi
        else:
            chi = np.linspace(0,2*np.pi,self.m)
            
        self.angles[:,0] = theta
        self.angles[:,1] = phi
        self.angles[:,2] = chi
        
        
    def spiralsampling(self):
        l = np.arange(self.m)
        theta = np.arccos(-1 + (2.0*(l)/(self.m-1)))
        C = 3.6
        phi = np.zeros(self.m)
        for l in range(1,self.m-1):
            h = -1 + (2.0*l)/(self.m-1)
            phi[l] = (phi[l-1] + (C/np.sqrt(self.m))*(1/np.sqrt(1-h**2.0))) % (2.0*np.pi)
        
        
        if self.types[1] == 'random':
            chi = np.random.rand(self.m)*2.0*np.pi
        else:
            chi = np.linspace(0,2*np.pi,self.m)
            
        self.angles[:,0] = theta
        self.angles[:,1] = phi
        self.angles[:,2] = chi


    def fibonaccisampling(self):
        N = int(np.ceil((self.m-1)/2.0))  # Has to be an odd number of points.
        theta = np.zeros(2*N + 1)
        phi = np.zeros(2*N + 1)
        gr = (1 + np.sqrt(5))/(2.0)
        k = 0
        for i in range(-N,N + 1):
            lat = np.arcsin(2.0*i/(2.0*N + 1))
            lon = 2.0*np.pi*i/gr
            theta[k] = np.pi/2 - lat
            phi[k] = np.atan2(np.cos(lat)*np.sin(lon), np.cos(lat)*np.cos(lon))
            if phi[k] < 0:
                phi[k] = phi[k] + 2.0*np.pi

            k += 1
        
        self.angles = np.zeros((2*N + 1,3),dtype = np.float64)
        
        if self.types[1] == 'random':
            chi = np.random.rand(2*N+1)*2.0*np.pi
        else:
            chi = np.linspace(0,2*np.pi,2*N + 1)
            
        self.angles[:,0] = theta
        self.angles[:,1] = phi
        self.angles[:,2] = chi 
        
    def hammersleysampling(self):
        ############################################################
        ## Generate van der Corput sequence according to 
        ########################################################
        def vdcorput(k,b):
            s = np.zeros(k+1)
            for i in np.arange(k)+1:
                a = basexpflip(i,b)
                g = np.power(b,np.arange(len(a))+1)
                s[i] = sum(np.divide(a,g))
            return s
        def basexpflip(k,b): # reversed base-b expansion of positive integer k
            j = int(np.fix(np.log(k)/np.log(b))) + 1
            a = np.zeros(j)
            q = b**(j-1)
            for ii in range(j):
                a[ii] = int(np.floor(k/q))
                k = k - q*a[ii]
                q = q/b
            a = a[::-1]
            return a
        ###################################################################
        ##
        ##################################################################
        t = vdcorput(self.m,2)
        t = 2*t[0:-1] - 1
        theta = np.arccos(t)
        phi = 2*np.pi*((2*(np.arange(self.m)+1)-1)/2.0/self.m)
        phi[phi < 0] = phi[phi < 0] + 2.0*np.pi
        
        
        if self.types[1] == 'random':
            chi = np.random.rand(self.m)*2.0*np.pi
        else:
            chi = np.linspace(0,2*np.pi,self.m)
            
        self.angles[:,0] = theta
        self.angles[:,1] = phi
        self.angles[:,2] = chi
        
    def polerandom(self):
        theta = np.random.rand(self.m)*np.pi
        phi   = np.random.rand(self.m)*2.0*np.pi
        chi = np.random.rand(self.m)*2.0*np.pi
        
        if self.types[1] == 'random':
            chi = np.random.rand(self.m)*2.0*np.pi
        else:
            chi = np.linspace(0,2*np.pi,self.m)
            
        self.angles[:,0] = theta
        self.angles[:,1] = phi
        self.angles[:,2] = chi
    
    def uniformrandom(self):
        theta = np.arcos(2.0*np.random.rand(self.m) - 1)
        phi = np.random.rand(self.m)*2.0*np.pi
        chi = np.random.rand(self.m)*2.0*np.pi
        
        if self.types[1] == 'random':
            chi = np.random.rand(self.m)*2.0*np.pi
        else:
            chi = np.linspace(0,2*np.pi,self.m)
            
        self.angles[:,0] = theta
        self.angles[:,1] = phi
        self.angles[:,2] = chi  
        
    def initialsnf(self):
        theta = np.random.rand(self.m)*np.pi
        #theta = np.arccos(2.0*np.random.rand(self.m) - 1)
        if self.types[1] == 'partial':
            theta = np.arccos(np.linspace(-1,1,self.m)) 
        #
        
        phi   = np.random.rand(self.m)*2.0*np.pi
        chi = (np.arange(self.m) % 2)*(np.pi/2)
        
        self.angles[:,0] = theta
        self.angles[:,1] = phi
        self.angles[:,2] = chi
    # def Pole_equatorrandom():
 