############################################################################################
## This function contains collection of all metrics, for example to calculate coherence,
## the coherence bound maximum product of same degree and orders
##
##
############################################################################################


import numpy as np
from itertools import combinations
from scipy.special import lpmv as asLeg
from numpy import linalg as LA

#######################################################
## Calculate Coherence of a column normalize matrix
## Input : Matrix with normalize column
## Output : Coherence
#######################################################

def Coherence(normA):
    N = len(normA[0,:])
    Gram = np.dot(normA.conjugate().T,normA)
    Coherence = np.max(abs(Gram - np.identity(N)))

    return Coherence

#######################################################
## Bound Coherence
## Calculate Welch bound and the coherence bound in the
## paper
##
## Input : row and column dimension, bandlimited B
## Outout : Welchbound and Bound in the paper
######################################################

def BoundCoherence(m,N,B):
    Welch = np.sqrt((N-m)/((N-1.0)*m))
    x = np.linspace(-1,1,m)
    PB1 = asLeg(0,B-1,x)
    PB3 = asLeg(0,B-3,x)
    Legbound = abs(np.inner((PB1/LA.norm(PB1)),(PB3/LA.norm(PB3))))
    return Welch, Legbound

###########################################################
## Maximum of the product same degree and orders
## 
## Input : normalized matrix, and combination degree orders
## Output : maximum product for same orders k1 = k2, n1 = n2
##
############################################################
def Maxprod(normA,lkn):
    N = len(normA[0,:])

    prod = []

    for subset in combinations(range(N),2):
        comb_column = np.array(subset)
        comb_lkn = [lkn[comb_column[0],:],lkn[comb_column[1],:]]
        k = comb_lkn[0][1] - comb_lkn[1][1]
        n = comb_lkn[0][2] - comb_lkn[1][2]
        knzero = abs(k) + abs(n)
        
        if knzero == 0:


            prod.append((normA[:,comb_column[0]]).dot(normA[:,comb_column[1]]))
    
        
    max_prod = max(abs(np.array(prod)))

    return max_prod
