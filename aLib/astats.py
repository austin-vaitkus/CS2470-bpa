"""
This module provides some comfort functions for doing stats, particularly random-number
generation, in situations where the builtin scipy libraries may lack certain features.

To get an up-to-date list of available functions, try:

>>> from aLib.misc import getFuns
>>> import astats
>>> getFuns(astats)

Each function therein should have adequate self documentation.

2014.04.26 -- A. Manalaysay
"""

from scipy import stats
from scipy import rand
import numpy as np

def binornd(n, p, size=[1]):
    """
    Generate an ndarray of random numbers chosen from a binomial distribution.  The user
    can specify a single n, p, or arrays of each.  If n and p are scalars, the array 
    size can be specified with the size input.
    
    ------ Inputs:
           n: Number of trials.  If n is an ndarray, each random number generated 
              separately corresponds to an entry in n.
           p: Probability of success.  If p is an ndarray, each random number generated 
              separately corresponds to an entry in p.  If both n and p are ndarrays, 
              they must be identical dimensions.
        size: If both n and p are scalars, the size of the output ndarray is specified
              by this parameter, which is an iterable (ndarray, list, tuple).  If n 
              and/or p are arrays, size is ignored.
    ----- Outputs:
        bOut: ndarray of binomial random numbers.
    
    Example:
    In [1]: n = round(5*rand(20)+10)
    In [2]: bRands = binornd(n, 0.4)
    
    2014.04.26 -- A. Manalaysay
    """
    # -- <Input checking> -- #
    if type(n) != np.ndarray:
        if (type(n) == tuple) or (type(n) == list):
            n = np.array(n)
        else:
            n = np.array([n])
    if type(p) != np.ndarray:
        if (type(p) == tuple) or (type(p) == list):
            p = np.array(p)
        else:
            p = np.array([p])
    if (n.size != 1) or (p.size != 1):
        flag_useSize = False
    else:
        flag_useSize = True
    if (n.size != 1) and (p.size != 1):
        if (n.size != p.size) or (n.ndim != p.ndim) or (n.shape[0] != p.shape[0]):
            raise ValueError('If both n and p are arrays, they must be of equal dimensions.')
    if np.any(np.round(n) != n):
        raise ValueError('All elements of n must be of integer value (but not necessarily integer type).')
    if any(p<0) or any(p>1):
        raise ValueError('All elements of p must live between zero and one.')
    if flag_useSize:
        if (type(size) != tuple) and (type(size) != list) and (type(size) != ndarray):
            raise ValueError("Input 'size' must be a tuple, list, or ndarray.")
    # -- </Input checking> -- #
    
    # Generate random numbers in the case that n and p are scalars, according to input 'size'
    if flag_useSize:
        bOut = stats.binom.rvs(n, p, size=size)
    else:
        if n.size == 1:
            n = n*np.ones_like(p)
        if p.size == 1:
            p = p*np.ones_like(n)
        bOut = np.zeros_like(n)
        for k in range(1,np.uint64(n.max()+1)):
            cutN = n >= k
            bOut[cutN] += rand(cutN.sum()) < p[cutN]
    
    if len(bOut) == 1:
        bOut = bOut[0]
    return bOut
    

def doubleExpPDF(x, L1, L2, a):
    """
    Double exponential distribution PDF.  This is not the "cuspy" Laplace distribution, nor the Gumbel
    distribution, both of which show up in a google search for "double exponential distribution".  This 
    is instead simply the distribution whose PDF is the sum of two exponentials:
    
    doubleExpPDF(x, L1, L2, a) = N*(exp(-L1*x) + a*exp(-L2*x)), 
    
    where N is the normalization constant, given by:
    
    N = L1 * L2 / (a*L1 + L2)
    
    ----------------------Inputs:
        x: Independent variable.
       L1: Exponential constant of the first component.
       L2: Exponential constant of the second component.
        a: Relative weight of the second component relative to the first.
    ---------------------Outputs:
       Output is the value of the PDF evaluated at x, given the three free parameters.
    
    2014.06.26 -- A. Manalaysay
    """
    N = L1 * L2 / (a*L1 + L2)
    outPDF = N * (np.exp(-L1*x) + a*np.exp(-L2*x))
    return outPDF

def doubleExpCDF(x, L1, L2, a):
    """
    Double exponential distribution CDF.  See the definition of this distribution in 'doubleExpPDF'.
    
    2014.06.26 -- A. Manalaysay
    """
    N = L1 * L2 / (a*L1 + L2)
    outCDF = N * ((1-np.exp(-L1*x))/L1 + a*(1-np.exp(-L2*x))/L2)
    return outCDF


from scipy.special import erf
def efficUncert(n, k):
    """
    Calculates proper error bars for the efficiency given by k/n.  This
    assumes that n_c came directly from the same data as what went into n,
    but with some cuts.  Treats the efficiency, ep, as the dependent variable
    in a binomial likelihood: L = ep^k * (1-ep)*(n-k)
    WARNING: no input checking at the moment.
    NOTE: When n = k = 0, it assumes the estimate for k/n is zero, and then
    erUp = 0.683 and erDn = 0.  Technically, it should be that k/n = 0.5, but
    whatever.
    
     inputs:
            n: Histogram of original data before cuts.
            k: Histogram of data after cuts.
    outputs:
         erUp: Vector of upward errorbars (1-sig).
         erDn: Vector of downward errorbars (1-sig).
    
    2010.03.21  -  AGM
    
    """
    raise ValueError("THIS FUNCTION ISN'T WORKING PROPERLY YET, sorry!!")
    ep = np.linspace(0,1,1e3)
    Lm = k/n  # Lm = likelihood max
    Lm00 = np.isnan(Lm)
    Lm[Lm00] = 0
    a1sig = erf(1/np.sqrt(2)) # ~68.3%
    erUp = np.zeros_like(n)
    erDn = np.zeros_like(n)
    for t in range(len(n)):
        if not Lm00[t]:
            Lik = (ep**k[t])*(1-ep)**(n[t]-k[t])
            Lik = Lik/Lik.sum()
            prop = np.linspace(0,Lik.max()*1.01)
            propArea = np.zeros_like(prop)
            for t1 in range(len(prop)):
                propArea[t1] = Lik[Lik>=prop[t1]].sum()
            prop1sig = prop[np.nonzero(propArea>a1sig)[0].max()]
            erUp[t] = ep[np.nonzero(Lik>=prop1sig)[0].max()]-Lm[t]
            erDn[t] = Lm[t] - ep[np.nonzero(Lik>=prop1sig)[0].max()]
        else:
            erUp[t] = a1sig
            erDn[t] = 0
    return erUp, erDn


















