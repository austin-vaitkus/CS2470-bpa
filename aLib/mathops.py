"""
Special functions.  Try "misc.getFuns(spfuns)" for a list of functions from the command line.

2014.05.02 -- A. Manalaysay
"""

import numpy as np
from scipy import special as sp

def cpoiss(x, a, b, L):
    """
    Continuous Poisson function.  NOTE: this is a FUNCTION, not a distribution.  Here, the familiar, 
    discrete Poisson distribution is promoted to a continuous function.  Unlike the normal approach, 
    where the factorial (which appears in the denominator of the Poisson distribution) is converted 
    to a gamma function, here a more computationally friendly method is used.  The factorial is 
    replaced by its Stirling approximation.  This is advantageous because the factorial appears only 
    in a ratio; therefore the code must calculate the ratio of two very large numbers.  By using the 
    Stirling approximation, this ratio can essentially be computed directly.
    
    The Stirling approximation is:
        n! ~= n^n * exp(-n) * sqrt(2*pi*n)
    The Poisson distribution is:
        P(k,L) = (L^k) * exp(-L) / (k!)
    The Continuous Poisson is:
        f(x,a,b,L) = a * N * (b*L*e/x)^(x/b) * (2*pi*x/b)^(-1/2)
        N is the normalization constant, such that the function is of unit height when a=1.
        The maximum of this function occurs at x=x_max, where
            x_max = -b / (2 * W(-1/2/L))
        where W() is the Lambert W function (provided by scipy.special.lambertw).
    
    Input parameters:
        x: Independent variable.
        a: Height normalization parameter.
        b: Scale parameter.
        L: Shape parameter.
    
    2014.05.05 -- A. Manalaysay
    """
    """
    # Define the template function
    f0 = lambda x, b, L: ((b*L*np.exp(1)/x)**(x/b)) / np.sqrt(2*np.pi*x/b)
    # x_max is the value of x that maximizes the function
    x_max = -b / (2*np.real(sp.lambertw(-1/2/L)))
    N = 1/f0(x_max,b,L)
    vOut = np.zeros_like(x)
    #vOut[x!=0] = a * N * f0(x[x!=0],b,L)
    vOut[x>0] = a * N * f0(x[x>0],b,L)
    return vOut
    """
    x0 = -b/(2*np.real(sp.lambertw(-1/2/L)))
    vOut = np.zeros_like(x)
    vOut[x>0] = a*((b*L*np.e)**((x[x>0]-x0)/b)) * ((x0/(x[x>0]**(x[x>0]/x0)))**(x0/b)) * np.sqrt(x0/x[x>0])
    return vOut

def mcpoiss(x, a, b, L):
    """
    Multiple (and constrained) continuous Poisson functions.  See the documentation for 'cpoiss',
    in this same module.  Here, input 'a' is a list (or other iterable) specifying the heights 
    of the various cpoiss components.  Input 'b' will be the same for each component, and input
    'L' will be multiplied by the component's iterated number.  So for example:
        mcpoiss(x,[4,5],1,3) = cpoiss(x,4,1,3) + cpoiss(x,5,1,6)
    
    2014.05.22 -- A. Manalaysay
    """
    try:
        aLen = len(a)
    except:
        raise TypeError("Input a must be iterable, otherwise use 'cpoiss' in this module.")
    vOut = np.zeros_like(x)
    for k, aa in enumerate(a):
        vOut += cpoiss(x,aa,b,(k+1)*L)
    return vOut

def AveBox(inMat_raw, n, r=1):
    """
    Perform a rolling box average filter on a set of data contained in input a, with a box width 
    specified by input 'bWidth'.  A can be a single signal, or an array of signals, but not 
    large than dimension 2.  The box filter is applied along the axis of the array specified by 
    kwarg axis.  For the moment, 'bWidth' must be an odd number.
    
    2014.05.07 -- A. Manalaysay
    2014.05.15 -- A. Manalaysay - Fixing the problem of adding the wrong number of samples on 
                                  either end of the trace.  This is step one, and only 
                                  implementing for 2-d arrays, smoothing along the vertical axis.
    """
    inMat = inMat_raw.copy()
    #if (axis != 0) and (axis != 1):
    #    raise ValueError("Input 'axis' must be either 0 or 1")
    if not hasattr(inMat,'ndim'):
        raise ValueError("Input 'inMat' must have an 'ndim' method")
    if inMat.ndim > 2:
        raise ValueError("Input 'inMat' cannot have more than 2 dimensions.")
    
    
    MatStart = inMat.copy()
    ptAddTp = np.floor(n/2)
    ptAddBt = np.floor((n-1)/2)
    
    cutNan = np.isnan(MatStart)
    MatStart[cutNan] = 0.
    cutNan = np.vstack([np.tile(cutNan[0,:],[ptAddTp,1]),cutNan,np.tile(cutNan[-1,:],[ptAddBt,1])])
    NotNanCMSM = (~cutNan).cumsum(0)
    NumDiv = (NotNanCMSM[(n-1):,:] - np.vstack([np.zeros([1,NotNanCMSM.shape[1]]),NotNanCMSM[:-n,:]]))
    for k in range(r):
        #MatStart = double([ repmat(MatStart(1,:),ptAddTp,1) ; MatStart ; repmat(MatStart(end,:),ptAddBt,1)]);
        MatStart = np.vstack([np.tile(MatStart[0,:],[ptAddTp,1]), MatStart, np.tile(MatStart[-1,:],[ptAddBt,1])])
        MatStart = MatStart.cumsum(0)
        #MatStart = (MatStart[(n-1):,:] - np.vstack([np.zeros([1,MatStart.shape[1]]),MatStart[:-n,:]]))/n
        MatStart = (MatStart[(n-1):,:] - np.vstack([np.zeros([1,MatStart.shape[1]]),MatStart[:-n,:]]))/NumDiv
    return MatStart

def randnCov(covmat, n=1, U=None, eVals=None):
    """
    Generate a set of random numbers that are drawn from a multidimensional Gaussian with 
    a specified covariance matrix.  The code works by first calculating a unitary operator 
    that transforms from the input basis into the eigenbasis of the covariance matrix.  In 
    this basis, the covariance matrix is diagonal, all random numbers in the set are 
    uncorrelated, and the eigenvalues give the variances of these random numbers.  Random
    numbers are generated in this basis and then transformed back to the original basis,
    thus introducing the desired correlations described by the covariance matrix.
    
    Inputs:
        covmat: Covariance matrix that expresses all variable-variable correlations.  Must be
                a numpy array (not list, not a numpy matrix), must be square.  If numpy chokes
                when trying to calculate the eigenvalues and eigenvectors, so will this 
                function :-)
             n: (optional) Number of sets of these random numbers to generate.
             U: (optional) The unitary operator that diagonalizes the covariance matrix. In cases
                where this function is called repeated times using the same covariance matrix,
                you might want to calculate U outside the function so that the same repeated 
                eigenval/vect calculation doesn't have to be repeated over and over again.
         eVals: (optional) Eigenvalues of covmat.  Optional for the same reason U is.
                U AND EVALS MUST BOTH BE SPECIFIED OR NEITHER.
    Outputs:
      randsOut: Array of correlated random numbers.  If n=1, the output is a regular numpy 1-D 
                array, of length given by the dimension of the covariance matrix.  If n>1, this
                output is a 2-D array, with n rows, and number of columns given by the dimension 
                of the covmat.  In other words, each row is a correlated set of random numbers.  
                Each row is an independent set of these random numbers.
    
    2016.03.23 -- A. Manalaysay
    """
    # ------- <Input checking> ------- #
    assert type(covmat)==np.ndarray, "Covariance matrix must be a numpy array."
    assert len(covmat.shape)==2, "Input 'covmat' must be a 2-D matrix."
    assert covmat.shape[0]==covmat.shape[1], "Input 'covmat' must be a square matrix."
    # ------- </Input checking> ------- #
    
    # Number of correlated random numbers (i.e. the dimension of the cov. matrix)
    numRands = covmat.shape[0]
    
    # The user can provide their own unitary transformation to save on computation time
    # in the case of repeated calls with the same covmat. Otherwise, it is calculated here.
    if U is None: 
        # Find the eigenvalues and eigenvectors of the covariance matrix
        eVals, eVects = np.linalg.eig(covmat)
        
        # Calculate the unitary operator that transforms from the input basis, into the
        # covariance matrix's eigenbasis.  Since only its transpose is used later on, I just
        # calculate the transpose-transformation directly.
        #U = eVects.T
        U_transpose = eVects
    
    assert 'eVals' in locals(), "You must also specify the eigenvectors."
    
    if n==1:
        # Generate random numbers in the diagonal basis (no correlations, obviously)
        randsDiag = np.random.randn(numRands)*np.sqrt(eVals)
        # Rotate the random numbers into the original basis.
        randsOut  = np.dot(U_transpose, randsDiag)
    else:
        #randsDiag = np.random.randn(n,numRands)*np.tile(np.sqrt(eVals),[n,1])
        #randsOut  = (np.dot(U_transpose, randsDiag.T)).T
        randsDiag = np.random.randn(numRands,n)*np.tile(np.c_[np.sqrt(eVals)], [1,n])
        randsOut  = (np.dot(U_transpose, randsDiag)).T
    return randsOut

def genBasis(n):
    """
    Generate a random, orthnormal set of basis vectors in n-dimensional space.  The method starts
    by generating an nxn matrix of unit-variance Gaussian random numbers, and treats each column
    as the components of a vector.  The first column is taken to be the first basis vector, and
    then the rest are generated via the Gram-Schmidt method.  The mutually orthagonal vectors are
    then normalized.
    
    Input:
                 n: The dimensionality of the space in which the vectors will be generated.  
                    Also, this is the number of vectors that will be generated (obviously).
    Output:
        basisVects: An nxn array, in which the columns are to be interpreted as the basis 
                    vectors.
    
    2016.03.25 -- A. Manalaysay
    """
    randVects = np.random.randn(n,n)
    basisVects = np.empty_like(randVects)
    
    # I repeat the process twice, in case any two vectors happened to be very nearly parallel to 
    # begin with, and they might not be completely orthogonal after the first iteration of Gram-
    # Schmidting due to floating-point precision.
    for k0 in range(2):
        # The for-loop handles the Gram-Schmidt stuff
        for k in range(n):
            if k==0:
                basisVects[:,k] = randVects[:,k]
            else:
                projCoeffs = np.dot(randVects[:,k],basisVects[:,:k])/(basisVects[:,:k]**2).sum(0)
                projections = (np.tile(projCoeffs,[n,1])*basisVects[:,:k]).sum(1)
                basisVects[:,k] = randVects[:,k] - projections
        
        # They are mutually orthogonal, but next they need to be normalized.
        basisVects = basisVects/np.tile(np.sqrt((basisVects**2).sum(0)),[n,1])
        randVects = basisVects.copy()
    return basisVects


