"""
Fitting utilities.  This module provides classes for doing various types of fits.  Fits can be 
done on user-defined fit functions, or via builtin functions of the class, which include:

gauss1          : Single Gaussian function
gauss2          : Sum of two Gaussian functions
gauss1_offset   : Single Gaussian function plus an overall offset.
exp1            : Single exponential function
exp2            : Sum of two exponential functions
exp1_offset     : Single exponential function plus an overall offset

The technique for using this module is to create an instance of one of the classes.  This object 
that is created will accept all the information needed for the fit, and will then store (and 
return) the results of the fit, including parameter uncertainties.

Classes:
    cfit: Fits using chi^2 minimization.
    lfit: Fits using binned Maximum Likelihood Estimation (MLE).

See the docstrings for the fit classes for more specific information and examples.  Note that the
normalization constant for the built-in fit functions has a different meaning between cfit and 
lfit.

2014.04.02 -- A. Manalaysay
2014.12.25 -- A. Manalaysay -- I had never finished the MLE fitting class, which is now complete
                               as a first draft.
"""

from scipy import optimize as op
from scipy import stats as st
import numpy as np
import types
import inspect
from scipy.special import erf
import scipy.integrate as it

# The Hessian matrix is useful when calculating the error covariance matrix from the likelihood function
def calcHessian(x0, func, *args):
    """
    A numerical approximation to the Hessian matrix of a function at
    location x0 (hopefully, the minimum)
    
    See: https://stat.ethz.ch/pipermail/r-help/2004-February/046272.html
    for info on getting errors from the Hessian matrix.
    
    This function is necessary in the case of MLE to calculate the covariance matrix
    of the parameter estimates.  If the function given is -log(Likelihood), then the
    covariance matrix is -Hessian(-log(Likelihood))^-1 , where "^-1" indicates matrix
    inverse, not 1/
    
    Algorithm adapted from https://gist.github.com/jgomezdans/3144636
    
    Inputs: ------------------------
             x0: The array of values at which to calculate the Hessian matrix.
           func: The function that will have its second derivatives taken.  This 
                 function must take x0 as its first input parameter, and *args 
                 (below) as additional params.  No kwargs allowed.
          *args: Additional parameters (2 through whatever) for the function 'func'.
    Outputs:------------------------
     theHessian: The matrix of second partial derivatives of the function 'func',
                 evaluated at x0.
    
    2014.12.23 -- A. Manalaysay
    """
    # calculate an approximation to the first derivative
    f1 = op.approx_fprime(x0, func, (1e-5)*x0, *args)
    
    # Allocate space for the hessian
    n = x0.shape[0]
    theHessian = np.zeros((n,n))
    
    # The next loop fill in the matrix
    xx = x0.copy()
    
    for j in range(n):
        xx0 = xx[j] # Store old value
        xx[j] = xx0 + (1e-5)*xx0 # Perturb with finite difference
        # Recalculate the partial derivatives for this new point
        f2 = op.approx_fprime(xx, func,(1e-5)*xx, *args)
        theHessian[:, j] = (f2 - f1)/(1e-5)/xx0 # scale...
        xx[j] = xx0 # Restore initial value of x0
    return theHessian 

class cfit():
    """
    Class for performing fits via chi^2 minimization; this is essentially a wrapper for the 
    function "curve_fit" of the module "scipy.optimize".
    
    Member variables:
             xdata: ndarray containing the x-data for the fit
             ydata: ndarray containing the y-data for the fit
              yerr: (Optional) ndarray of 1-sigma errors on the ydata
            fitFun: The function which will be optimized against the data.  This can be a user-
                    defined function, in which case it must accept N inputs, where the first 
                    input is the independent variable (x), and the remaining (N-1) inputs are 
                    the fit parameters.  Alternatively, one can choose any of the pre-defined 
                    functions, including:
                               gauss1: a*exp(-(x-mu)**2 / 2 / sig**2)
                               gauss2: a1*exp(-(x-mu1)**2 / 2 / sig1**2) + a2*exp(-(x-mu2)**2 / 2 / sig2**2)
                        gauss1_offset: a*exp(-(x-mu)**2 / 2 / sig**2) + b
                                 exp1: a*exp(-x/mu)
                                 exp2: a1*exp(-x/mu1) + a2*exp(-x/mu2)
                          exp1_offset: a*exp(-x/mu) + b
                    If a custom fit function is desired, a function handle must be given.  If 
                    one of these template functions is desired, the name of the template is 
                    given as a string.
         startVals: (Optional) A list (or tuple) of starting values for the fit parameters.
         fitParams: ndarray containing the best-fit values of the fit parameters.
         fitErrors: ndarray containing the 1-sigma errors on the fit parameters.
            fitCov: ndarray containing the covariance matrix of the fit parameters.
    
    Member functions:
              __init__(**kwargs): Initialization method for the class object. kwargs can contain 
                                  any of the member variables (except for fitParams, fitErrors, 
                                  or fitCov, obviously).
                set_xdata(xdata): Set the xdata
                set_ydata(ydata): Set the ydata
                  set_yerr(yerr): Set the yerrors. Elements of 'yerr' that are zero are set to 
                                  unity.
              set_fitFun(fitFun): Set the fit function
        set_startVals(startVals): Set the starting parameter values.
                           fit(): Perform the fit, print the results, and set member variables 
                                  'fitParams', 'fitErrors', and 'fitCov'.
                    chi2(Params): Return the chi^2 value of the best-fit params (if no ipnput is 
                                  given, or of the given input params.
    
    Example: (using a library fit function)
    (assuming : from numpy import *
                from matplotlib.pyplot import plot
                from aLib.misc import *
    are already performed)
    
    In [1]: from aLib import fits
    In [2]: data = 5.4*randn(1e3) + 55
    In [3]: x_edges = linspace(0,100)
    In [4]: n_bins = histogram(data,x_edges)[0]
    In [5]: stairs(x_edges,n_bins,'-')
    Out[5]: <matplotlib.lines.Line2D at 0x1119012d0>
    In [6]: g = fits.cfit(xdata=xsh(x_edges),ydata=n_bins,yerr=sqrt(n_bins),fitFun='gauss1')
    In [7]: g.fit()
      fitParams: [ 149.39556285   55.06156714   -5.39130857]
      fitErrors: [ 3.12501383  0.09152225  0.0687756 ]
    In [8]: x_fine = linspace(0,100,300)
    In [9]: plot(x_fine,g.fitFun(x_fine,*g.fitParams),'r-')
    Out[9]: [<matplotlib.lines.Line2D at 0x111d87590>]
    In [10]: g.fitParams
    Out[10]: array([ 149.39556285   55.06156714   -5.39130857])
    In [11]: g.fitErrors
    Out[10]: array([ 3.12501383  0.09152225  0.0687756 ])
    
    -------------------------------------------------------------
    Example: (using a custom fit function)
    
    In [1]: x = linspace(0,8*pi,50)
    In [2]: y = sin(x)+.25*randn(x.size)  # sinusoid with gaussian noise of sig=0.25
    In [3]: def sinFun(x,A,w):
      ....:     return A*sin(w*x)
      ....: 
    
    In [4]: g = fits.cfit(xdata=x, ydata=y, yerr=0.25*ones_like(x), fitFun=sinFun)
    In [5]: g.set_startVals([1.1, 1.01])
    In [6]: g.fit()
      fitParams: [ 0.96318572  0.99856167]
      fitErrors: [ 0.05049935  0.00351179]
    In [7]: g.fitCov
    Out[7]:
    array([[  2.55018386e-03,   5.62591133e-06],
           [  5.62591133e-06,   1.23326589e-05]])
    In [8]: g.fitParams
    Out[8]: array([ 0.96318572,  0.99856167])
    In [9]: g.fitErrors
    Out[9]: array([ 0.05049935,  0.00351179])
    
    2014.04.02 -- A. Manalaysay
    """
    # Static dictionary of built-in fit functions; more can be added, obviously
    _builtinFuns = {
        'gauss1': lambda x, a, mu, sig: a * np.exp(-(((x-mu)/sig)**2)/2), 
        'gauss2': lambda x, a1, mu1, sig1, a2, mu2, sig2: \
                  a1*np.exp(-(((x-mu1)/sig1)**2)/2) + a2*np.exp(-(((x-mu2)/sig2)**2)/2), 
        'gauss1_offset': lambda x, a, mu, sig, b: a * np.exp(-(((x-mu)/sig)**2)/2) + b,
        'exp1': lambda x, a, mu: a * np.exp(-x/mu), 
        'exp2': lambda x, a1, mu1, a2, mu2: a1*np.exp(-x/mu1) + a2*np.exp(-x/mu2), 
        'exp1_offset': lambda x, a, mu, b: a * np.exp(-xmu) + b} 
    
    _NonRequiredArgs = ('yerr', 'startVals', 'fitParams','fitErrors','fitCov')
    
    def __init__(self, **kwargs):
        self.__dict__ = {
            'xdata': None,
            'ydata': None,
            'yerr': None,
            'fitFun': None,
            'startVals': None,
            'fitParams': None,
            'fitErrors': None,
            'fitCov': None}
        for dKey in self.__dict__.keys():
            if dKey in kwargs.keys():
                self.__dict__[dKey] = kwargs[dKey]
        if self.fitFun is not None:
            self.set_fitFun(self.fitFun)
        if self.yerr is not None:
            self.set_yerr(self.yerr)
        """
        if self.xdata != None:
            self.set_xdata(self.xdata)
        if self.ydata != None:
            self.set_ydata(self.ydata)
        if self.startVals != None:
            self.set_startVals(self.startVals)
        """
    
    def __repr__(self):
        reprString = "cfit object:\n"
        if (type(self.fitFun)==str) or (self.fitFun is None):
            reprString += "Function: " + self.fitFun + "\n"
        elif isinstance(self.fitFun, types.FunctionType):
            if self.fitFun.__name__ == "<lambda>":
                reprString += "Function: " + inspect.getsource(self.fitFun) + "\n"
            else:
                reprString += "Function: " + self.fitFun.__name__ + "\n"
        else:
            reprString += "Function: invalid \n"
        reprString += "fitParams: {:s}\n".format(self.fitParams.__str__())
        reprString += "fitErrors: {:s}".format(self.fitErrors.__str__())
        return reprString
    
    def set_xdata(self, xdata):
        self.xdata = xdata
    
    def set_ydata(self, ydata):
        self.ydata = ydata
    
    def set_yerr(self, yerr):
        yerr = np.abs(yerr)
        if any(yerr==0):
            yerr[yerr==0] = 1
        self.yerr = yerr
    
    def set_fitFun(self, fitFun):
        if (fitFun == 'gauss1') and (self.startVals == None):
            yMax = self.ydata.max()
            yMean = (self.ydata*self.xdata).sum()/self.ydata.sum()
            yVar = (self.ydata*(self.xdata**2)).sum()/self.ydata.sum() - yMean**2
            self.startVals = (yMax, yMean, np.sqrt(yVar))
        if fitFun in ('gauss2','gauss1_offset','exp1','exp2','exp1_offset'):
            print("Warning: Starting values might be necessary for '{:s}'".format(fitFun))
        if fitFun in self._builtinFuns.keys():
            self.fitFun = self._builtinFuns[fitFun]
        else:
            self.fitFun = fitFun
    
    def set_startVals(self, startVals):
        self.startVals = startVals
    
    def fit(self, flag_Verbose=True):
        """
        Perform the fit based on the information that has been given to the instance of the
        class.  The fit results will be printed as output, and stored in member variables:
            fitParams: Best-fit values of the fit parameters
            fitErrors: 1-sigma uncertainties on the fit values
               fitCov: Covariance matrix of the fit values
        A boolean kw input can be given, 'flag_Verbose', that controls whether this function
        prints the results.  Default: True (it prints).
        2014.04.02 -- A. Manalaysay
        """
        for dKey in self.__dict__.keys():
            if (self.__dict__[dKey] is None) and (dKey not in self._NonRequiredArgs):
                raise ValueError("{:s} is not yet set!".format(dKey))
        if type(self.fitFun) == str:
            if self.fitFun in self._builtinFuns.keys():
                self.fitFun = self._builtinFuns[self.fitFun]
            else:
                raise ValueError("'{:s}' is not a builtin function.".format(self.fitFun))
        fitResults = op.curve_fit(self.fitFun, self.xdata, self.ydata, p0=self.startVals, 
                     sigma=self.yerr, absolute_sigma=True)
        self.fitParams = fitResults[0]
        self.fitErrors = np.sqrt(fitResults[1].diagonal())
        self.fitCov    = fitResults[1]
        if flag_Verbose:
            for fRes in ('fitParams','fitErrors'):
                print("{:>11s}: {:s}".format(fRes, self.__dict__[fRes].__str__()))
    
    def chi2(self, *args):
        """
        Return the chi^2 value.  If no inputs are given, this function returns the chi^2 
        value when evaluated at the best-fit parameter values.  If an argument is given, 
        it must be a list/tuple/ndarray of parameter values at which to evaluate chi^2.
        
        2014.04.06 -- A. Manalaysay
        """
        if len(args)==0:
            if self.fitParams == None:
                self.fit()
            fParams = self.fitParams
        elif len(args[0])==len(self.startVals):
            fParams = args[0]
        else:
            raise ValueError("Input must have the proper number of elements.")
        if self.yerr == None:
            chi2Val = ((self.ydata-self.fitFun(self.xdata,*fParams))**2).sum()
        else:
            chi2Val = (((self.ydata-self.fitFun(self.xdata,*fParams))/self.yerr)**2).sum()
        return chi2Val

class lfit():
    """
    Class for doing binned likelihood fits.  Specifically, this class is good for fitting a 
    distribution to a histogram containing integer-valued bin contents.  The given distribution,
    along with its normalization, is integrated over the width of each bin to produce an array 
    of bin-content expectation values.  These expectation values are treated as the mean of of a
    Poisson distribution, and the observed contents of each bin are compared to this 
    
    Member variables:
                    xdata: ndarray containing the BIN EDGES of the histogram that is being fit.
                    ydata: ndarray containing the bin contents of the histogram. This array must
                           have one fewer element than 'xdata'.
     fitFunAntiderivative: The function handle that returns the antiderivative of the fit 
                           function. This function must accept N inputs: the first input must be
                           the independent variable (x value), and the remaining N-1 inputs are 
                           the model free parameters that are being estimated.
                   fitFun: The function that will be fit to the supplied histogram. There are 
                           two options: user can supply a custom function (input rules the same 
                           as fitFunAntiderivative), OR the user can supply a string that refers
                           to one of the predefined functions. The library of builtin functions
                           include:
                                       gauss1: (a/sig/sqrt(2*pi))*exp(-(x-mu)**2/2/sig**2)
                                       gauss2: (a1/sig1/sqrt(2*pi))*exp(-(x-mu1)**2/2/sig1**2) + 
                                               (a2/sig2/sqrt(2*pi))*exp(-(x-mu2)**2/2/sig2**2)
                                gauss1_offset: (a/sig/sqrt(2*pi))*exp(-(x-mu)**2/2/sig**2) + b
                                         exp1: (a/mu)*exp(-x/mu)
                                         exp2: (a1/mu1)*exp(-x/mu1) + (a2/mu2)*exp(-x/mu2)
                                  exp2_offset: (a/mu)*exp(-x/mu) + b
                           NOTE: These definitions are slightly different than those in class cfit!
                startVals: (optional) A list (or tuple) of starting values for the fit parameters.
                           STRONGLY RECOMMENDED!!! (except for 'gauss1' and 'exp1')
                fitParams: ndarray containing the best-fit values of the fit parameters.
                fitErrors: ndarray containing 1-sigma errors of the fit parameters.
                   fitCov: ndarray containing the covariance matrix of the fit parameters.
                fitResult: scipy.optimize.optimize.OptimizeResult object, which is the output of the
                           effort to minimize the negative-log-likelihood function.  Contains 
                           information about the fitting process (number of function evals, etc.), 
                           as well as information about whether the minimization terminated 
                           successfully or not.
    
    Member function:
      __init__(**kwargs): Initialization method for the class object.  kwargs can contain any of
                          the member variables (except for fitParams, fitErrors, or fitCov, 
                          obviously).
        set_xdata(xdata): Set the xdata.
        set_ydata(ydata): Set the ydata.
        set_fitFunAntiderivative(fitFunAntiderivative): 
                          Set the function that calculates the antiderivative of the fit 
                          function. See member variables (above) for input format requirements.
              set_fitFun(fitFun): 
                          Set the fit function. Can be either a string, denoting one of the pre-
                          defined fit functions (see above) or a function handle to a custom fit
                          function.
        set_startVals(startVals): 
                          Set the starting parameter values.
                   fit(): Perform the fit, print the results, and set member variables 
                          'fitParams', 'fitErrors', and 'fitCov'.
    
    Example: (coming soon, sorry!)
    
    2014.12.29 -- A. Manalaysay
    """
    # Static dictionary of built-in fit functions; more can be added, obviously.
    # THESE ARE THE ANTIDERIVATIVES OF FIT FUNCTIONS.
    _builtinFuns = {
        'gauss1': lambda x, a, mu, sig: (a/2)*erf((x-mu)/sig/np.sqrt(2)),
        'gauss2': lambda x, a1, mu1, sig1, a2, mu2, sig2: \
            (a1/2)*erf((x-mu1)/sig1/np.sqrt(2)) + (a2/2)*erf((x-mu2)/sig2/np.sqrt(2)),
        'gauss1_offset': lambda x, a, mu, sig, b: (a/2)*erf((x-mu)/sig/np.sqrt(2))+b*x,
        'exp1': lambda x, a, mu: a*(1.-np.exp(-x/mu)),
        'exp2': lambda x, a1, mu1, a2, mu2: a1*(1.-np.exp(-x/mu1))+a2*(1.-np.exp(-x/mu2)),
        'exp1_offset': lambda x, a, mu, b: a*(1.-np.exp(-x/mu)) + b*x} 
    
    _NonRequiredArgs = ('startVals', 'fitParams','fitErrors','fitCov','fitFun',
                        'fitFunAntiderivative','fitResult')
    
    def __init__(self, **kwargs):
        self.__dict__ = {
            'xdata': None,
            'ydata': None,
            'fitFun': None,
            'fitFunAntiderivative': None,
            'startVals': None,
            'fitParams': None,
            'fitErrors': None,
            'fitCov': None,
            'fitResult': None}
        for dKey in self.__dict__.keys():
            if dKey in kwargs.keys():
                self.__dict__[dKey] = kwargs[dKey]
        if self.fitFunAntiderivative is not None:
            if self.fitFun is not None:
                print("Warning: fitFunAntiderivative provided, ignoring input given for fitFun!")
        elif self.fitFun is not None:
            self.set_fitFun(self.fitFun)
    
    def __str__(self):
        return "lfit object"
    
    def __repr__(self):
        reprString = "lfit object:\n"
        if (self.fitFunAntiderivative is not None) and \
           isinstance(self.fitFunAntiderivative, types.FunctionType) and (type(self.fitFun)!=str):
            if self.fitFunAntiderivative.__name__ == "<lambda>":
                try:
                    reprString += "Function antiderivative: " + inspect.getsource(self.fitFunAntiderivative) + \
                                  "\n"
                except:
                    reprString += "Function antiderivative: <lambda>\n"
            else:
                reprString += "Function antiderivative: " + self.fitFunAntiderivative.__name__ + "\n"
        elif (type(self.fitFun)==str) or (self.fitFun is None):
            reprString += "Function: " + self.fitFun.__str__() + "\n"
        elif isinstance(self.fitFun, types.FunctionType):
            if self.fitFun.__name__ == "<lambda>":
                try:
                    reprString += "Function: " + inspect.getsource(self.fitFun) + "\n"
                except:
                    reprString += "Function: <lambda>\n"
            else:
                reprString += "Function: " + self.fitFun.__name__ + "\n"
        else:
            reprString += "Function: invalid \n"
        reprString += "fitParams: {:s}\n".format(self.fitParams.__str__())
        reprString += "fitErrors: {:s}".format(self.fitErrors.__str__())
        return reprString
    
    def set_xdata(self, xdata):
        self.xdata = xdata
    
    def set_ydata(self, ydata):
        self.ydata = ydata
    
    def set_fitFunAntiderivative(self, fitFunAntiderivative):
        self.fitFunAntiderivative = fitFunAntiderivative
        if self.startVals is None:
            # Determine the number of free parameters of the fit function, which is
            # the number of params minus 1 (first is the independent variable).
            nParams = len(inspect.getargspec(self.fitFunAntiderivative).args) - 1
            self.startVals = np.ones(nParams, dtype=np.int64)
    def set_fitFun(self, fitFun):
        # Set the start values for gauss1 or exp1, if they have not been provided
        if (fitFun == 'gauss1') and (self.startVals is None):
            xShifted = self.xdata[:-1]+.5*np.diff(self.xdata)
            ySum = self.ydata.sum()
            yMean = (self.ydata*xShifted).sum()/ySum
            yVar = (self.ydata*(xShifted**2)).sum()/ySum - yMean**2
            self.startVals = (ySum, yMean, np.sqrt(yVar))
        elif (fitFun == 'exp1') and (self.startVals is None):
            xShifted = self.xdata[:-1]+.5*np.diff(self.xdata)
            ySum = self.ydata.sum()
            yMean = (self.ydata*xShifted).sum()/ySum
            self.startVals = (ySum, yMean)
        # Warn the user that start values will be appreciated.
        if (fitFun in ('gauss2','gauss1_offset','exp2','exp1_offset')) and (self.startVals is None):
            print("Warning: Starting values might be necessary for '{:s}'".format(fitFun))
        # If a str for fitFun was provided, but it's not in the library, raise an error flag.
        if (type(fitFun)==str) and (fitFun not in self._builtinFuns.keys()):
            raise ValueError("The function called '{:s}' is not in the library of builtin functions.".
                             format(fitFun))
        # If a str was given for fitFun, 
        if (fitFun in self._builtinFuns.keys()) and (self.fitFunAntiderivative is None):
            self.set_fitFunAntiderivative(self._builtinFuns[fitFun])
        if self.fitFunAntiderivative is None:
            print("Warning: for custom fit distributions, it's faster if you supply the\n" + \
                  "antiderivative function.")
            def fitCDF(x,*args):
                if type(x) == np.ndarray:
                    outArray = np.array([it.quad(fitFun,0.,xx,args=args)[0] for xx in x])
                else:
                    outArray = it.quad(fitFun,0.,x,args=args)[0]
                return outArray
            self.set_fitFunAntiderivative(fitCDF)
    
    def set_startVals(self, startVals):
        self.startVals = startVals
    
    def fit(self, flag_Verbose=True):
        """
        Perform the fit based on the information that has been given to the instance of the
        class.  The fit results will be printed as output, and stored in member variables:
            fitParams: Best-fit values of the fit parameters
            fitErrors: 1-sigma uncertainties on the fit values
               fitCov: Covariance matrix of the fit values
            fitResult: Object that the minimizer spits out with optimization statuses.
        A boolean kw input can be given, 'flag_Verbose', that controls whether this function
        prints the results.  Default: True (it prints).
        
        2014.12.25 -- A. Manalaysay
        """
        for dKey in self.__dict__.keys():
            if (self.__dict__[dKey] is None) and (dKey not in self._NonRequiredArgs):
                raise ValueError("{:s} is not yet set!".format(dKey))
        if len(self.xdata) != (len(self.ydata)+1):
            raise ValueError("\nInput 'xdata' must be an array of BIN EDGES, while 'ydata' is\n" + \
                             "an array of bin contents.  This means than xdata must have exactly\n" + \
                             "one element more than ydata.")
        assert isinstance(self.fitFunAntiderivative, types.FunctionType), \
                          "fitFunAntiderivative not set (neither implicitly nor explicitly)."
        # Define the negative log-likelihood function
        def negLogLike(theta):
            nus = np.diff(self.fitFunAntiderivative(self.xdata,*theta))
            nus[nus==0] = 1e-100
            return -(self.ydata*np.log(nus)-nus).sum()
        
        # Minimize the negative-log-likelihood function
        minResult = op.minimize(negLogLike, self.startVals, method='Nelder-Mead')
        self.fitParams = minResult.x
        self.fitResult = minResult
        
        # Calculate the Hessian matrix of the negative-log-likelihood function
        HessMat = calcHessian(self.fitParams, negLogLike)
        
        # Get the parameter covariance matrix by inverting the Hessian matrix
        self.fitCov = np.linalg.inv(HessMat)
        
        # Get the fit errors from the square root of the diagonal of the covariance matrix
        self.fitErrors = np.sqrt(np.diag(self.fitCov))


# for generating starting points, see:
# /Applications/MATLAB_R2011b.app/toolbox/curvefit/curvefit/@fittype/private/sethandles.m
# for gauss2, maybe do the correlation - derivative - product - sign - diff thing


