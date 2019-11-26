"""
Fitting utilities.  This module provides classes for doing various types of fits.  Fits can be 
done on user-defined fit functions, or via builtin functions of the class, which include:

gauss1          : a*exp(-(x-mu)**2 / 2 / sig**2)
gauss2          : a1*exp(-(x-mu1)**2 / 2 / sig1**2) + a2*exp(-(x-mu2)**2 / 2 / sig2**2)
gauss1_offset   : a*exp(-(x-mu)**2 / 2 / sig**2) + b
exp1            : a*exp(x*b)
exp2            : a1*exp(x*b1) + a2*exp(x*b2)
exp1_offset     : a*exp(x*b) + c

The technique for using this module is to create an instance of one of the classes.  This object 
will accept all the information needed for the fit, and will then store (and return) the results
of the fit, including parameter uncertainties.

Classes:
    cfit: Perform fits using chi^2 minimization.
    lfit: Perform fits using MLE.

See the docstrings for the fit classes for more specific information and usage.

2014.04.02 -- A. Manalaysay
"""

from scipy import optimize as op
import numpy as np
import types
import inspect

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
                                 exp1: a*exp(x*b)
                                 exp2: a1*exp(x*b1) + a2*exp(x*b2)
                          exp1_offset: a*exp(x*b) + c
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
    
    Example:
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
    
    2014.04.02 -- A. Manalaysay
    """
    # Static dictionary of built-in fit functions; more can be added, obviously
    _builtinFuns = {
        'gauss1': lambda x, a, mu, sig: a * np.exp(-(((x-mu)/sig)**2)/2), 
        'gauss2': lambda x, a1, mu1, sig1, a2, mu2, sig2: \
                  a1*np.exp(-(((x-mu1)/sig1)**2)/2) + a2*np.exp(-(((x-mu2)/sig2)**2)/2), 
        'gauss1_offset': lambda x, a, mu, sig, b: a * np.exp(-(((x-mu)/sig)**2)/2) + b,
        'exp1': lambda x, a, b: a * np.exp(x*b), 
        'exp2': lambda x, a1, b1, a2, b2: a1*np.exp(x*b1) + a2*np.exp(x*b2), 
        'exp1_offset': lambda x, a, b, c: a * np.exp(x*b) + c} 
    
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
        if self.fitFun != None:
            self.set_fitFun(self.fitFun)
        if self.yerr != None:
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
        if (type(self.fitFun)==str) or (self.fitFun == None):
            reprString += "Function: " + self.fitFun + "\n"
        elif isinstance(self.fitFun, types.FunctionType):
            if self.fitFun.__name__ == "<lambda>":
                reprString += "Function: " + inspect.getsource(self.fitFun) + "\n"
            else:
                reprString += "Function: " + self.fitFun.__name__ + "\n"
        else:
            reprString += "Function: invalid \n"
        reprString += "fitParams: {:s}\n".format(self.fitParams)
        reprString += "fitErrors: {:s}".format(self.fitErrors)
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
            if (self.__dict__[dKey] == None) and (dKey not in self._NonRequiredArgs):
                raise ValueError("{:s} is not yet set!".format(dKey))
        if type(self.fitFun) == str:
            if self.fitFun in self._builtinFuns.keys():
                self.fitFun = self._builtinFuns[self.fitFun]
            else:
                raise ValueError("'{:s}' is not a builtin function.".format(self.fitFun))
        fitResults = op.curve_fit(self.fitFun, self.xdata, self.ydata, p0=self.startVals, sigma=self.yerr)
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

from scipy import stats

class lfit():
    """
    Class for doing likelihood fits.
    """
    # Static dictionary of built-in fit functions; more can be added, obviously
    """
    _builtinFuns = {
        'gauss1': lambda x, a, mu, sig: a * np.exp(-(((x-mu)/sig)**2)/2), 
        'gauss2': lambda x, a1, mu1, sig1, a2, mu2, sig2: \
                  a1*np.exp(-(((x-mu1)/sig1)**2)/2) + a2*np.exp(-(((x-mu2)/sig2)**2)/2), 
        'gauss1_offset': lambda x, a, mu, sig, b: a * np.exp(-(((x-mu)/sig)**2)/2) + b,
        'exp1': lambda x, a, b: a * np.exp(x*b), 
        'exp2': lambda x, a1, b1, a2, b2: a1*np.exp(x*b1) + a2*np.exp(x*b2), 
        'exp1_offset': lambda x, a, b, c: a * np.exp(x*b) + c} 
    """
    #_builtinCDFs = {
    #    'gauss1': stats.norm.cdf,
    #    'gauss1_offset': lambda x, a, mu, sig, b:
    #    'exp1': stats.expon.cdf
    #    'exp1_offset': lambda x, a, b, c:} 
    
    _NonRequiredArgs = ('yerr', 'startVals', 'fitParams','fitErrors','fitCov')
    
    def __init__(self, **kwargs):
        self.__dict__ = {
            'xdata': None,
            'ydata': None,
            'yerr': None,
            'fitFun': None,
            'CDFfun': None,
            'startVals': None,
            'fitParams': None,
            'fitErrors': None,
            'fitCov': None}
        for dKey in self.__dict__.keys():
            if dKey in kwargs.keys():
                self.__dict__[dKey] = kwargs[dKey]
    


# for generating starting points, see:
# /Applications/MATLAB_R2011b.app/toolbox/curvefit/curvefit/@fittype/private/sethandles.m
# for gauss2, maybe do the correlation - derivative - product - sign - diff thing
