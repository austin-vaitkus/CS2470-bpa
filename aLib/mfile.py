"""
mfile module.
This module acts as a wrapper for various functions that interface pylab with matlab
Functions to include are:
    mfile.mwhos(...)  --> display contents of a mat file
    mfile.mload(...)  --> load a mat file, but with good options set automatically
    mfile.msave(...)  --> save to a mat file
    
    2014.02.28 -- A. Manalaysay
"""

from scipy import io
import aLib

def mwhos(*args, **kwargs):
    """
    mfile.mwhos: Displays contents of a matfile
    See the documentation for scipy.io.whosmat.  This is just a wrapper for that.
    
    Example:
    In [1]: mfile.mwhos('file.mat')
    
        Contents of file.mat:
        
             ('y', (1, 10), 'double')
             ('x', (1, 10), 'double')
    
    2014.02.28 -- A. Manalaysay
    """
    #from scipy import io
    print("\n    Contents of {:s}:\n".format(args[0]))
    for a in io.whosmat(*args, **kwargs):
        print('\t',a)
    #del io

def mload(fileName, **kwargs):
    """
    mfile.mload: Loads a mat file, into appropriate numpy data structures.
    See also the documentation for scipy.io.loadmat; this is a wrapper function for that.
    mfile.mload forces the kwarg 'squeeze_me' to be true, and converts the base dict 
    object returned by scipy.io.loadmat into my own spruced up dict object (class 'S').
    
    Example:
    In [1]: d = mfile.mload('testMatSCIPY.mat')
    In [2]: d
    Out[2]:
    Dict contents:
            FIELDS: TYPE                  CONTENTS
            ------  ----                  --------
       __globals__: [0 'list],             
        __header__: [1 'bytes],            
       __version__: [3 'str],             1.0
                 x: [(10,) float64array], [ 0.879018    0.62386238  0.67...
                 y: [(10,) float64array], [ 0.58120043 -0.8185307  -0.43...
    
    2014.02.28 -- A. Manalaysay
    """
    #from scipy import io
    #import myLib
    #d = io.loadmat(*args, **kwargs)
    if 'mdict' in kwargs.keys():
        io.loadmat(fileName, squeeze_me=True, **kwargs)
    else:
        d = io.loadmat(fileName, squeeze_me=True, **kwargs)
        return aLib.S(d)
    
#def msave(fileName, d, *args, **kwargs):
def msave(*args, **kwargs):
    """
    mfile.msave: Saves numpy data structures in a mat file.
    See also the documentation for scipy.io.savemat; this is a wrapper function for that.
    mfile.msave forces kwargs 'do_compression'=True and 'oned_as'="row".
    """
    #from sys import _getframe as gf
    #print("Function '{:s}' is a placeholder at the moment".format(gf().f_code.co_name))
    #io.savemat('140311_DD_ACv1.1_catted.mat', d, do_compression=True, oned_as='row')
    #io.savemat(fileName, d, *args, do_compression=True, oned_as='row', **kwargs)
    io.savemat(*args, do_compression=True, oned_as='row', **kwargs)
























