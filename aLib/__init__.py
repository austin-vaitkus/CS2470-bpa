"""
This is aLib.
aLib's development state is "unstable".  I will be changing things and not versioning them, but
I will try to give an indication below when a modification has occured.

2014.02.09 -- A. Manalaysay (created)
2014.03.20 (updated)
2014.03.22 (updated)
2014.03.27 (updated)
2014.03.28 (updated)
2014.04.10 (updated)
2014.07.07 (updated)
"""

from __future__ import division
from __future__ import print_function

import sys
pyVersion = sys.version_info.major
del sys

class S(dict):
    """
    S is a custom subclass of the built-in Python dict class. Its purpose is basically to allow
    usability of a Python dict object in a way that is similar to Matlab struct objects. 
    Objects of type S will have keys specified by strings (as can be true for normal dict 
    objects), but can also be accessed by using the ".<name>" syntax (or in python lingo, 
    you can access keys as attributes). 
    
    Example:
    In [1]: from aLib import *
    In [2]: d = S()
    In [3]: d
    Out[3]: No fields defined yet
    In [4]: d.x = rand(1e3)
    In [5]: d.y = randn(1e3)
    In [6]: d.t_index = r_[1:6]
    In [7]: d
    Out[7]: 
    Dict contents:
        FIELDS: TYPE                    CONTENTS
        ------  ----                    --------
       t_index: [(5,) int64array],      [1 2 3 4 5]
             x: [(1000,) float64array], [ 0.7228869   0.962533    0.38...
             y: [(1000,) float64array], [-0.13009171  0.42487732  0.79...
    
    In [8]: d.t_index                  # access of key as an attribute
    Out[8]: array([1, 2, 3, 4, 5])
    In [9]: d['t_index']               # access of key in the normal dict way
    Out[9]: array([1, 2, 3, 4, 5])
    In [10]: whos
    Variable   Type    Data/Info
    ----------------------------
    d          S       Dictionary with 3 fields
    
    Example:
    In [1]: d_stand = {'x':3, 'y':27} # (a standard python dict)
    In [2]: d_stand
    Out[2]: {'x':3, 'y':27}
    In [3]: d_nice = S(d_stand)
    In [4]: d_nice
    Out[4]:
    Dictionary contents:
       FIELDS: TYPE      CONTENTS
       ------  ----      --------
            x: [1 'int], 3
            y: [1 'int], 27
    
    2014.2.27 -- A. Manalaysay
    """
    def __init__(self, *args, **kwargs):
        super(S, self).__init__(*args, **kwargs)
        self.__dict__ = self
    
    def __str__(self):
        return "Dictionary with {:d} fields".format(len(self))
    
    def __repr__(self):
        maxNameLen = 6
        maxShapeLen = 0
        hasFound = False
        for keyName in sorted(self):
            hasFound = True
            if len(keyName) > maxNameLen:
                maxNameLen = len(keyName)
        maxNameLen = min(maxNameLen,35)
        formString = '{0:>' + '{0:d}'.format(maxNameLen+3) + '}'
        attributeStrings = []
        maxAttrStringLen = 0
        headStrings = []
        headStringLength = 30
        k = 0
        for fieldName in sorted(self):
            fieldType = type(self[fieldName])
            # numpy array
            if fieldType == type(np.array([])): 
                shapeString = str(self[fieldName].shape)
                fieldType = self[fieldName].dtype
                fieldTypeString = str(fieldType) + "array"
                headStrings.append(str(self[fieldName].flatten()[:20])[:headStringLength])
                if len(headStrings[k][:headStringLength])==headStringLength:
                    headStrings[k] += "..."
            
            # numpy matrix
            elif fieldType == type(np.mat([])): 
                shapeString = str(self[fieldName].shape)
                fieldType = self[fieldName].dtype
                fieldTypeString = str(fieldType) + "matrix"
                headStrings.append(str(self[fieldName].flatten()[:20]))
                if len(headStrings[k][:headStringLength])==headStringLength:
                    headStrings[k] += "..."
            
            # python list or tuple
            elif (fieldType == type([]))or(fieldType == type(())): 
                shapeString = "{:d}".format(len(self[fieldName]))
                fieldTypeString = str(fieldType)
                fieldTypeString = fieldTypeString[7:-2]
                headStrings.append(" ")
            
            # python string
            elif fieldType == type(''): 
                shapeString = "{:d}".format(len(self[fieldName]))
                fieldTypeString = str(fieldType)
                fieldTypeString = fieldTypeString[7:-2]
                headStrings.append(self[fieldName][:headStringLength])
                if len(headStrings[k][:headStringLength])==headStringLength:
                    headStrings[k] += "..."
            
            # single values (int, float, long)
            #elif (fieldType==type(int(0)))or(fieldType==type(long(0)))or(fieldType==type(float(0))):
            elif (fieldType==type(int(0)))or(fieldType==type(float(0))): 
                shapeString = "1"
                fieldTypeString = str(fieldType)
                fieldTypeString = fieldTypeString[7:-2]
                headStrings.append(str(self[fieldName]))
            
            # not an array
            else: # not an array
                shapeString = "1"
                fieldTypeString = str(fieldType)
                fieldTypeString = fieldTypeString[7:-2]
                headStrings.append(" ")
                
            attributeStrings.append(': [' + shapeString + ' ' + fieldTypeString + '],')
            if len(attributeStrings[k])>maxAttrStringLen:
                maxAttrStringLen = len(attributeStrings[k])
            k += 1
        if hasFound:
            maxAttrStringLen = min(maxAttrStringLen, 35)
            formAttrString = '{0:<' + '{0:d}'.format(maxAttrStringLen+1) + '}'
            summaryString = "Dictionary contents:\n"
            summaryString += formString.format("FIELDS") + formAttrString.format(": TYPE") + \
                "CONTENTS\n"
            summaryString += formString.format("------") + formAttrString.format("  ----") + \
                "--------\n"
            for k, fieldName in enumerate(sorted(self)):
                summaryString += formString.format(fieldName[0:maxNameLen]) + \
                    formAttrString.format(attributeStrings[k][0:maxAttrStringLen]) + \
                    headStrings[k] + \
                    '\n'
        else:
            summaryString = "No fields defined yet."
        return summaryString



import numpy as np
import matplotlib as mpl
if pyVersion == 3:
    from aLib import mfile
    from aLib import dp
    from aLib import fits
    from aLib.misc import *
    from aLib import astats
    from aLib import rates
    from aLib import pyNEST as pn
    from aLib import mathops as mo
else:
    #from __future__ import division
    import mfile
    import dp
    import fits
    from misc import *
    import astats
    import rates
    import pyNEST as pn
    import mathops as mo

# PLEASE NOTE: The above import loads all functions of aLib.misc into the current work space.
#              This is simply the way I like it, but if you don't like that behavior, please 
#              just let me know and we can figure out a better solution... maybe a conditional
#              behavior based on the username or something would be cleanest way to satisfy
#              everyone.  -Aaron
