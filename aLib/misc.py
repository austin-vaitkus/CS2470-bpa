"""
A bunch of miscellaneous functions.
To see the functions available, type:
>>> misc.getFuns(misc)

2014.03.04 -- A. Manalaysay
"""

import matplotlib as mpl
import numpy as np

def stairs(x, y, *args, **kwargs):
    """
    For use like the Matlab stairs function.  However, numpy's histogram function returns a
    vector with one fewer elements than the bin edges vector, and thus stairs requires input
    'y' to have one fewer elements than input 'x'.  Note that numpy's histogram actually 
    returns a 2-element tuple, so it is good to de-reference the first element as in the 
    example below.
    -------------Inputs:
              x: (arg, required) Bin EDGES
              y: (arg, required) Bin contents (one fewer element than x).
             fs: (arg, optional) Format string. Ex: 'r-' for a red, solid line.
             y2: (kwarg, optional) For use when using hatching or filling.  This input will 
                 set the bottom boundary of the fill area, and should be an object similar 
                 to the input 'y'.
          hatch: (kwarg, optional) Hatch format string.  See documentation for 
                 matplotlib.pyplot.fill for the hatch string. For no hatching, set hatch 
                 equal to None, or leave blank.
      facecolor: (kwarg, optional) Color of the fill.  Choose 'none' (not None) for no fill,
                 or don't set.
     hatchcolor: (kwarg, optional) Color of the hatching. If left blank and "hatch" kwarg is 
                 set, hatchcolor is set to the color of the step curve.
    ------------Outputs:
              h: Either a single handle (in the case of no filling or hatching), or a list 
                 of two handles if filling and/or hatching is added.  In the latter case, 
                 the first handle is the step line, while the second handle is to the fill 
                 object that contains the filling and/or hatching.
    
    Example 1:
    In [1]: a = randn(1e3)+5
    In [2]: x_edges = linspace(0,10,101)
    In [3]: n_data = histogram(a, x_edges)[0]
    In [4]: figure()
    Out[5]: <matplotlib.figure.Figure at 0x1230a8810>
    In [6]: stairs(x_edges, n_data, 'b-', linewidth=2)
    
    Example 2 (from the last line of Example 1):
    In [6]: stairs(x_edges, n_data, 'b-', hatch='/////')
    
    2014.03.04 -- A. Manalaysay
    2014.03.28 -- A. Manalaysay - Added functionality for filling the histogram with either 
                                  solid color (possibly with non-unity alpha), and/or 
                                  hatching.  The previous version of this function used the 
                                  built-in matlab.pyplot.step function.  However, this tool 
                                  does not support filling/hatching, so a custom job is 
                                  utilized here.
    """
    if 'y2' in kwargs.keys():
        y2 = kwargs.pop('y2')
    else:
        y2 = -1
    def createStep(x,y):
        xx = np.tile(x,[2,1]).T.flatten()[1:-1]
        yy = np.tile(y,[2,1]).T.flatten()[:-2]
        return xx, yy
        
    #if any([opt in kwargs.keys() for opt in ("hatch","facecolor","hatchcolor")]):
    xx, yy = createStep(x,np.hstack([y,y[-1]]))
    
    flag_y2 = False
    if type(y2)==np.ndarray:
        flag_y2 = True
    
    if flag_y2:
        xx, y2 = createStep(x, np.hstack([y2,y2[-1]]))
    
    if "hatch" in kwargs.keys():
        hatch = kwargs.pop('hatch')
    else:
        hatch = None
    
    if "facecolor" in kwargs.keys():
        facecolor = kwargs.pop('facecolor')
    else:
        facecolor = 'none'
    
    if "hatchcolor" in kwargs.keys():
        hatchcolor = kwargs.pop('hatchcolor')
    else:
        hatchcolor = None
    
    h = []
    tH = mpl.pyplot.plot(xx, yy, *args, **kwargs)
    if type(tH) == list:
        tH = tH[0]
    h.append(tH)
    if (hatch is not None) or (facecolor != 'none') or (hatchcolor is not None):
        ymin, ymax = mpl.pyplot.ylim()
        if hatchcolor is None:
            hatchcolor = tH.get_color()
        tH = mpl.pyplot.fill_between(xx, 
                                     yy,
                                     y2=y2,
                                     hatch=hatch,
                                     edgecolor=hatchcolor,
                                     facecolor=facecolor,
                                     linewidth=0)
        if type(tH) == list:
            tH = tH[0]
        h.append(tH)
        mpl.pyplot.ylim(ymin,ymax)
    
    #else:
    #    h = mpl.pyplot.step(x,np.hstack([y,y[-1]]),*args,where='post',**kwargs)
    
    if len(h)==1:
        h = h[0]
    return h

def lstairs(x, y, *args, **kwargs):
    """
    Like stairs, but it defaults to a logarithmic vertical scale, while allowing continuity of 
    the step line when a bin has zero entries.  See the help documentation for misc.stairs.
    May give funky behavior when input y is not counts.
    
    2014.03.12 -- A. Manalaysay
    2014.03.28 -- A. Manalaysay - Along with 'stairs', added support for filling the histogram 
                                  with a solid/transparent color and/or a hatching pattern.
    """
    
    if 'y2' in kwargs.keys():
        y2 = kwargs.pop('y2')
    else:
        y2 = 5e-11
    
    n = np.float64(y.copy())
    n[n<=0] = 1e-50
    
    flag_y2 = False
    if type(y2)==np.ndarray:
        flag_y2 = True
    
    if flag_y2:
        n2 = np.float64(y2.copy())
        n2[n2<=0] = 1e-50
    
    def createStep(x,y):
        xx = np.tile(x,[2,1]).T.flatten()[1:-1]
        yy = np.tile(y,[2,1]).T.flatten()[:-2]
        return xx, yy
    
    #if any([opt in kwargs.keys() for opt in ("hatch","facecolor","hatchcolor")]):
    xx, nn = createStep(x,np.hstack([n,n[-1]]))
    
    if flag_y2:
        xx, y2 = createStep(x,np.hstack([n2,n2[-1]]))
    
    if "hatch" in kwargs.keys():
        hatch = kwargs.pop('hatch')
    else:
        hatch = None
    
    if "facecolor" in kwargs.keys():
        facecolor = kwargs.pop('facecolor')
    else:
        facecolor = 'none'
    
    if "hatchcolor" in kwargs.keys():
        hatchcolor = kwargs.pop('hatchcolor')
    else:
        hatchcolor = None
    
    h = []
    tH = mpl.pyplot.plot(xx, nn, *args, **kwargs)
    if type(tH) == list:
        tH = tH[0]
    h.append(tH)
    if (hatch is not None) or (facecolor != 'none') or (hatchcolor is not None):
        ymin, ymax = mpl.pyplot.ylim()
        if hatchcolor is None:
            hatchcolor = tH.get_color()
        tH = mpl.pyplot.fill_between(xx,
                                     nn,
                                     y2,
                                     hatch=hatch,
                                     edgecolor=hatchcolor,
                                     facecolor=facecolor,
                                     linewidth=0)
        if type(tH) == list:
            tH = tH[0]
        h.append(tH)
        mpl.pyplot.ylim(ymin,ymax)
    
    #else:
    #    h = mpl.pyplot.step(x,np.hstack([y,y[-1]]),*args,where='post',**kwargs)
    
    #mpl.pyplot.step(x,np.hstack([n,1e-10]),*args,where='post',**kwargs)
    yMax = mpl.pyplot.gca().get_ylim()[-1]
    yMax = 10**np.ceil(np.log10(yMax))
    mpl.pyplot.gca().set_yscale('log')
    mpl.pyplot.ylim([0.8, yMax])
    
    if len(h)==1:
        h = h[0]
    return h

def gcfn():
    """
    Returns the number of the current figure (the one into which shit will get plotted if you 
    just blindly executed a "plot" command).  Works similar to the Matlab 'gcf' function; the 
    problem with the matplotlib version of gcf is that it returns the figure handle, which is 
    not the figure number as it is in Matlab.  Here 'gcfn' stands for 'get current figure 
    number'.
    
    No Inputs.
    
    2014.03.12 -- A. Manalaysay
    2014.12.11 -- A. Manalaysay - Simplified the code; it no longer needs to go through and 
                  update all open figures just to return the number of the current figure.
    """
    return mpl.pyplot.gcf().number

def fup():
    """
    Try to update the current figure.  No inputs.
    
    2014.12.11 -- A. Manalaysay
    """
    mpl.pyplot.figure(gcfn())

def plot2d(x, y, x_range=None, y_range=None, x_bins=50, y_bins=50, flag_log=False, flag_mask=True, cmap=None):
    """
    Plot a 2d histogram in color from some data.  This produces something similar to 'hist2d', 
    but with some important differences.  First, hist2d produces a pixel image, while plot2d 
    produces a vector image.  Second, hist2d (because it is an image) draws a rectangle for all
    bins, while plot2d can plot only non-zero bins (or not, this is selectable).  But most 
    importantly (as the first point was made), this plot will be *scalable*!
    
             Inputs:
             x: 1-D ndarray of x values.
             y: 1-D ndarray of y values
       x_range: (optional) The range in x to make the histogram. Default behavior is to make the
                first edge at the lowest x-value, and the last edge at the highest x-value.
       y_range: (optional) Same as x_range, but for the y-values (obviously)
        x_bins: (optional) Number of bins in the x-directions.  Default is 50 bins.
        y_bins: (optional) Number of bins in the y-directions.  Default is 50 bins.
      flag_log: (optional) Boolean (True or False) specifying whether the color scale should be 
                linear (False) or logarithmic (True).  Default is False (linear scale).
     flag_mask: (optional) Boolean specifying whether or not to throw out bins that have no 
                content; True=throw out the empty bins, False=keep them. Default is True.
          cmap: (optional) The colormap object to use for the color scale and colorbar.  Default
                is the standard blue-green-red "jet" colormap.  Other colormaps are constructed
                in matplotlib.cm.
            Outputs:
      p_handle: Handle to the QuadMesh collection containing all the colorful boxes.
    
    Example:
    In [1]: x = randn(1e4)
    In [2]: y = randn(1e4)
    In [3]: colMap = matplotlib.cm.hot
    In [4]: plot2d(x, y, [-5,5], [-5,5], 70, 70, cmap=colMap)
    In [5]: colorbar()
    In [6]: minorticks_on()
    In [7]: cbarmticks()
    
    2014.03.19 -- A. Manalaysay
    2014.04.14 -- A. Manalaysay - The histogram is now displayed with pcolormesh, rather than the
                                  hard-coded PatchCollection.
    2014.04.16 -- A. Manalaysay - Added the capability to toggle whether or not to mask out the 
                                  empty bins.
    """
    
    #**** <INPUT HANDLING> ****#
    if cmap is None:
        cmap = mpl.cm.jet
    if (type(cmap) != type(mpl.cm.jet)) and (type(cmap) != type(mpl.cm.plasma)):
        raise TypeError("Input cmap must be a matplotlib color map object.")
    if x_range is None:
        x_range = [x.min(), x.max()]
    if y_range is None:
        y_range = [y.min(), y.max()]
    if (x_range[1]-x_range[0])<0:
        x_range = x_range[::-1]
    if (y_range[1]-y_range[0])<0:
        y_range = y_range[::-1]
    #**** </INPUT HANDLING> ****#
    
    # Make the bin edges
    x_xe = np.linspace(x_range[0], x_range[1], x_bins+1) # add 1 because these are bin edges
    x_ye = np.linspace(y_range[0], y_range[1], y_bins+1) # add 1 because these are bin edges
    x_width = np.abs(x_xe[1]-x_xe[0])
    y_width = np.abs(x_ye[1]-x_ye[0])
    
    # Calculate the 2-D histogram and flatten it
    n = np.histogramdd(np.array([x,y]).T, [x_xe, x_ye])[0]
    
    if flag_mask:
        n_max = n[n>0].max()
        n_min = n[n>0].min()
        n = np.ma.masked_where(n<=0, n)
    else:
        if flag_log:
            n[n<=0] = 0.1
        n_max = n.max()
        n_min = n.min()
    
    # Create a normalization function based on whether the user wants a lin or log color scale.
    if flag_log:
        n_Norm = mpl.colors.LogNorm(vmin=n_min, vmax=n_max)
    else:
        n_Norm = mpl.colors.Normalize(vmin=n_min, vmax=n_max)
    
    h = mpl.pyplot.pcolormesh(x_xe, x_ye, n.T, norm=n_Norm, cmap=cmap)
    h.set_edgecolor('face')
    h.set_linewidth(0.1)
    if flag_log and (not flag_mask):
        cMax = h.get_clim()[1]
        h.set_clim((.8,cMax))
    fup()
    return h

def plot2d_old(x, y, x_range=None, y_range=None, x_bins=50, y_bins=50, flag_log=False, 
    cmap=None, alpha=1., boxedgewidth=0.1):
    """
    Use plot2d instead; this is the old version and kept here only for legacy purposes, and as a
    useful reference for how to work with Collections classes.
    
    Update(2014.07.19) Ok ok, this is still useful if you want transparent bins. Woohoo.
    """
    
    #**** <INPUT HANDLING> ****#
    if cmap is None:
        cmap = mpl.cm.jet
    if (type(cmap) != type(mpl.cm.jet)) and (type(cmap) != type(mpl.cm.plasma)):
        raise TypeError("Input cmap must be a matplotlib color map object.")
    if x_range is None:
        x_range = [x.min(), x.max()]
    if y_range is None:
        y_range = [y.min(), y.max()]
    if (x_range[1]-x_range[0])<0:
        x_range = x_range[::-1]
    if (y_range[1]-y_range[0])<0:
        y_range = y_range[::-1]
    #**** </INPUT HANDLING> ****#
    
    # Make the bin edges
    x_xe = np.linspace(x_range[0], x_range[1], x_bins+1) # add 1 because these are bin edges
    x_ye = np.linspace(y_range[0], y_range[1], y_bins+1) # add 1 because these are bin edges
    x_width = np.abs(x_xe[1]-x_xe[0])
    y_width = np.abs(x_ye[1]-x_ye[0])
    
    # Calculate the 2-D histogram and flatten it
    n = np.histogramdd(np.array([x,y]).T, [x_xe, x_ye])[0]
    n_fl = n.flatten()
    
    # Create the list of patches, where each patch is a bin, and cut out the bins with zero entries.
    patches = [mpl.patches.Rectangle((xval,yval),x_width,y_width,fill=True) for xval in x_xe[:-1] \
        for yval in x_ye[:-1]]
    patches = [patches[k] for k in range(len(patches)) if n_fl[k]!=0]
    n_fl = n_fl[n_fl!=0]
    n_max = n_fl.max()
    n_min = n_fl.min()
    
    # Create a normalization function based on whether the user wants a lin or log color scale.
    if flag_log:
        n_Norm = mpl.colors.LogNorm(vmin=n_min, vmax=n_max)
    else:
        n_Norm = mpl.colors.Normalize(vmin=n_min, vmax=n_max)
    
    # Create a PatchCollection from the list of patches and set the histogram as its data
    p = mpl.collections.PatchCollection(patches, cmap=cmap, norm=n_Norm, alpha=alpha,
                                        edgecolor="face",linewidth=boxedgewidth)
    p.set_array(n_fl)
    
    # Put the PatchCollection into the current axis, set the axis limits.
    mpl.pyplot.gca().add_collection(p)
    mpl.pyplot.axis([x_range[0],x_range[1],y_range[0],y_range[1]])
    
    fup()
    return p

def getchildren(ax=None):
    """
    Get the handles to the objects in the current axis, but only ones that have been added
    by the user (not including stuff like axis labels, lines, etc.).
    
        Inputs:
           ax: (Optional) The axes in which you would like to search.  If no axes object is
               provided, it will take the one that is returned by matplotlib.pyplot.gca().
       Outputs:
           hh: A list of handles to the objects in the axes.
    
    2014.03.18 -- A. Manalaysay
    """
    if ax == None:
        ax = mpl.pyplot.gca()
    
    hh = [h for h in ax.get_children() if h._remove_method is not None]
    
    return hh

def deletelastchild():
    """
    misc.deletelastchild: (no arguments)
    Delete the last child in the current figure axis, which has the highest zorder value.
    No arguments, no output.
    
    2014.03.11 -- A. Manalaysay
    2014.03.18 -- A. Manalaysay - Update: Using the function "getchildren" in this module
    2014.04.30 -- A. Manalaysay - Matplotlib does not hold a strict correspondence between the 
                                  order of objects in ax.get_children() and the order in which 
                                  they were added to the current axis. Nor is there a 
                                  correspondence with the layering of objects in the plot.  This
                                  can be a problem when using fill objects.  I have now modified 
                                  it so that it will delete the last object that is on TOP. This
                                  is handled by considering only objects whose zorder parameter 
                                  is equal to the maximum zorder of all objects in the axis.
    """
    hCh = getchildren()
    hh = [h for h in hCh if h.get_zorder()==max([h0.get_zorder() for h0 in hCh])]
    if len(hh)>0:
        hh[-1].remove()
    
    fup()

def mfindobj(obj, **kwargs):
    """
    misc.mfindobj:
    Search an object for sub-objects that match certain properties; usage/philosophy is 
    similar to Matlab's 'findobj' function. Properties to search for must be member 
    functions of the objects contained in the input "obj", and these member functions must 
    take zero arguments (i.e. get_color, or get_markersize, see below).
    
    Example:
    In [1]: misc.mfindobj(gca(), get_color='b', get_markersize=6)
    Out[1]: <matplotlib.lines.Line2D at 0x109b73d90>
    
    2014.03.12 -- A. Manalaysay
    """
    if hasattr(obj, 'findobj'):
        found_list = []
        hh = obj.findobj(include_self=False)
        for h in hh:
            tFound = True
            for prop in kwargs.keys():
                tFound = tFound & hasattr(h, prop)
                if tFound:
                    tFound = tFound & (eval("h.{:s}()".format(prop))==kwargs[prop])
            if tFound:
                found_list.append(h)
        if len(found_list)==0:
            return "No matching objects"
        elif len(found_list)==1:
            return found_list[0]
        else:
            return found_list
    else:
        print("Object provided has no 'findobj' method")

def eff(cut):
    """
    eff: Returns the efficiency of a 1-D boolean array.
    If input "cut" is not of type numpy.ndarray, this function returns nothing.
    If cut.dtype is not "bool", it returns the fraction of non-zero elements.
    
    2014.03.12 -- A. Manalaysay
    """
    if type(cut)==np.ndarray:
        if cut.dtype == bool:
            effic = np.float(cut.sum())/len(cut)
        else:
            effic = (cut!=0).sum()/len(cut)
    else:
        effic = None
    return effic

def xsh(x):
    """
    xsh takes an ordered array and shifts the elements by half of their spacing.  This is 
    useful, for example, when you want to turn bin edges into bin centers.  The returned 
    numpy array has one fewer elements than the input array.  Input array can be non-
    uniformly spaced, but must be monotonically increasing.
    
    2014.03.13 -- A. Manalaysay
    """
    return x[:-1]+.5*np.diff(x)

def inrange(a, *args):
    """
    Returns a boolean ndarray the same length/size of input data 'a', indicating where a is 
    in range of values specified.
    
        Inputs:
          a: ndarray of data from which to analyze.
       args: Either one or two inputs. If one input is given, it should be a python list 
             object specifying [a_min, a_max].  If two arguments are given, the first should
             be a_min, the second should be a_max.  If a_min is greater than a_max, it blindly
             switches them.
       Outputs:
        cut: Boolean ndarray the same length/size as input a.  Logically, this represents 
             the following statement:
                (a >= a_min) & (a < a_max)
    
    2014.03.18 -- A. Manalaysay
    """
    # -----<INPUT HANDLING>----- #
    #if type(a) != np.ndarray:
    #    raise TypeError("Input data must be a numpy array (type numpy.ndarray).")
    if len(args) == 1:
        if len(args[0]) != 2:
            raise TypeError("If only one extra argument given, it must be a 2-element list.")
        a_min = args[0][0]
        a_max = args[0][1]
    else:
        a_min = args[0]
        a_max = args[1]
    if a_min > a_max:
        a_min, a_max = a_max, a_min
    # -----</INPUT HANDLING>----- #
    
    cut = (a >= a_min) & (a < a_max)
    return cut

def leg(*args, **kwargs):
    """
    A wrapper for the matplotlib.pyplot.legend function.
    If no arguments are given, it searches the plot for plotted elements and creates a legend 
    with all elements, labeled by their index in the list of elements.
    
    To select and/or reorder certain elements for selection in a legend, N+1 arguments must be
    provided, where N is the number of desired entries in the legend.  The first argument must 
    be a list of ints indicating which elements to add (and in which order), which is taken from
    the legend produced by leg().  The remaining N arguments are strings that indicate the 
    associated labels. A handle to the legend object is returned.
    
    Example:
    In [1]: figure()
    In [2]: plot(randn(10,5),'-')
    In [3]: leg() #produce a legend with all elements and their indices
    In [4]: leg([4,2],'Data','Monte Carlo') # pick only the 5th and 3rd elements (in that order) 
                                            # and label.
    
    2014.03.20 -- A. Manalaysay
    """
    #------<INPUT HANDLING>------#
    if len(args)>0:
        if type(args[0]) != list:
            raise TypeError("First provided argument must be a list of ints.")
        if [type(aa)==int for aa in args[0]].count(True) != len(args[0]):
            raise TypeError("First provided argument must be a list of ints.")
        if len(args[0]) != len(args[1:]):
            raise TypeError("The length of the first provided argument must be the same as\n" + \
                            "\t   the number of remaining arguments")
        if [type(aa)==str for aa in args[1:]].count(True) != len(args[0]):
            raise TypeError("All arguments after the first one must be strings.")
    #------</INPUT HANDLING>------#
    
    hh = getchildren() # function above from this module
    # construct handlermap dictionary
    hMap = {}
    for h in hh:
        hMap[h] = mpl.legend_handler.HandlerLine2D(numpoints=1)
    if len(args)==0:
        h_L = mpl.pyplot.legend(hh,["{:d}".format(k) for k in range(len(hh))],
              handler_map=hMap,**kwargs)
    else:
        h_L = mpl.pyplot.legend([hh[k] for k in args[0]],args[1:],
              handler_map=hMap,**kwargs)
    return h_L

def legloc(legPos, ax=None, legHandle=None):
    """
    Set the location of the legend.  A relative location can be given as a string, or actual 
    coordinates can be given.
    
    -----------Inputs:
         legPos: Position input.  This input can be a string, indicating the relative location, 
                 in which case it must be one of the following strings:
                    'best'
                    'upper right'
                    'upper left'
                    'lower left'
                    'lower right'
                    'right'
                    'center left'
                    'center right'
                    'lower center'
                    'upper center'
                    'center'
                 Alternatively, a length-2 tuple or list can be given that specifies the x-y 
                 coordinates of the lower-left corner of the legend.  The coordinates are in the
                 range [0,1], and relate to the position in the figure (not the axes).
             ax: (Optional) The handle to the axes in which to adjust the legend. Default: gca()
      legHandle: (Optional) The handle to the legend to be adjusted.  Default: the legend found 
                 in the current (or specified) axes.
    ----------Outputs:
      legHandle: The handle to the legend that was adjusted.
      
    Example:
    In [1]: figure()
    In [2]: plot(randn(10,5),'-')
    In [3]: leg() # "leg" from this module
    In [4]: leg([4,2],'Data','Monte Carlo') 
    In [5]: legloc('lower left')
    
    2014.03.28 -- A. Manalaysay
    """
    locDict = { 
        'best':0,
        'upper right':1,
        'upper left':2,
        'lower left':3,
        'lower right':4,
        'right':5,
        'center left':6,
        'center right':7,
        'lower center':8,
        'upper center':9,
        'center':10}
    if type(legPos) == str:
        if legPos not in locDict.keys():
            raise ValueError("Your location string must match one listed in the documentation.")
        legPosNum = locDict[legPos]
    elif (type(legPos)==tuple) or (type(legPos)==list):
        if len(legPos)!=2:
            raise ValueError("Your location list/tuple must be 2 elements.")
        legPosNum = legPos
    else:
        raise ValueError("I don't recognize your legend location input.")
    
    if ax is None:
        ax = mpl.pyplot.gca()
    if legHandle is None:
        legHandle = ax.get_legend()
    
    legHandle._set_loc(legPosNum)
    fup()
    return legHandle

import types
def getFuns(modName):
    """
    Return a list of the names of functions contained in a module.
    --------Inputs:
      modName: The module that is to be searched.
    -------Outputs:
      funList: A list of names of the functions contained in modName.
      
    2014.03.24 -- A. Manalaysay
    """
    funList = [modName.__dict__.get(a).__name__ for a in dir(modName) \
        if isinstance(modName.__dict__.get(a), types.FunctionType)]
    return funList

def getObj(obj):
    """
    For an object of a class whose __repr__ method has been customized, and the class name and 
    memory location are desired, this function can get them.  The returned string is identical 
    to what would be output if the class (and its super classes) don't have a __repr__ method 
    specified.  This is useful to see if a frequently modified object is getting overwritten, or 
    re-instantiated, for example.
    
    Example:
    In [1]: x = 3
    In [2]: x
    Out[2]: 3
    In [3]: getObj(x)
    Out[3]: '<builtins.int object at 0x1076642b0>'
    
    2014.03.29 -- A. Manalaysay
    """
    objStr = "<{:s}.{:s} object at {:s}>".format(obj.__class__.__module__,
                                                 obj.__class__.__name__,
                                                 hex(id(obj)))
    return objStr

def cbarmticks(cbar=None, ax=None, side='both'):
    """
    Turn on minorticks in a colorbar, and puts ticks on both the left and right.  Works for both
    linear and logarithmic color scales.  If the clim is reset after applying minor ticks, and 
    the color scale is logarithmic, this function will need to be run again.
    ---------------------- Inputs: -----------------------:
        cbar: (Optional) kwarg specifying the handle of the colorbar to modify.  Default: the 
              first colorbar found in ax.
          ax: (Optional) kwarg specifying the handle of the axes in which to make these changes.
              Default: the axes handle returned by gca()
        side: (Optional) The side of the colorbar where tick marks should be located.  
              Matplotlib creates them on the right only, by default, but the default for this 
              function is to put them on both side.  Possible values of this string kwarg are
              'left','right','both','default','none'. Default: 'both'.
    ----------------------Outputs: -----------------------:
        cbar: The handle to the colorbar that was modified.
    
    2014.04.10 -- A. Manalaysay
    2014.05.25 -- A. Manalaysay -- Added the optional default feature to add tick marks to both 
                                   sides of the colorbar (not just the right side as is done by 
                                   default by matplotlib).
    """
    if ax is None:
        ax = mpl.pyplot.gca()
    if cbar is None:
        objWcbarNoNone = [hh for hh in getchildren(ax) if hasattr(hh,'colorbar') and (hh.colorbar is not None)]
        if len(objWcbarNoNone) != 0:
            cbar = objWcbarNoNone[0].colorbar
    if cbar is None:
        raise ValueError('No colorbar found')
    
    if type(cbar.norm) == mpl.colors.LogNorm:
        vmin = cbar.vmin
        vmax = cbar.vmax
        locs = cbar.ax.yaxis.get_major_locator().locs
        locsExpts = locs * (np.log10(vmax) - np.log10(vmin)) + np.log10(vmin)
        locsExpts = np.hstack([locsExpts[0]-1,locsExpts,locsExpts[-1]+1])
        locsN = 10**locsExpts
        
        minLocsN = np.tile(locsN,[9,1]).T.cumsum(1)[:,1:].flatten()
        minLocsN = minLocsN[(minLocsN>=vmin)&(minLocsN<=vmax)]
        minlocs = (np.log10(minLocsN)-np.log10(vmin))/(np.log10(vmax) - np.log10(vmin))
        cbar.ax.yaxis.set_minor_locator(mpl.ticker.FixedLocator(minlocs))
    else:
        cbar.ax.minorticks_on()
    
    # This line makes the colorbar look nice when viewed in OS X Preview, for Apple users.
    cbar.solids.set_edgecolor('face')
    
    # Set the side where tick marks will be located. Changes them to 'both' by default, but is 
    #controlled by kwarg 'side'.
    cbar.ax.yaxis.set_ticks_position(side)
    
    fup()
    return cbar

def cbarylabel(string, ax=None, cbar=None, **kwargs):
    """
    Set the ylabel of the colorbar, if one exists.  The default behavior of matplotlib is to
    set the rotation of the ylabel such that it reads bottom to top.  Since I don't like 
    this, this function hardwires it to be the other way (which requires also a change to 
    the 'verticalalignment' kwarg.  Additional kwargs can be added that specify the 
    fontsize, etc.
    
    2014.05.09 -- A. Manalaysay
    """
    if ax == None:
        ax = mpl.pyplot.gca()
    if cbar == None:
        objWcbarNoNone = [hh for hh in getchildren(ax) if hasattr(hh,'colorbar') and (hh.colorbar!=None)]
        if len(objWcbarNoNone) != 0:
            cbar = objWcbarNoNone[0].colorbar
    if cbar == None:
        raise ValueError('No colorbar found')
    
    cbar.ax.set_ylabel(string, rotation=270., verticalalignment='bottom', **kwargs)
    
    # This line makes the colorbar look nice when viewed in OS X Preview, for Apple users.
    cbar.solids.set_edgecolor('face')
    
    fup()
    return cbar

def ticksdbl(ax=None, which='x'):
    """
    Set the spacing of the major ticks in an axes to be half what they currently are.
    
    Inputs:
           ax: (Optional) Handle to the axes which are to be modified.  If left blank, the axes 
               returned by gca() are used.
        which: (Optional) A one-character string specifying either 'x' or 'y'.  Default is 'x'.
    Outputs: (none)
    
    2014.05.03 -- A. Manalaysay
    """
    if ax==None:
        ax = mpl.pyplot.gca()
    
    if which=='x':
        axHandle = ax.xaxis
    elif which=='y':
        axHandle = ax.yaxis
    else:
        raise ValueError("Input 'which' must be either 'x' or 'y'.")
    
    tickSpacing = np.diff(axHandle.get_ticklocs()[:2])[0]
    axHandle.set_major_locator(mpl.ticker.MultipleLocator(.5*tickSpacing))
    fup()

def ticksauto(ax=None, which='b'):
    """
    Set the spacing of the major ticks in an axes to be auto.
    
    Inputs:
           ax: (Optional) Handle to the axes which are to be modified.  If left blank, the axes 
               returned by gca() are used.
        which: A one-character string specifying either 'x', 'y', or 'b' (for both).  Default is 
               'b'.
    Outputs: (none)
    
    2014.05.05 -- A. Manalaysay
    """
    if ax==None:
        ax = mpl.pyplot.gca()
    
    if which=='x':
        axHandle = [ax.xaxis]
    elif which=='y':
        axHandle = [ax.yaxis]
    elif which=='b':
        axHandle = [ax.xaxis, ax.yaxis]
    else:
        raise ValueError("Input 'which' must be either 'x' or 'y'.")
    for hax in axHandle:
        hax.set_major_locator(mpl.ticker.AutoLocator())
    fup()

def tickxm(mult, ax=None):
    """
    Set the multiple to be used as the major tick mark on the x-axis.
    
    Inputs:
         mult: Within the range of the horizontal axis, a tick mark will be placed at every 
               multiple of the number given as 'mult'.  E.g. if mult=2 and the range is [-5,7], 
               then tick marks will be placed at -4, -2, 0, 2, 4, 6.  Mult need not be an 
               integer.
           ax: (Optional) Handle to the axes which are to be modified.  If left blank, the axes 
               returned by gca() are used.
    Outputs: (none)
    
    2014.09.30 -- A. Manalaysay
    """
    if ax is None:
        ax = mpl.pyplot.gca()
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(mult))
    figNumber = ax.get_figure().number
    mpl.pyplot.figure(figNumber)

def tickym(mult, ax=None):
    """
    Set the multiple to be used as the major tick mark on the y-axis.
    
    Inputs:
         mult: Within the range of the vertical axis, a tick mark will be placed at every 
               multiple of the number given as 'mult'.  E.g. if mult=2 and the range is [-5,7], 
               then tick marks will be placed at -4, -2, 0, 2, 4, 6.  Mult need not be an 
               integer.
           ax: (Optional) Handle to the axes which are to be modified.  If left blank, the axes 
               returned by gca() are used.
    Outputs: (none)
    
    2014.09.30 -- A. Manalaysay
    """
    if ax is None:
        ax = mpl.pyplot.gca()
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(mult))
    figNumber = ax.get_figure().number
    mpl.pyplot.figure(figNumber)

def figlike(fnumSource, fnumTarget=None):
    """
    Create a figure with the same dimensions as another figure.
    Inputs:
        fnumSource: The number of the figure which to copy.
        fnumTarget: Optional. If none is supplied, it uses the default next-in-the-sequence like
                    the 'figure' function normally does.
    Outputs:
           fHandle: The handle to the created figure.
    
    2014.12.12 -- A. Manalaysay
    """
    hSource = mpl.pyplot.figure(fnumSource)
    fHeight = hSource.get_figheight()
    fWidth = hSource.get_figwidth()
    
    fHandle = mpl.pyplot.figure(fnumTarget,figsize=[fWidth,fHeight])
    return fHandle

from __main__ import __dict__ as mainGlobals
def getSameVars(obj, us=True):
    """
    Find other variable names in the user workspace that point to the same object in memory
    as the one given.  Returns a list of strings, each one a name of a variable that points to
    the address in memory where input 'obj' is located.
    Inputs:
          obj: Object to look for.
           us: (kw, Optional): Short for "underscore", this is a boolean kwarg; when true, the 
               output list includes items whose variable name starts with an underscore ('_').
               If false, these underscore-prefixed variables names are filtered out. 
               Default: True
    Outputs:
       vNames: List of strings representing the variable names that point to the same mem 
               address as in put obj.
    
    2014.12.28 -- A. Manalaysay
    """
    vNames = []
    glKeys = mainGlobals.keys()
    for n in glKeys:
        if id(mainGlobals[n])==id(obj):
            vNames.append(n)
    assert type(us)==bool, "Input 'us' must be a boolean."
    if not us:
        vNames = [v for v in vNames if v[0]!='_']
    return vNames

def points2eq(x1,y1,x2,y2):
    """
    Gives the slope and y-intercept of the line defined by the input points.
    Output: (slope, y-intercept)
    
    2016.05.27 -- A. Manalaysay
    """
    m = (y2-y1)/(x2-x1)
    b = y1 - m*x1 
    return m,b

