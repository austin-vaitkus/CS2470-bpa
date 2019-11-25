"""
HDF5 Convenience functions (read-only, so far)

hdf provides tools for navigating an HDF file like it's a file system.  

Methods:
        ls: Print contents of a directory (or "group", in hdf5 language.
        cd: Change current directory.
       pwd: Print working directory.
       cat: Spit out a variable (no support for slicing yet).
2014.03.22 -- A. Manalaysay
              (This module seemed like a good idea before I started it, but
               now it doesn't feel like it adds much convenience beyond the 
               normal hdf5 usage...)
"""

import h5py
import numpy as np

def _checkCWD(fi):
    """
    Private method of module aLib.hdf.
    Checks for a well formed cwd key in h5py object 'fi'.  If fi.cwd is not well formed,
    it resets it to ['/']
    
    2014.03.22 -- A. Manalaysay
    """
    # Create cwd as the base directory, if it doesn't exist
    if not('cwd' in dir(fi)):
        fi.cwd = ['/']
    
    # Make sure cwd is a list; re-initialize if not
    if type(fi.cwd) != list:
        fi.cwd = ['/']
    
    # Make sure each element of cwd is a string; re-initialize if not
    if [type(aa)==str for aa in fi.cwd].count(True) != len(fi.cwd):
        fi.cwd = ['/']
    while (len(fi.cwd)>1) and (fi.cwd[0]=='/'):
        fi.cwd = fi.cwd[1:]
    if (fi.cwd[0]!='/'):
        fi.cwd = ['/'] + fi.cwd

def _formCWDstring(fi, propDir=''):
    """
    Private method of module aLib.hdf.
    
    2014.03.22 -- A. Manalaysay
    """
    _checkCWD(fi)
    if propDir == '':
        searchDir = "/".join(fi.cwd)
    elif propDir.split("/")[0] == "..":
        numBack = min(len(fi.cwd),propDir.split("/").count(".."))
        propDir = "/".join(propDir.split("/")[numBack:])
        #print("numBack = {:d}".format(numBack))
        
        searchDir = "/".join(fi.cwd[:-numBack]+propDir.split("/"))
    else:
        searchDir = "/".join(fi.cwd) + "/" + propDir
    list_of_all_contents = []
    fi.visit(list_of_all_contents.append)
    list_of_all_contents = ['/'] + list_of_all_contents
    if (searchDir[0] == '/') and (len(searchDir)>1):
        searchDir = searchDir[1:]
    while (len(searchDir)>1) and (searchDir[-1]=='/'):
        searchDir = searchDir[:-1]
    while (len(searchDir)>1) and (searchDir[0]=='/'):
        searchDir = searchDir[1:]
    if not(searchDir in list_of_all_contents) and searchDir[0]!='/':
        print("PATH NICHT GEFUND: '{:s}'".format(searchDir))
        searchDir = "**NOTFOUND**"
    #if (searchDir!="**NOTFOUND**") and (type(fi[searchDir])!=h5py._hl.group.Group):
    if (searchDir!="**NOTFOUND**") and (type(fi[searchDir])==h5py._hl.dataset.Dataset):
        searchDir = "**NOTGROUP**" + searchDir
    return searchDir

def ls(fi, theDir=''):
    """
    Navigate an HDF5 file as though it were a directory structure.  
    hdf.ls is intended to be like ls.
    
    2014.03.22 -- A. Manalaysay
    """
    
    _checkCWD(fi)
    srchDir = _formCWDstring(fi, theDir)
    
    if srchDir == "**NOTFOUND**":
        print("Path not found.")
    elif srchDir[:12] == "**NOTGROUP**":
        print("Not a group: {:s}".format(theDir))
    else:
        # Spit out the attributes
        print("Attributes: of '{:s}':".format(srchDir))
        for mm in list(fi[srchDir].attrs):
            print("    {:s}".format(mm))
        
        #print("srchDir: '{:s}'".format(srchDir))
        typeGroup = [type(fi[srchDir+'/'+a])==h5py._hl.group.Group \
            for a in list(fi[srchDir].keys())]
            #for a in list(fi["/".join(fi.cwd)].keys())]
        typeItem  = []
        for k, item in enumerate(typeGroup):
            if typeGroup[k]:
                typeItem.append('G')
            else:
                typeItem.append('D')
        
        # Spit out the contents
        print("Contents of '{:s}':".format(srchDir))
        for k, mm in enumerate(list(fi[srchDir].keys())):
            print("    ({:s}) {:s}".format(typeItem[k], mm))
    

def pwd(fi):
    """
    Navigate an HDF5 file as though it were a directory structure.
    hdf.cd is intended to be like .
    
    2014.03.22 -- A. Manalaysay
    """
    _checkCWD(fi)
    pwdString = "/".join(fi.cwd)
    if pwdString.count("/")==len(pwdString):
        pwdString = "/"
    else:
        while pwdString[0]=="/":
            pwdString = pwdString[1:]
        pwdString = '/' + pwdString
    #print("/".join(fi.cwd))
    print(pwdString)


def cd(fi,newDir = ''):
    """
    Navigate an HDF5 file as though it were a directory structure.  
    hdf.cd is intended to be like cd.
    
    2014.03.22 -- A. Manalaysay
    """
    _checkCWD(fi)
    
    newString = _formCWDstring(fi, newDir)
    
    if newString == "**NOTFOUND**":
        print("Path not found.")
    elif newString[:12] == "**NOTGROUP**":
        print("Not a group.")
    else:
        fi.cwd = newString.split("/")

def cat(fi, dName=''):
    """
    Spit out a variable name.
    
    2014.03.22 -- A. Manalaysay
    """
    _checkCWD(fi)
    
    newString = _formCWDstring(fi, dName)
    
    if newString == "**NOTFOUND**":
        print("Data not found")
    elif newString[:12] == "**NOTGROUP**":
        a = fi[newString[12:]][:]
        if type(a) == np.ndarray:
            return a.squeeze()
        else:
            return a
    else:
        print("'{:s}': Provided name is not a data set.".format(newString))








    