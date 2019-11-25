"""
This module contains functions for calculating rates from neutrinos.

2014.05.31 -- A. Manalaysay
"""
import numpy as np
from aLib.rates.WIMP import FQ_Helm
import os
from numpy.random import rand

_PathToThisFile = os.path.split(__file__)[0]

# Create a dictionary that holds values of the physical constants; units are given as comments
_physCons = { 
    'Gf'        : 1.1663787e-11,   # MeV^{-2}
    'c'         : 299792458.,      # m/s
    'hbar'      : 6.58211928e-22,  # MeV*s
    'eCh'       : 1.602176565e-19, # Coulombs; also this many Joules = 1 eV
    'sin2thW'   : 0.2386,          # low-E limit, from PDG
    'MeVperAMU' : 931.494061,      # MeV, from PDG
    }

def _getTargetParams(target):
    """
    Private function, used by several public functions in this module.  Takes a string as
    input that specifies the target, and returns as arrays (or scalars) the following details:
        Z_N    : Number of protons in the target nucleus
        N_N    : Number of neutrons in the target nucleus (this is an array, to account for 
                 all naturally occurring isotopes.
        A_N    : Number of total nucleons in all isotopes of the target nucleus.
        m_N    : Mass of each isotope of the target nucleus (in units of MeV/c^2)
        abund_N: Relative abundance of each isotope (using natural abundances by default).
    """
    #   ------------ <INPUT CHECKING> ------------   #
    if type(target) != str:
        raise TypeError("Input 'target' must be a string.")
    if ~np.any([target==tgt for tgt in ['Xe','Ge','Si','Ar']]):
        raise TypeError("Input 'target' must be a recognized element name; see help documentation.")
    #   ------------ </INPUT CHECKING> ------------   #
    if target == 'Xe':
        Z_N = 54.  # "_N" denotes properties of the target nucleus
        N_N = np.r_[70., 72., 74., 75., 76., 77., 78., 80., 82.]
        A_N = N_N + Z_N
        m_N = np.r_[123.9,125.9,127.9,128.9,129.9,130.9,131.9,133.9,135.9]*_physCons['MeVperAMU'] #masses in MeV/c^2
        abund_N = np.r_[.09, .09, 1.92, 26.44, 4.08, 21.18, 26.89, 10.44, 8.87]*0.01  # abundances
    elif target == 'Ge':
        Z_N = 32.
        N_N = np.r_[38., 40., 41., 42., 44.]
        A_N = N_N + Z_N;
        m_N = np.r_[69.9, 71.9, 72.9, 73.9, 75.9]*_physCons['MeVperAMU']
        abund_N = np.r_[20.84, 27.54, 7.73, 36.28, 7.61] * 0.01
    elif target == 'Si':
        Z_N = 14.
        N_N = np.r_[14., 15., 16.]
        A_N = N_N + Z_N
        m_N = np.r_[27.98, 28.98, 29.98]*_physCons['MeVperAMU']
        abund_N = np.r_[92.23, 4.68, 3.09] * 0.01
    elif target == 'Ar':
        Z_N = 18.
        N_N = np.r_[18., 20., 22.]
        A_N = N_N + Z_N
        m_N = np.r_[36.0, 38.0, 40.0]*_physCons['MeVperAMU']
        abund_N = np.r_[0.34, 0.06, 99.6] * 0.01
    else:
        raise ValueError("Target not recognized.")
    
    return Z_N, N_N, A_N, m_N, abund_N

def _setNuSource(nu_source):
    """
    Private function, used by several public functions in this module.  This takes
    as input either a string (which is a label of a library source, B8, hep, atm, or DSNB), or a
    length-3 list: [flux, en_centers, en_edges].  In return, this function gives variables 
    'neut_flux', 'en_centers', 'en_edges'
    """
    #   ------------ <INPUT CHECKING> ------------   #
    if type(nu_source) == list:
        flag_nuSrcCustom = True
        if len(nu_source) != 3:
            raise TypeError("Input 'nu_source' as a list must have a length of 3.")
        for k0, src in enumerate(nu_source):
            if (type(src) != np.ndarray) or (src.dtype != np.float64):
                raise TypeError("Input 'nu_source' must be a list of float64 ndarrays.")
            if k0 == 0:
                nu_source_length = len(src)
            elif k0==1:
                if len(src) != nu_source_length:
                    raise TypeError("Second element of input 'nu_source' must be the same length \
                                     as the first.")
            else:
                if len(src) != (nu_source_length+1):
                    raise TypeError("Third element of input 'nu_source' must have one more element \
                                     than the first two.")
    elif type(nu_source) == str:
        flag_nuSrcCustom = False
        if ~np.any([nu_source==src for src in ['B8','hep','DSNB','atm']]):
            raise TypeError("Input 'nu_source' as a character string must be a recognized template; " + \
                            "see help documentation.")
    #   ------------ </INPUT CHECKING> ------------   #
    
    if flag_nuSrcCustom:  # user specifies custom source
        neut_flux  = nu_source[0]
        en_centers = nu_source[1]
        en_edges   = nu_source[2]
    else:
        with np.load(os.path.join(_PathToThisFile,"CoNeutTemplateNuSrcs.npz")) as nuSrcs:
            if nu_source == 'B8':
                en_centers = nuSrcs['B8_en_centers']
                en_edges   = nuSrcs['B8_en_edges']
                neut_flux  = nuSrcs['B8_flux']
            elif nu_source == 'hep':
                en_centers = nuSrcs['hep_en_centers']
                en_edges   = nuSrcs['hep_en_edges']
                neut_flux  = nuSrcs['hep_flux']
            elif nu_source == 'DSNB':
                en_centers = nuSrcs['DSNB_en_centers']
                en_edges   = nuSrcs['DSNB_en_edges']
                neut_flux  = nuSrcs['DSNB_3MeV'] + nuSrcs['DSNB_5MeV'] + 4*nuSrcs['DSNB_8MeV']
            elif nu_source == 'atm':
                en_centers = nuSrcs['atm_en_centers']
                en_edges   = nuSrcs['atm_en_edges']
                neut_flux  = nuSrcs['atm_nue_flux'] + nuSrcs['atm_nueBar_flux'] + \
                             nuSrcs['atm_numu_flux'] + nuSrcs['atm_numuBar_flux']
    
    return neut_flux, en_centers, en_edges

def CoNeutRecoilSpect(Er, target='Xe', nu_source='B8'):
    """
    CoNeutRecoilSpect.m: Computes the differential nuclear recoil spectrum,
    given input vector of recoil energies, target medium, and neutrino
    source.  All input vectors horizontal, please.
    -----------------------------------------------------------------------------------
             inputs:
           Er: ndarray of nuclear recoil energies, in keVnr, in the specified target, at 
               which the corresponding differential spectrum will be evaluated.
       target: Identity of the target medium, given as the element abbreviation as a 
               string.  Considers the different naturally occurring isotopes and their 
               respective natural abundances separately, and then combines them.  
               Currently available: 'Xe','Ge','Si','Ar'.
    nu_source: The neutrino source.  Can be given as a string indicating solar B8 
               ('B8'), solar hep ('hep'), diffuse supernova neutrino background 
               ('DSNB'), or atmospheric ('atm').  Alternatively, a custom neutrino 
               source can be given.  In this case, a list must be given for this 
               input: [flux, en_centers, en_edges]. 'flux' is the ndarray of 
               differential flux (must be in neutrinos/cm^2/s/MeV), 'en_centers' is the 
               ndarray of energy bins (in MeV), 'en_edges' is the ndarray of bin edges 
               (in MeV).  'flux' and 'en_centers' must be the same length, while 
               'en_edges' must have one more element than the other two.
    -----------------------------------------------------------------------------------
            outputs:
     nr_spect: Differential spectrum of nuclear recoils, in evts/kg/day/keV, matching 
               the array of Er that was input.
    -----------------------------------------------------------------------------------
      Example: 
          In [1]: Er = linspace(0,2e2,2e3)
          In [2]: dRdEr_B8 = CoNeutRecoilSpect(Er,'Xe','B8')
    -----------------------------------------------------------------------------------
    
    2014.05.31 -- A. Manalaysay
    2016.01.20 -- A. Manalaysay - Fixed two bugs.  First, a variable-type issue caused 
                                  the output spectrum to have type "object", instead of 
                                  the same type as the input Er.  Second, a floating-
                                  point-operation error was giving a small negative 
                                  value to the last element in the integral over 
                                  neutrino energies; this led to strange behavior in the 
                                  high-energy portion of the recoil spectrum.
    2016.01.25 -- A. Manalaysay - Updated the neutrino fluxes.
    2016.02.18 -- A. Manalaysay - Code cleanup.  Moved nu_source parsing (and input 
                                  checking) to a separate function.  Same with input 
                                  'target'.  Moved physical constants to a dictionary 
                                  accessible by all functions in this module.
    """
    #   ---------- <SET NEUTRINO SOURCE> ----------   #
    neut_flux, en_centers, en_edges = _setNuSource(nu_source)    
    # convert flux from /cm^2/s/MeV to natural units
    neut_flux = neut_flux*(_physCons['hbar']**3)*(_physCons['c']**2)*(100.**2)
    #   ---------- </SET NEUTRINO SOURCE> ----------  #
    
    #   ------- <CONVERT ER TO NAT. UNITS> -------   #
    # Working in natural units with MeV
    Er = Er/(1e3)
    #   ------- </CONVERT ER TO NAT. UNITS> -------   #
    
    #   -------- <SET TARGET PROPERTIES> --------   #
    # Xe, Ge, Si, Ar
    Z_N, N_N, A_N, m_N, abund_N = _getTargetParams(target)
    Q_N = N_N - (1-4*_physCons['sin2thW'])*Z_N # nuclear weak charge
    #   -------- </SET TARGET PROPERTIES> --------   #
    
    #   ---------- <CALCULATE RATES> -----------   #
    d_Eneut = np.diff(en_edges)
    nr_spect = np.zeros_like(Er)
    N1Total = (neut_flux*d_Eneut).sum()
    N2Total = (neut_flux*d_Eneut/(en_centers**2)).sum()
    N1_en = N1Total - (neut_flux*d_Eneut).cumsum()
    N2_en = N2Total - (neut_flux*d_Eneut/(en_centers**2)).cumsum()
    
    
    # For some reason, the last element in N1 and N2 gets a small negative value, which screws
    # up the high-energy part of the resulting spectrum, so below they are forced to be non-negative.
    N1_en[N1_en<0.] = 0.
    N2_en[N2_en<0.] = 0.
    
    nr_spect = np.zeros_like(Er)
    for k0, NN in enumerate(N_N):
        EnuMINs = .5*(Er + np.sqrt(Er**2 + 2*Er*m_N[k0])) # Min neutrino energy for each recoil energy
        #N1_Er = np.interp(np.sqrt(m_N[k0]*Er/2),en_centers,N1_en)
        #N2_Er = np.interp(np.sqrt(m_N[k0]*Er/2),en_centers,N2_en)
        N1_Er = np.interp(EnuMINs,en_centers,N1_en)
        N2_Er = np.interp(EnuMINs,en_centers,N2_en)
        
        # The interpolation algorithm goes wonky when evaluating a point outside the input range (i.e. 
        # when it is asked to extrapolate instead of interpolate), so I have to force the neutrino 
        # integrals to be zero when that happens.
        #cut_ErTooGreat = Er > (2*(en_centers.max()**2)/m_N[k0])
        max_EnCenters = en_centers.max()
        cut_ErTooGreat = Er > 2*(max_EnCenters**2)/(m_N[k0]+2*max_EnCenters)
        N1_Er[cut_ErTooGreat] = 0.
        N2_Er[cut_ErTooGreat] = 0.
        
        nr_spect_component = (_physCons['Gf']**2)/(4*np.pi)*(Q_N[k0]**2) * (FQ_Helm(Er*1e3,A_N[k0])**2) * \
                             (N1_Er - (m_N[k0]*(Er)/2)*N2_Er)*abund_N[k0]
        nr_spect_component[nr_spect_component<0.] = 0.
        nr_spect = nr_spect + nr_spect_component
    
    # convert from natural units to physical units (/tonn/year/keVnr)
    #nr_spect = nr_spect *((_physCons['c']**2)*1000*3600*24*365.25)/(_physCons['hbar']*(1e6)*_physCons['eCh']*(1e3))
    
    # convert from natural units to physical units (/kg/day/keVnr)
    nr_spect = nr_spect * ((_physCons['c']**2.)*3600.*24.)/(_physCons['hbar']*(1e6)*_physCons['eCh']*(1e3))
    
    # floating-point arithmetic sometimes produces tiny negative values; forcing them to zero here:
    #nr_spect[nr_spect<0.] = 0.
    #   ---------- </CALCULATE RATES> -----------   #
    return nr_spect

def Er_max_CNNS(target='Xe', nu_source='B8'):
    """
    Returns the maximum recoil energy, in keV, for neutrinos from a source in a target.
    Inputs: -----------------------
      target: (Optional) Default: "Xe", see documentation for function 'CoNeutRecoilSpect'
              in this same module.
   nu_source: (Optional) Default: "B8", see documentation for function 'CoNeutRecoilSpect'
              in this same module.
    Outputs: ----------------------
       E_max: Maximum possible energy of a recoil in the specified target from the 
              specified source.  Given in units of keV
    
    2016.02.18 -- A. Manalaysay
    """
    m_N = _getTargetParams(target)[3] # masses in MeV
    neut_flux, en_centers, en_edges = _setNuSource(nu_source) # en_centers in MeV
    nFlux_nonzero = neut_flux > 0.
    max_NeutEnergy = en_centers[nFlux_nonzero].max() 
    Er_max = 2*(max_NeutEnergy**2)/(m_N.min() + 2*max_NeutEnergy) # obvs this is in MeV
    Er_max = Er_max * (1e3) # converting MeV into keV
    return Er_max

def genRandEnergiesCNNS(n, target='Xe', nu_source='B8'):
    """
    Generate random energies according to coherent neutrino-nucleus scattering from a 
    given source.
    Inputs: -----------------------
           n: Number of energies to generate.
      target: (Optional) Target used in the detection of the neutrinos.  See documentation
              for function 'CoNeutRecoilSpect' in this same module. Default: "Xe"
   nu_source: (Optional) The neutrino source.  See documentation for function 
              'CoNeutRecoilSpect' in this same module.  Default: "B8"
    Outputs: ----------------------
          Er: Nuclear recoil energies, in keV.  This is a numpy ndarray.
    
    2016.02.18 -- A. Manalaysay
    """
    # Calculate maximum possible recoil energy:
    Er_max = Er_max_CNNS(target=target, nu_source=nu_source)
    
    # Generate array of recoil energies from which to sample
    Enr = np.linspace(0, 1.1*Er_max,2e3)
    dR = CoNeutRecoilSpect(Enr, target=target, nu_source=nu_source)
    Rcum = dR.cumsum()/dR.sum()  # Energy cumulative distribution function
    cutRange = Rcum < Rcum[-1]
    Rcum = Rcum[cutRange]
    dR = dR[cutRange]
    Enr = Enr[cutRange]
    r_uniform = rand(n)
    Er = np.interp(r_uniform, Rcum, Enr)
    return Er



















