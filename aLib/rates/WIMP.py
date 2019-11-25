"""
WIMP rates module
"""

import numpy as np
import sys
from scipy.special import erf
from numpy.random import rand


def FQ_Helm(Q,A):
    """
    Using the Helm form factor from Lewin and Smith. Q is the energy in keV, A is the mass number of 
    the nucleus (dimensionless).
    Inputs:-----------------------------------------------------------------------------------------
              Q: ndarray of recoil energy values, in keV.  This should be the raw kinetic energy of the 
                 recoiling nucleus.
              A: The mass number of the target nucleus.
    Outputs:----------------------------------------------------------------------------------------
             ff: The form factor for A, evaluated at Q (ndarray).
    
    2014.06.05 -- A. Manalaysay
    """
    cparam = 1.23*(A**(1/3))-0.6
    aparam = 0.52
    sparam = 0.9
    j1 = lambda x: np.sin(x)/(x**2) - np.cos(x)/x
    
    rn = np.sqrt((cparam**2)+(7/3)*(np.pi**2)*(aparam**2)-5*(sparam**2))
    q = (6.92e-3)*np.sqrt(A)*np.sqrt(Q)
    #ff = 3*(j1(q*rn)/(q*rn))*np.exp(-((q*sparam)**2)/2)
    #ff = np.zeros_like(Q)
    ff = np.ones_like(Q)
    cutNZ = q*rn != 0
    ff[cutNZ] = 3*(j1(q[cutNZ]*rn)/(q[cutNZ]*rn))*np.exp(-((q[cutNZ]*sparam)**2)/2)
    return ff

def zeta_McCabe(Er,A,M_W, v_0=220., v_esc=544., v_earth=245.):
    """
    This function returns the value of zeta(E_R), from C. McCabe, PRD 82, 023530 (2010) arXiv: 1005.0579
    
    Inputs:-----------------------------------------------------------------------------------------
             Er: ndarray of recoil energy values, in keV.  This should be the raw kinetic energy of the 
                 recoiling nucleus.
              A: The mass number of the target nucleus.
            M_W: The mass of the WIMP, in GeV/c^2
            v_0: (kwarg, optional) The characteristic circular velocity at our location in the galaxy, 
                 in km/s.  Default: 220 km/s
          v_esc: (kwarg, optional) The galactic escape velocity at our location in the galaxy, in km/s.
                 Default: 544 km/s
        v_earth: (kwarg, optional) The pecular velocity of the earth, in km/s.  Default: 245 km/s
    Outputs:----------------------------------------------------------------------------------------
           zeta: The parameter in McCabe's paper that encompasses the entire velocity integral that goes
                 into the calculation of the differential rate.
    
    2014.06.05 -- A. Manalaysay
    """
    # Initial parameter calculations
    v_0 = v_0 / 2.998e5             # convert to units of c
    v_esc = v_esc / 2.998e5         # convert to units of c
    v_earth = v_earth / 2.998e5     # convert to units of c
    MN = A * 0.931494061            # calculate the mass of the nucleus in GeV/c^2
    Mr = (M_W * MN) / (M_W + MN)    # calculate the WIMP-nucleus reduced mass
    Er = Er / 1e6                   # convert from keV to GeV
    
    # Calculate the ndarray of v_min, following McCabe's Eq.2, with delta=0
    v_min = np.sqrt(Er*MN/(2*(Mr**2)))
    
    # initialize the vector of xi values
    zeta = np.zeros_like(Er)
    
    # McCabe normalizes each velocity by v_0:
    x_esc   = v_esc / v_0
    x_min   = v_min / v_0
    x_earth = v_earth / v_0
    
    # The normalization constant for the velocity distribution (eq. B3)
    N = (np.pi**(3/2))*(v_0**3) * \
        (erf(x_esc) - 4/np.sqrt(np.pi)*np.exp(-(x_esc**2))*(x_esc/2 + (x_esc**3)/3))
    
    # McCabe gives four conditions regarding the relationship between v_min, and calculates 
    # xi(Er) separately for each
    cut1 = x_min <= x_esc - x_earth
    cut2 = (x_min > x_esc - x_earth) & (x_min < x_esc + x_earth)
    cut3 = x_min < x_earth - x_esc
    cut4 = x_min >= x_earth + x_esc
    # Below are the four conditions that McCabe identifies
    zeta[cut1] = (np.pi**(3/2))*(v_0**2)/2/N/x_earth * \
               (erf(x_min[cut1]+x_earth) - erf(x_min[cut1]-x_earth) - \
               4*x_earth/np.sqrt(np.pi)*np.exp(-(x_esc**2))*(1+x_esc**2-(x_earth**2)/3-x_min[cut1]**3))
    zeta[cut2] = (np.pi**(3/2))*(v_0**2)/2/N/x_earth * \
               (erf(x_esc) + erf(x_earth - x_min[cut2]) - \
               2/np.sqrt(np.pi) * np.exp(-(x_esc**2)) * \
               (x_esc+x_earth-x_min[cut2]-(1/3)*(x_earth-2*x_esc-x_min[cut2])*(x_esc+x_earth-x_min[cut2])**2))
    zeta[cut3] = 1/v_0/x_earth
    zeta[cut4] = 0
    return zeta
    

def WIMPdiffRate(Er, mW=50., sig=1e-45, A=131.29, v_0=220., v_esc=544., v_earth=245.):
    """
    The differential rate of WIMP recoils in a target, in units of evts/kg/day/keV.  Using the formalism
    of McCabe: C. McCabe, PRD 82, 023530 (2010) arXiv: 1005.0579.  This is for 'vanilla' spin-
    independent, isospin-invariant scattering.
    
    Inputs:-----------------------------------------------------------------------------------------
             Er: ndarray of recoil energy values, in keV.  This should be the raw kinetic energy of the 
                 recoiling nucleus.
             mW: (optional) Mass of the WIMP, in GeV. Default: 50
            sig: (optional) WIMP-nucleon cross section, in cm^2.  Default=1e-45
              A: (optional) Mass number of the target nucleus, in amu.  Default: 131.29
            v_0: (optional) Characteristic circular velocity, in km/s, at our point in the galaxy.
                 Default: 220 km/s
          v_esc: (optional) Escape velocity at our point in the galaxy, in km/s. Default: 544 km/s
        v_earth: (optional) Velocity of the earth within the galaxy, in km/s. Default: 245 km/s
    Outputs:----------------------------------------------------------------------------------------
             dR: Different
    
    2014.06.05 -- A. Manalaysay
    """
    
    #   -------- <SET PHYSICAL CONSTANTS> --------   #
    #Gf = 1.1663787e-11 # MeV^{-2}
    c = 299792458. # m/s
    hbar = 6.58211928e-22 # MeV*s
    eCh = 1.602176565e-19 # Coulombs; also this many Joules = 1 eV
    #sin2thW = 0.2386 # low-E limit, from PDG
    #   -------- </SET PHYSICAL CONSTANTS> --------   #
    
    # ---------- <SET PARAMS IN NAT UNITS> ---------- #
    # rho in GeV/cm^3; convert GeV to MeV, cm^3 to m^3, m^3 to MeV^-3 (note, hbar above is in MeV*s)
    rho = 0.3 * (1e3)* (100.**3.) * (c**3.) * (hbar**3.)
    # sig in cm^2; convert cm^2 to m^2, m^2 to MeV^-2
    sig = sig * (1/(100.**2.)) * (1./c**2.) * (1./hbar**2.)
    # mW is in GeV, need to convert to MeV
    mW = mW * 1e3
    # mN is the mass of the nucleus.  Need to convert input 'A' to units of MeV
    mN = A * 931.494061
    # mn is the mass of a neutron, in MeV
    mn = 939.565378
    # mu_Wn is the WIMP-neutron reduced mass
    mu_Wn = mW*mn/(mW + mn)
    # ---------- </SET PARAMS IN NAT UNITS> ---------- #
    
    # ---------- <CALCULATE THE RATE> ---------- #
    # dR is in units of evts/energy/time/detector_mass
    dR = (rho/mN/mW)*(mN*sig*(A**2.)/2./(mu_Wn**2.)) * \
         (FQ_Helm(Er,A)**2.) * \
         zeta_McCabe(Er,A,mW/1e3, v_0=v_0, v_esc=v_esc, v_earth=v_earth)
    # convert dR into evts/kg/day/keV
    dR = dR * ((c**2.)*3600.*24.)/(hbar*(1e6)*eCh*(1e3))
    # ---------- </CALCULATE THE RATE> ---------- #
    return dR

def Er_max_WIMP(mW=50., A=131.29, v_0=220., v_esc=544., v_earth=245.):
    """
    Return the maximum recoil energy, in keV.
    Inputs: -----------------------
           mW: (Optional) WIMP mass, in GeV/c^2. Default: 50 GeV
            A: (Optional) Mass of the target nucleus, in amu. Default: 131.29
          v_0: (Optional) Characteristic galactic circular velocity, at our location, km/s.
               Default: 220 km/s
        v_esc: (Optional) Galactic escape velocity at our location, in km/s.
               Default: 544 km/s
    Outputs: ----------------------
    
    """
    c = 2.998e5 # in km/s
    v_WIMPmax = (v_earth + v_esc)/c # max relative WIMP velocity, in Earth's frame
    E_WIMP = .5 * (v_WIMPmax**2) * mW * 1e6 # max WIMP ke in units of keV
    m_Nuc = 0.931494061 * A # mass of the target nucleus, in GeV
    Er_MAX = 4. * E_WIMP * mW*m_Nuc / ((mW + m_Nuc)**2) # max recoil energy, in keV
    return Er_MAX

def genRandEnergiesWIMP(n, mW=50., A=131.29, v_0=220., v_esc=544.):
    """
    Generate random energies according to the expected recoil spectrum of a given WIMP mass.
    Inputs: -----------------------
           n: Number of energies to generate.
          mW: (Optional) Mass of the WIMP in GeV/c^2. Default: 50 GeV
           A: (Optional) Mass of of the nucleus, in amu. Default: 131.29 amu
         v_0: (Optional) Characteristic circular velocity of objects around the galaxy at our 
              location, in km/s. Default: 220 km/s
       v_esc: (Optional) Galactic escape velocity of objects at our location, in km/s. 
              Default: 544 km/s
    Outputs: ----------------------
          Er: Nuclear recoil energies, in keV.  This is an numpy ndarray.
    
    2014.12.11 -- A. Manalaysay
    """
    # Calculate maximum possible recoil energy:
    Er_MAX = Er_max_WIMP(mW=mW, A=A, v_0=v_0, v_esc=v_esc)
    
    # Generate array of recoil energies from which to sample
    Enr = np.linspace(0, 1.1*Er_MAX,2e3)
    dR = WIMPdiffRate(Enr, mW=mW, A=A, v_0=v_0, v_esc=v_esc)
    Rcum = dR.cumsum()/dR.sum()  # Energy cumulative distribution function
    cutRange = Rcum < Rcum[-1]
    Rcum = Rcum[cutRange]
    dR = dR[cutRange]
    Enr = Enr[cutRange]
    r_uniform = rand(n)
    Er = np.interp(r_uniform, Rcum, Enr)
    return Er





























