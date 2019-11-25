"""
This module generates simulated S1 and S2 (and other) values, based on given desired
input energy spectra.

If you want to generate simulated S1 and S2 values, this is available via the function 
"S1S2fromEnr"; see its help documentation.

2014.06.27 -- A. Manalaysay - For the moment, only applying this to nuclear recoils.
"""

import os
import numpy as np
from numpy.random import randn
from numpy.random import rand
from aLib import *
from aLib.rates.WIMP import WIMPdiffRate
from aLib.astats import binornd
import scipy.stats as st
import matplotlib as mpl

PathToThisFile = os.path.split(__file__)[0]

def _almostEq(a,b, tol=1e-5):
    """
    Returns true for elements of a and b are within a small value (kwarg 'tol') of
    each other.
    """
    return np.abs(a-b) < tol

def genNRfromSilver(**kwargs):
    """
    Generate NR S1 and S2 values by randomly selecting events from the Silver NR sims.
    """
    if ('WIMPmass' in kwargs.keys()) and ('WIMPsig' in kwargs.keys()):
        flag_WIMP = True
    elif ('expConst' in kwargs.keys()) and ('expNorm' in kwargs.keys()):
        flag_WIMP = False
    else:
        raise ValueError("kwargs 'WIMPmass' and 'WIMPsig' must be set, OR 'expConst' and 'expNorm'.")
    if 'expsr' not in kwargs.keys():
        raise ValueError("A kwarg for the exposure, in kg*days, must be supplied.")
    
    # effNR(E) gives the efficiency to detect a nuclear recoil event of energy Er (in keV).
    # This is determined from the Silver Sims, which probe the full volume of the detector.
    # This function is the CDF of two superimposed exponential PDFs.
    effNR = lambda E: (2.1*0.3/(.014*2.1+0.3)) * \
                      ((1/2.1)*(1-np.exp(-2.1*E)) + (.014/0.3)*(1-np.exp(-.3*E)))
    
    # Er is the array of energy values, whose spacing and extent are defined by those
    # energies simulated in the Silver sims.
    Er = np.arange(.1,100.1,.1)
    
    #def WIMPdiffRate(Er, mW=50., sig=1e-45, A=131.29, v_0=220., v_esc=544., v_earth=245.):
    if flag_WIMP:
        # dRdEdM = "rate per energy per mass", or "events per time per energy per mass" (i.e. DRU)
        dRdEdM = WIMPdiffRate(Er, mW=kwargs['WIMPmass'], sig=kwargs['WIMPsig'])
    else:
        # blah... need to think more about this first.
        print("Oops, no exp support yet.")
    
    # dNdE is the number at each energy, since we are multiplying out the mass and time (and bin width)
    N_at_E = dRdEdM * kwargs['expsr'] * np.diff(Er[:2])
    
    # Now scale this rate by the efficiency
    N_at_E = N_at_E * effNR(Er)
    
    # Find the location of the current module, and load the data file with 
    # the results of the SilverSim.
    # CurrentFileAndPath = os.path.split(__file__)
    # dS = mfile.mload(os.path.join(CurrentFileAndPath[0],"SilverSimsForNRgeneration.mat"))
    dS = mfile.mload(os.path.join(PathToThisFile,"SilverSimsForNRgeneration.mat"))
    
    # I will now treat this number, at each energy, as a Poisson mean, and sample a number
    # assuming that mean.
    S1 = np.array([])
    S2 = np.array([])
    N_real = np.zeros_like(N_at_E)
    for k, EEr in enumerate(Er):
        # Find the indices of Silver sims which match the energy of this iteration
        ErIndsAll = np.nonzero(_almostEq(dS.mc_E_keV,EEr))[0]
        
        # Determine that one of these events should get picked
        hitProb = N_at_E[k] / len(ErIndsAll)
        
        # Generate a boolean array that samples randomly these events with the appropriate probability
        ErIndsPick = np.random.rand(len(ErIndsAll)) < hitProb
        N_real[k] = ErIndsPick.sum()
        
        #print(ErIndsPick.__repr__())
        # Grab those indices in ErIndsAll which were chosen
        ErIndsChosen = ErIndsAll[ErIndsPick]
        
        if len(ErIndsChosen) > 0:
            S1 = np.hstack([S1, dS.S1[ErIndsChosen]])
            S2 = np.hstack([S2, dS.S2[ErIndsChosen]])
    
    #return S1, S2, N_at_E, N_real
    return S1, S2

def binorndFano(N, p, fano=1.):
    """
    Generate a binomial random number with the possible modification of the distribution's width.
    
    Inputs: 
         N: Number of trials (also acts as an absolute upper bound).
         p: Probability of success of each trial.
      fano: (optional) Modification of the binomial's variance: 
                    fano = 1 -> no modification (default behavior)
                    fano < 1 -> distribution's width is tightened
                    fano > 1 -> distribution's width is widened.
    Outputs:
         k: A random integer (or array of integers) that are selected from the desired distribution.
    
    2014.07.25 -- A. Manalaysay 
    """
    # ----- <INPUT CHECKING> -----#
    if (type(N)==np.ndarray) and (type(p)==np.ndarray):
        if len(N) != len(p):
            raise ValueError("If inputs N and p are arrays, they must be of the same dimension.")
    if type(N)==np.ndarray:
        numSims = len(N)
    elif type(p)==np.ndarray:
        numSims = len(p)
    else:
        numSims = 1
    # ----- </INPUT CHECKING> -----#
    if fano > 1:
        #p = p + (fano-1)*np.sqrt(p*(1-p))*(1/N)*randn(numSims)
        p = p + np.sqrt(((fano-1)/N))*randn(numSims)
        p[p<0] = 0.
        p[p>1] = 1.
        k = binornd(N,p)
    elif fano < 1:
        Nfano = np.round(N/fano)
        cutNeq0 = N==0
        kfano = np.float64(binornd(Nfano,p))
        k = np.zeros_like(kfano)
        k[~cutNeq0] = np.floor(kfano[~cutNeq0]*(N[~cutNeq0]/Nfano[~cutNeq0]) + rand(numSims)[~cutNeq0])
    else:
        k = binornd(N,p)
    return k

def poissrndFano(mu, F=1., Nmax=np.inf, size=1):
    """
    Generate a Poisson random number with the possible modification of the distribution's width.
    
    Inputs: 
        mu: Number of trials (also acts as an absolute upper bound).
         F: (optional) Modification of the binomial's variance: 
                    fano = 1 -> no modification (default behavior)
                    fano < 1 -> distribution's width is tightened
                    fano > 1 -> distribution's width is widened.
      Nmax: (optional) Maximum possible output (kind of like N in a binomial distribution).  If Nmax
            is an array, it must be of the same size as mu (if mu is also an array).
            Default: infinity
      size: (optional) If mu is an array. then kwarg 'size' does nothing.  Otherwise, it specifies
            the number of output numbers.
    Outputs:
     rNums: A random integer (or array of integers) that are selected from the desired distribution.
    
    2014.08.26 -- A. Manalaysay     
    """
    # ----- <INPUT CHECKING> -----#
    if (type(mu)==np.ndarray) and (type(Nmax)==np.ndarray):
        if len(mu) != len(Nmax):
            raise ValueError("If inputs mu and Nmax are arrays, they must be of the same dimension.")
    if type(mu)==np.ndarray:
        numSims = len(mu)
    elif type(Nmax)==np.ndarray:
        numSims = len(Nmax)
    else:
        numSims = size
    # ----- </INPUT CHECKING> -----#
    
    if F > 1.:
        poissMean = mu + np.sqrt((F-1.)*mu)*randn(numSims)
        poissMean[poissMean<=0] = 1e-100
        rNums = st.poisson.rvs(poissMean)
    else:
        muMod = mu.copy()
        muMod[mu<=0.] = 1e-100
        rNums = np.floor(st.poisson.rvs(muMod/F, size=numSims)*F + rand(numSims))
    if np.any(rNums > Nmax):
        rNums[rNums > Nmax] = Nmax[rNums > Nmax]
    return rNums

# Define values for the NEST scint. and ionization model:
NEST          = S()
NEST.W        = 0.0137 # keV/quantum
NEST.Z        = 54. # Xe atomic number
NEST.k        = 0.1394
NEST.alpha    = 1.240
NEST.zeta     = 0.0472
NEST.beta     = 239.
NEST.gamma    = 0.0554
NEST.delta    = 0.0620
NEST.eta      = 3.32
NEST.theta    = 1.141
def NEST_NphNe(Er, F=181., f_Ni=1., f_Nex=1., f_recomb=1., f_biex=1.):
    """
    This is an implementation of NEST for LXe, equivalent to NEST v1.0.  This function
    generates random values of [ionization] electrons and [scintillation] photons,
    given an array of energies and an input field.  No detector effects are included, as
    this function simulates the integer, true number of these values.
    
    Inputs: ------------------------------------------
           Er: Array of nuclear recoil energies, in keV.
            F: (optional) The magnitude of the applied electric field, in V/cm.  This input
               can be a single, scalar value (in which case it is applied to each element of
               Er), or it can be an array, with each element corresponding to its associated 
               Er value.  Default: 181. V/cm
         f_Ni: (optional) "fano" factor for ion creation. Default: 1
        f_Nex: (optional) "fano" factor for exciton creation. Default: 1
     f_recomb: (optional) "fano" factor for recombination creation. Default: 1
       f_biex: (optional) "fano" factor for biexcitonic quenching creation. Default: 1
    Outputs: ------------------------------------------
          Nph: Number of scintillation photons.
           Ne: Number of ionization electrons.
    
    2014.07.24 -- A. Manalaysay
    """
    # First, determine the number of quanta ala Lindhard and put into Nex and Ni:
    ep          = 11.5*Er*(NEST.Z**(-7/3))
    lambd       = 3*(ep**0.15) + 0.7*(ep**0.6) + ep
    lindParam   = NEST.k*lambd / (1 + NEST.k*lambd)
    NexNi       = NEST.alpha*(F**(-NEST.zeta))*(1-np.exp(-NEST.beta*ep))
    # Generate the Ni and Nex separately, not by first generating Nq and then distributing;
    # there is really no reason Ni and Nex should be anticorrelated.
    '''
    Ni          = binorndFano(np.round(Er/NEST.W), lindParam/(1+NexNi), f_Ni)
    Nex         = binorndFano(np.round(Er/NEST.W), lindParam*NexNi/(1+NexNi), f_Nex)
    '''
    Ni          = poissrndFano(np.round(Er/NEST.W)*lindParam/(1+NexNi), F=f_Ni,Nmax=np.round(Er/NEST.W))
    Nex         = poissrndFano(np.round(Er/NEST.W)*(lindParam*NexNi/(1+NexNi)), F=f_Nex, Nmax=np.round(Er/NEST.W))
    Nq          = Ni + Nex
    # Distribute Ni to Nph and Ne due to recombination:
    TIB                 = NEST.gamma * (F ** (-NEST.delta)) # Thomas-Imel parameter
    xi                  = (Ni/4) * TIB
    cutXiPos            = xi>0. # check that xi is greater than zero, for the log later on.
    rFrac               = np.zeros_like(xi) # "rFrac" = recombination fraction (or probability)
    rFrac[cutXiPos]     = 1- (1/xi[cutXiPos])*np.log(1 + xi[cutXiPos])
    #Ne                  = binorndFano(Ni, 1-rFrac, f_recomb)
    Ne                  = poissrndFano(Ni*(1-rFrac), F=f_recomb, Nmax = Ni)
    Nph                 = np.round(Nq - Ne) # should not need to round, but doing so just to be sure
    
    # Biexcitonic quenching:
    #Nph = binorndFano(Nph, 1/(1 + NEST.eta * (ep**NEST.theta)), f_biex)
    Nph = poissrndFano(Nph/(1 + NEST.eta * (ep**NEST.theta)), F=f_biex, Nmax=Nph)
    return Nph, Ne


# Define values specific to LUX:
LUX                 = S()
LUX.F               = 181. # E-field, in V/cm
#LUX.g1             = 0.0908 # photon detection efficiency (NOT phe/photon, because of the VUV stuff)
LUX.g1              = 0.12 # photon detection efficiency (NOT phe/photon, because of the VUV stuff)
LUX.e_extract_effic = 0.682 # electron liquid-gas extraction efficiency
#LUX.SE_S2b_mean    = 9.7 # mean value of the single-electron S2 on the bottom PMT array
LUX.SE_S2b_mean     = 9.35 # mean value of the single-electron S2 on the bottom PMT array
#LUX.SE_S2b_width   = 3.53 # stdv value of the single-electron S2 on the bottom PMT array
LUX.SE_S2b_width    = 3.64 # stdv value of the single-electron S2 on the bottom PMT array
LUX.S2_botFrac      = 10.4/24.6 # ratio of SE peaks from the PRL, should be independent of VUV stuff.
LUX.SPE_res         = 0.35 # SPE resolution (real SPE, not average of detected photons)
LUX.dT_max          = 324. # maximum drift time
LUX.dblPheFrac      = 0.2 # Fraction of detected photons that produce double photoelectrons
LUX.NumPMTs         = 122 - 3 # 122 PMTs total, Run3 had 3 left unbiased

def S1S2fromEnr(Er, e_lifetime=800.,f_Ni=1., f_Nex=1., f_recomb=1., f_biex=1.):
    """
    Take an array of nuclear recoil energies (Er) and generate a pair of S1 and S2 values, as per
    the NEST model (outlined in Matthew's slides from shortly before 7/7/2014).
    
    Er is assumed to be a numpy array, not just a single value, in real energy.
    
    Inputs: ----------------------------------------------------------------------------------------
             Er: Raw recoil energies in keV.
     e_lifetime: (optional) Electron lifetime, in microseconds.  Default: 800. us.  This value can
                 either be a scalar, or an array.  If it is an array, it must have the same length
                 as Er.
           f_Ni: (optional) Fano-like factor for the fluctuations in number of ions. Default=1.
          f_Nex: (optional) Fano-like factor for the fluctuations in number of excitons. Default=1.
       f_recomb: (optional) Fano-like factor for the fluctuations recombination. Default=1.
         f_biex: (optional) Fano-like factor for the fluctuations biexcitonic quenching. Default=1.
    Outputs: ---------------------------------------------------------------------------------------
         S1_obs: Observed area of the S1 peak, in units of detected photons.  Signal represents the
                 sum over the full PMT array.
         S2_obs: Observed area of the S2 peak, in units of detected photons.  Signal represents the
                 sum over the full PMT array.
        S2b_obs: Observed area of the S2 peak (bottom PMT array only), in units of detected photons.
             dT: Drift time of the event (chosen from a uniform random number).
          nFold: Boolean, True if the event satisfied the 2-fold coincidence requirement.
    
    Example: ------------------------------------------
    In [1]: from aLib import *
    In [2]: import scipy.stats as st
    In [3]: Er = st.expon.rvs(scale=30., size=1e3) # generate exponentially random recoil energies
    In [4]: S1, S2, S2b, dT, nFold = rates.S1S2fromEnr(Er) # using default optional inputs
    In [5]: cut = nFold & (S2 > 200.)
    In [6]: plot(S1[cut], log10(S2b[cut]/exp(-dT[cut]/800.)/S1[cut]), '.')
    In [7]: # Above, I have made a cut on S2_total_raw, and plotted S2_bottom_corrected
    
    2014.07.28 -- A. Manalaysay
    """
    
    # Generate a random number of electrons and photons via the NEST module
    Nph, Ne = NEST_NphNe(Er, LUX.F, f_Ni=f_Ni, f_Nex=f_Nex, f_recomb=f_recomb, f_biex=f_biex)
    
    # Given Nph (raw), determine the number of detected photons ("dph" = "detected photons")
    S1_raw_dph = binornd(Nph, LUX.g1) # binomial for light collection
    # Choose how many of the detected photons will have an extra photoelectron
    Num_extra_S1phe = binornd(S1_raw_dph,LUX.dblPheFrac)
    # Add the extra phe to the number of detected photons to get the detected phe
    S1_raw_phe = S1_raw_dph + Num_extra_S1phe
    
    # Generate a boolean of whether or not the event was detected in 2 or more PMTs
    # For N pmts and M detected photons, the probability to satisfy the 2-fold requirement
    # is effic = 1 - N^(1-M).  LUX.NumPMTs
    nFold = rand(len(Er)) < (1 - LUX.NumPMTs**(1-S1_raw_dph))
    
    # Uniform random drift time
    dT = LUX.dT_max*rand(len(S1_raw_phe))
    
    # Calculate number of extracted electrons, taking into account losses due to drifting and losses
    # due to incomplete liquid->gas extraction
    Ne_extracted = binornd(Ne, np.exp(-dT/e_lifetime)*LUX.e_extract_effic)
    # From the extracted electrons, generate a random number of dph
    #S2b_raw_phe = LUX.SE_S2b_mean*Ne_extracted+np.sqrt(Ne_extracted)*LUX.SE_S2b_width*randn(len(Ne_extracted))
    #S2b_raw_phe = np.round(S2b_raw_phe)
    S2b_raw_dph = LUX.SE_S2b_mean*Ne_extracted+np.sqrt(Ne_extracted)*LUX.SE_S2b_width*randn(len(Ne_extracted))
    S2b_raw_dph = np.round(S2b_raw_dph)
    # Do the same calculation for the top PMTs, using the botom/full ratio to get the top SE mean.
    SE_S2t_mean  = LUX.SE_S2b_mean * (1-LUX.S2_botFrac)/LUX.S2_botFrac
    SE_S2t_width = LUX.SE_S2b_width * np.sqrt((1-LUX.S2_botFrac)/LUX.S2_botFrac)
    #S2t_raw_phe = SE_S2t_mean*Ne_extracted+np.sqrt(Ne_extracted)*SE_S2t_width*randn(len(Ne_extracted))
    #S2t_raw_phe = np.round(S2t_raw_phe)
    S2t_raw_dph = SE_S2t_mean*Ne_extracted+np.sqrt(Ne_extracted)*SE_S2t_width*randn(len(Ne_extracted))
    S2t_raw_dph = np.round(S2t_raw_dph)
    # Calculate extra photoelectrons
    Num_extra_S2bphe = binornd(S2b_raw_dph,LUX.dblPheFrac)
    Num_extra_S2tphe = binornd(S2t_raw_dph,LUX.dblPheFrac)
    S2b_raw_phe = S2b_raw_dph + Num_extra_S2bphe
    S2t_raw_phe = S2t_raw_dph + Num_extra_S2tphe
    
    # Now generate S1 and S2 areas, in observed phe
    S1_obs = S1_raw_phe + LUX.SPE_res*np.sqrt(S1_raw_phe)*randn(len(S1_raw_phe))
    S1_obs = S1_obs / (LUX.dblPheFrac*2. + (1-LUX.dblPheFrac)*1.) # rescale to account for the VUV gains
    S2b_obs = S2b_raw_phe + LUX.SPE_res*np.sqrt(S2b_raw_phe)*randn(len(S2b_raw_phe))
    S2b_obs = S2b_obs / (LUX.dblPheFrac*2. + (1-LUX.dblPheFrac)*1.) # rescale to account for the VUV gains
    S2t_obs = S2t_raw_phe + LUX.SPE_res*np.sqrt(S2t_raw_phe)*randn(len(S2t_raw_phe))
    S2t_obs = S2t_obs / (LUX.dblPheFrac*2. + (1-LUX.dblPheFrac)*1.) # rescale to account for the VUV gains
    S2_obs = S2b_obs + S2t_obs
    
    return S1_obs, S2_obs, S2b_obs, dT, nFold






















