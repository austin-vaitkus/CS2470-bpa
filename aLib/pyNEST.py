from __future__ import print_function
import numpy as np
from numpy.random import rand
from numpy.random import randn
from numpy import pi
import scipy.stats as st
from scipy.special import erf
import scipy.interpolate as ip
from scipy import io
import os

# The following are default values that are not adjustable by the user. Other default 
# values that are user-adjustable are set as kwargs in the function definition (below).
AVO            = 6.022e23
e_life         = 800. # us
S2_rat         = 0.441 # fraction on bottom array?
gasGap         = 0.5   # EL gap in cm
elGain         = 0.140
elThld         = 0.474
numberPMTs     = 119.
erf_mu         = 1.61  # S1 threshold
erf_sg         = 0.61  # S1 threshold
nFold          = 2.     # number of PMTs necessary for coincidence
spEff          = 0.3   # threshold in phe
P_dphe         = 0.173 # probability of a second phe (given a single phe)
vuvFudgeFactor = 0.915
Xe_rho         = 2.888 # xenon density
#gasRho         = 15e-3 # gm/cm^3 --> promoting this to an input: run4 looking like maybe ~17e-3-ish
W              = (21.717-2.7759*Xe_rho)*1e-3
Z              = 54.
molarMass      = 131.293
# rootNest.cc lines 116-118
Fex            = 1.
Fi             = 1.
Fr             = 1.
kappa          = 0.1394 # for the Lindhard factor
m_e            = 511. # keV
aa             = 1/137. # fine structure constant
v_d            = 0.151 # cm/microsecond
det_edge       = 325. # microseconds

# Find the path to THIS file:
PathToThisFile = os.path.split(__file__)[0]

# Load the field maps.
Fmap_run4 = np.loadtxt(os.path.join(PathToThisFile,'run4_axisymmetric_fieldmodel.txt'),
                       skiprows=1,delimiter=',')
Fmap_run3 = np.loadtxt(os.path.join(PathToThisFile,'run3_axisymmetric_fieldmodel.txt'),
                       skiprows=1,delimiter=',')
Fmap_run4 = Fmap_run4[~np.isnan(Fmap_run4[:,2]),:] # cut out NaN rows
Fmap_run3 = Fmap_run3[~np.isnan(Fmap_run3[:,2]),:] # cut out NaN rows

# Load Richard's corrections file based on TrueKrypCal results, and format for python interpolation
KrypcalC = io.loadmat(os.path.join(PathToThisFile,'TrueKrypCal_Corrections.mat'), squeeze_me=True)
KrypcalC['s2_xy_xbinsPTS'],KrypcalC['s2_xy_ybinsPTS'] = np.meshgrid(KrypcalC['s2_xy_xbins'],KrypcalC['s2_xy_ybins'])
KrypcalC['s2_xy_xbinsPTS'] = KrypcalC['s2_xy_xbinsPTS'].flatten(1)
KrypcalC['s2_xy_ybinsPTS'] = KrypcalC['s2_xy_ybinsPTS'].flatten(1)
KrypcalC['norm_S2_xyPTS'] = KrypcalC['norm_S2_xy'].flatten(1)
KrypcalC['s1_xy_xbinsPTS'],KrypcalC['s1_xy_ybinsPTS'] = np.meshgrid(KrypcalC['s1_xy_xbins'],KrypcalC['s1_xy_ybins'])
KrypcalC['s1_xy_xbinsPTS'] = KrypcalC['s1_xy_xbinsPTS'].flatten(1)
KrypcalC['s1_xy_ybinsPTS'] = KrypcalC['s1_xy_ybinsPTS'].flatten(1)
KrypcalC['norm_S1PTS'] = KrypcalC['norm_S1_both'].flatten(1)

def _tritSpectrum(E, QQ=18.5898, ZZ=2.):
    """
    Private function that returns the spectrum of beta particles coming from tritium decay.
    The spectrum is not normalized, and maxes out at around 1e7.
    """
    
    dR = np.sqrt((E**2) + 2*E*m_e) * (E + m_e) * ((QQ - E)**2) * \
         (1 + pi*aa*ZZ*(E + m_e)/np.sqrt((E**2) + 2*E*m_e))
    dR[E>QQ] = 0.
    dR[E==0.] = 0.
    return dR

def VonNeumann( xMin, xMax, yMin, yMax, N):
    """
    This generates a random array of energies representative of the DD recoil spectrum.
    """
    xTry = xMin + (xMax-xMin)*rand(N)
    yTry = yMin + (yMax-yMin)*rand(N)
    
    FuncValue =  2.4348 - 0.19783*xTry + 0.0060931*(xTry**2) - \
                (8.2243e-5)*(xTry**3) + (4.2505e-7)*(xTry**4)
    FuncValue[xTry<9.8] = 4.1411 - 3.1592*np.log10(xTry[xTry<9.8])
    
    for ievt in range(int(N)):
        while yTry[ievt] > FuncValue[ievt]:
            xTry[ievt] = xMin + (xMax-xMin)*rand()
            yTry[ievt] = yMin + (yMax-yMin)*rand()
            
            if xTry[ievt] < 9.8:
                FuncValue[ievt] = 4.1411-3.1592*np.log10(xTry[ievt])
            else:
                FuncValue[ievt] =  2.4348 - 0.19783*xTry[ievt] + 0.0060931*(xTry[ievt]**2) + \
                                  (-8.2243e-5)*(xTry[ievt]**3) + (4.2505e-7)*(xTry[ievt]**4)
    
    return xTry

def modPoisRnd(poisMeanIn, preFactor):
    """
    Generates random integers from a modified Poisson distributions
    """
    # --------------- <INPUT HANDLING> --------------- #
    if type(preFactor) != np.ndarray:
        preFactor = preFactor * np.ones_like(poisMeanIn)
    if poisMeanIn.shape != preFactor.shape:
        raise ValueError("If 'poisMeanIn' and 'preFactor' are both numpy arrays, " + \
                         "they must be of the same shape.")
    poisMeanIn[poisMeanIn<1e-6] = 1e-6
    # --------------- </INPUT HANDLING> --------------- #
    
    poisOut = np.zeros_like(poisMeanIn)
    Fgeq1 = preFactor >= 1.
    Flt1  = (preFactor < 1.)&(preFactor >1e-6)
    
    # First handle the case when preFactor >=1 (Fgeq1)
    poisMean = poisMeanIn[Fgeq1] + \
               np.sqrt((preFactor[Fgeq1]-1.)*poisMeanIn[Fgeq1])*randn(*poisMeanIn[Fgeq1].shape)
    poisMean[poisMean<1e-6] = 1e-6
    poisOut[Fgeq1] = st.poisson.rvs(poisMean)
    
    # Next handle the case when preFactor < 1 (Flt1)
    poisOut[Flt1] = np.floor(st.poisson.rvs(poisMeanIn[Flt1]/preFactor[Flt1]) * preFactor[Flt1] + \
                       rand(*poisMeanIn[Flt1].shape))
    return poisOut
    """
    if  preFactor >= 1.:
        poisMean = poisMeanIn + np.sqrt((preFactor-1.) * poisMeanIn) * randn(*poisMeanIn.shape)
        poisMean[poisMean<0.] = 0.
        poisOut = st.poisson.rvs(poisMean)
    else:
        poisOut = np.floor(st.poisson.rvs(poisMeanIn/preFactor)*preFactor+rand(*poisMeanIn.shape))
    
    if poisOut < 0.:
        poisOut = 0.
    """

def binomFluct(N00, prob):
    """
    A special binomial fluctuator for the NEST thing.
    """
    N0 = np.array(N00, dtype=np.int64)
    mean_value = N0 * prob
    sigma      = np.sqrt(N0*prob*(1-prob))
    N1         = np.zeros_like(N0)
    
    if (prob == 0.):
        return
    if (prob == 1.):
        return
    
    N1[(N0>0)&(N0 < 10)] = st.binom.rvs(N0[(N0>0)&(N0 < 10)], prob)
    N1[N0 >=10.] = np.array(np.round(np.floor(mean_value[N0>=10.] + sigma[N0>=10.]*randn(*N0[N0>=10.].shape)+0.5)),
                            dtype=N1.dtype)
    N1[N1 > N0] = N0[N1 > N0]
    N1[N1 < 0.] = 0.
    N1[np.isnan(N1)] = 0.
    return N1

def fastbinornd(n, p):
    """
    From Scott Hertel's version of this function in matlab.
    """
    n[np.isnan(n)] = 0.
    n[np.isinf(n)] = 0.
    r = np.zeros_like(n)
    for k in range(len(n)):
        r[k] = (rand(n[k])<p[k]).sum()
    return r

def rGaus(mean, sigma):
    """
    Draw a random variate from a Gaussian distribution with specified mean and sigma.
    """
    out = mean + sigma*randn(*mean.shape)
    return out

def getFieldFromTrueCoords(r, z, fmap='run4'):
    """
    Get the electric field from REAL coordinates.  These are the true coordinates of the 
    event, NOT the observed coordinates.
    Inputs:--------------
            r: Numpy array of real r values of events, in cm.
            z: Numpy array of real z values of events, in cm (must be the same size as r).
         fmap: (Optional) The field configuration to use.  This input is a string, either
               'run4' (default), or 'run3'.
    Outputs:-------------
            E: Numpy array of electric field magnitudes at each event, in units of V/cm.
    
    2014.11.26 -- A. Manalaysay
    """
    assert type(r)==np.ndarray, "Input 'r' must be a numpy array."
    assert type(z)==np.ndarray, "Input 'z' must be a numpy array."
    assert len(r.shape)==1, "Input 'r' must be 1-dimensional."
    assert len(z.shape)==1, "Input 'z' must be 1-dimensional."
    assert len(r)==len(z), "Inputs 'r' and 'z' must be the same length."
    if fmap == 'run3':
        FmapRZ = Fmap_run3
    elif fmap == 'run4':
        FmapRZ = Fmap_run4
    else:
        raise ValueError("I don't recognize your input 'fmap'.")
    
    assert FmapRZ.shape[1]==7, "The field map that was loaded appears to be the " + \
                                   "wrong size; I need 7 columns."
    # The first step is to do a linear 2D interpolation of field values.
    E = ip.griddata(FmapRZ[:,[0,1]],FmapRZ[:,4],np.vstack([r,z]).T)
    # The above interpolation sets E=nan for any points in r,z that lie outside of the field map.
    # The next step identifies these points, and fixes them to be the value of the nearest 
    # field-map point.
    cutEnan = np.isnan(E)
    E[cutEnan] = ip.griddata(FmapRZ[:,[0,1]],FmapRZ[:,4],
                 np.vstack([r[cutEnan],z[cutEnan]]).T, method='nearest')
    return E

def rawCoords2measuredCoords(r_real, z_real, fmap='run4'):
    """
    Get the measured, uncorrected (and unsmeared) positions, in r and dt, based on the real,
    true vertex positions.  This uses the same field map as in the function 
    'getFieldFromTrueCoords'.
    Inputs:--------------
        r_real: Numpy array of real r values of events, in cm.
        z_real: Numpy array of real z values of events, in cm (must be the same size as r).
          fmap: (Optional) The field configuration to use.  This input is a string, either
                'run4' (default), or 'run3'.
    Outputs:-------------
     r_surface: Numpy array of values for the radius of the electron cloud as it crosses the
                liquid surface, in cm.
            dt: Numpy array of values for the drift time of the event, in micro-seconds.
    
    2014.11.26 -- A. Manalaysay
    """
    assert type(r_real)==np.ndarray, "Input 'r_real' must be a numpy array."
    assert type(z_real)==np.ndarray, "Input 'z_real' must be a numpy array."
    assert len(r_real.shape)==1, "Input 'r_real' must be 1-dimensional."
    assert len(z_real.shape)==1, "Input 'z_real' must be 1-dimensional."
    assert len(r_real)==len(z_real), "Inputs 'r_real' and 'z_real' must be the same length."
    if fmap == 'run3':
        FmapRZ = Fmap_run3
    elif fmap == 'run4':
        FmapRZ = Fmap_run4
    else:
        raise ValueError("I don't recognize your input 'kind'.")
    
    assert FmapRZ.shape[1]==7, "The field map that was loaded appears to be the " + \
                                   "wrong size; I need 7 columns."
    r_surface = ip.griddata(FmapRZ[:,[0,1]],FmapRZ[:,2],np.vstack([r_real,z_real]).T)
    dt        = ip.griddata(FmapRZ[:,[0,1]],FmapRZ[:,3],np.vstack([r_real,z_real]).T)
    
    cutRnan = np.isnan(r_surface)
    cutDTnan = np.isnan(dt)
    r_surface[cutRnan] = ip.griddata(FmapRZ[:,[0,1]],FmapRZ[:,2], 
                         np.vstack([r_real[cutRnan],z_real[cutRnan]]).T,method='nearest')
    dt[cutDTnan] = ip.griddata(FmapRZ[:,[0,1]],FmapRZ[:,3],
                         np.vstack([r_real[cutDTnan],z_real[cutDTnan]]).T,method='nearest')
    return r_surface, dt

def measuredCoords2trueCoords(r_surface, dt, fmap='run4'):
    """
    Get the true coordinates (r_real, z_real) from measured coordinates.  It uses the field map
    to reverse-interpolate.
    Inputs:--------------
     r_surface: Numpy array of values for the radius of the electron cloud as it crosses the
                liquid surface, in cm.
            dt: Numpy array of values for the drift time of the event, in micro-seconds.
          fmap: (Optional) The field configuration to use.  This input is a string, either
                'run4' (default), or 'run3'.
    Outputs:-------------
        r_real: Numpy array of real r values of events, in cm.
        z_real: Numpy array of real z values of events, in cm (must be the same size as r).
    
    2014.11.26 -- A. Manalaysay
    """
    assert type(r_surface)==np.ndarray, "Input 'r_surface' must be a numpy array."
    assert type(dt)==np.ndarray, "Input 'dt' must be a numpy array."
    assert len(r_surface.shape)==1, "Input 'r_surface' must be 1-dimensional."
    assert len(dt.shape)==1, "Input 'dt' must be 1-dimensional."
    assert len(r_surface)==len(dt), "Inputs 'r_surface' and 'dt' must be the same length."
    if fmap == 'run3':
        FmapRZ = Fmap_run3
    elif fmap == 'run4':
        FmapRZ = Fmap_run4
    else:
        raise ValueError("I don't recognize your input 'kind'.")
    
    assert FmapRZ.shape[1]==7, "The field map that was loaded appears to be the " + \
                                   "wrong size; I need 7 columns."
    r_real  = ip.griddata(FmapRZ[:,[2,3]],FmapRZ[:,0],np.vstack([r_surface,dt]).T)
    z_real  = ip.griddata(FmapRZ[:,[2,3]],FmapRZ[:,1],np.vstack([r_surface,dt]).T)
    
    cutRnan = np.isnan(r_real)
    cutZnan = np.isnan(z_real)
    r_real[cutRnan] = ip.griddata(FmapRZ[:,[2,3]],FmapRZ[:,0],
                      np.vstack([r_surface[cutRnan],dt[cutRnan]]).T,method='nearest')
    z_real[cutZnan] = ip.griddata(FmapRZ[:,[2,3]],FmapRZ[:,1],
                      np.vstack([r_surface[cutZnan],dt[cutZnan]]).T,method='nearest')
    return r_real, z_real

def Nph_Ne(pt, ef, ke, verbose=False):
    """
    This function determines the initial, absolute numbers of photons and electrons emitted
    from the interaction site.  Each energy (in ke) must have an associated field.
    
    pt can be either 'NR' or 'ER'
    
    The function returns Nph (raw number of photons), and Ne (raw number of electrons)
    """
    # --------------- <LINDHARD> --------------- #
    if (pt=="NR") or (pt=="DD"):   # Lindhard only for nuclear recoils
        if verbose:
            print(" ... calculating Lindhard quantities...\n")
        
        # Calculate the energy function
        epsilon = 11.5*ke*(Z**(-7/3))
        lamb = 3*(epsilon**.15) + .7*(epsilon**.6) + epsilon
        # Choose the isotope
        isotope = rand(*ke.shape)*100
        A = 136.*np.ones_like(ke)
        A[(isotope>= 0.00)&(isotope< 0.09)] = 124.
        A[(isotope>= 0.09)&(isotope< 0.18)] = 126.
        A[(isotope>= 0.18)&(isotope< 2.10)] = 128.
        A[(isotope>= 2.10)&(isotope<28.54)] = 129.
        A[(isotope>=28.54)&(isotope<32.62)] = 130.
        A[(isotope>=32.62)&(isotope<53.80)] = 131.
        A[(isotope>=53.80)&(isotope<80.69)] = 132.
        A[(isotope>=80.69)&(isotope<91.13)] = 134.
        
        # Now put them together to get the Lindhard parameter
        LindkFactor = kappa*np.sqrt(A/molarMass)
        L = (LindkFactor*lamb)/(1+LindkFactor*lamb)
        L[L>1.] = 1.
        L[L<0.] = 0.
    # --------------- </LINDHARD> --------------- #
    
    # --------------- <NexNi> --------------- #
    if verbose:
        print(" ... Calculating Nex/Ni ...\n")
    if (pt=="NR") or (pt=="DD"):
        NexONi = 1.24 * (ef**-.0472) * (1-np.exp(-239.*epsilon))
    elif (pt=="ER") or (pt=="CH3T"):
        NexONi = 0.059813 + 0.031228*Xe_rho
    alf = 1 / (1 + NexONi)
    # --------------- </NexNi> --------------- #
    
    # --------------- <THOMAS-IMEL RECOMB.> --------------- #
    if verbose:
        print(" ... Calculating recombination via Thomas-Imel ...\n")
    if (pt=="NR") or (pt=="DD"):
        TIB      = (0.0554 * (ef**-.062)) * np.ones_like(ke)
        tibCurlZ = 0.
    elif (pt=="ER") or (pt=="CH3T"):
        L        = 2./3.
        tibMain  = 0.6347 * np.exp(-.00014*ef)
        tibEdep  = -0.373*np.exp(-ef*0.001/tibMain) + 1.5
        tibCurlA = 10.*(ef**-.04)*np.exp(18./ef)
        tibCurlB = 0.188*(ef**(1./3.))
        tibCurlZ = 1. - (ef**0.2147) + 3.
        TIB      = np.zeros_like(ke)
        c_ke     = ke >= tibCurlZ # regulate ke>tibCurlZ is positive, otherwise TIB gives NaN
        TIB[c_ke]= tibMain[c_ke]*(ke[c_ke]**(-tibEdep[c_ke])) * \
                   (1.-np.exp(-(((ke[c_ke]-tibCurlZ[c_ke])/tibCurlA[c_ke])**tibCurlB[c_ke])))
        #TIB      = tibMain*(ke**(-tibEdep))*(1.-np.exp(-(((ke-tibCurlZ)/tibCurlA)**tibCurlB)))
    TIB = TIB * (Xe_rho / 2.888)**0.3
    # --------------- </THOMAS-IMEL RECOMB.> --------------- #
    
    # --------------- <FLUCUATE INITIAL QUANTA> --------------- #
    if verbose:
        print(" ... Fluctuating initial quanta ...\n")
    if (pt=="NR") or (pt=="DD"):
        Ni  = modPoisRnd((L*alf*ke)/W, Fi)
        Nex = modPoisRnd((L*NexONi*alf*ke)/W, Fex)
        Nq  = Nex + Ni
    elif (pt=="ER") or (pt=="CH3T"):
        Nq  = binomFluct(st.poisson.rvs(ke/(L*W)), L)
        Ni  = binomFluct(Nq, alf)
    # --------------- </FLUCUATE INITIAL QUANTA> --------------- #
    
    # --------------- <RECOMBINATION FLUCTUATIONS> --------------- #
    if verbose:
        print(" ... recombination fluctuations ...\n")
    R              = np.zeros_like(ke)
    R[TIB>0]       = 1. - np.log(1. + (Ni[TIB>0]/4.) * TIB[TIB>0]) / ((Ni[TIB>0]/4.)*TIB[TIB>0])
    R[R>1.]        = 1.
    R[R<0.]        = 0.
    R[ke<tibCurlZ] = 0.
    
    Fr = 0.008 * Ni
    Fr[Fr<1e-6] = 1e-6
    
    Ne = modPoisRnd(Ni * (1.-R), Fr)
    Ne[Ne>Nq] = Nq[Ne>Nq]
    
    Nph = Nq - Ne
    Ne[Ne<0.] = 0.
    Nph[Nph<0.] = 0.
    Nph[Nph>Nq] = Nq[Nph>Nq]
    if (pt=="NR") or (pt=="DD"):
        #Nph = fastbinornd(Nph, 1./(1.+3.32*epsilon**1.141))
        Nph = st.binom.rvs(np.int64(Nph), 1./(1.+3.32*epsilon**1.141), size=len(Nph))
    # --------------- </RECOMBINATION FLUCTUATIONS> --------------- #
    
    return Nph, Ne

def S1S2RawFromNphNe(Nph, Ne, r_real, z_real, verbose=False, g1=.11, sp=.37, ee=0.54, g2=.0831/vuvFudgeFactor, 
    ff=2.5, df=181., af=6000., dt_min=38., dt_max=305., dtCntr=160.,gasRho=15e-3):
    """
    This function takes in a number of photons and electrons as input and spits out the 
    UNCORRECTED (raw) S1 and S2 values.
    """
    # --------------- <S1 YIELD> --------------- #
    # Using the values for run3 by default, with a kludge for run4 if requested in input S1_dt_dep
    if verbose:
        print(" ... Assigning Npe values ...\n")    
    dt_run3 = rawCoords2measuredCoords(r_real, z_real, fmap='run3')[1]
    S1pol =  [0.84637 , 0.0016222 , -9.9224e-6 , 4.7508e-8 , -7.1692e-11]  #  the run3 S1(dt) map
    posDep = S1pol[0]+S1pol[1]*dt_run3+S1pol[2]*dt_run3**2+S1pol[3]*dt_run3**3+S1pol[4]*dt_run3**4
    # normalize posDep to be 1 at dtCntr
    posDep = posDep/(S1pol[0]+S1pol[1]*dtCntr+S1pol[2]*dtCntr**2+S1pol[3]*dtCntr**3+S1pol[4]*dtCntr**4)
    
    # this variation step takes a loooong time.  Heaviest line of code in this script.
    #Npe = fastbinornd(Nph, g1*posDep)
    Npe = st.binom.rvs(np.int64(Nph), g1*posDep, size=len(Nph))
    # --------------- </S1 YIELD> --------------- #
    
    # --------------- <S1 COINCIDENCE REQUIREMENT> --------------- #
    if verbose:
        print(" ... S1 coincidence requirement ...\n")
    if (nFold > 0.):
        S1 = np.zeros_like(Nph)
        nHits = np.zeros_like(Nph)
        
        # loop over all events (this is slow... must be a better way!?!?)
        for ievt in range(len(Nph)):
            phe1_areas = rGaus(np.ones(Npe[ievt]), sp*np.ones(Npe[ievt]))
            phe2_areas = rGaus(np.ones(Npe[ievt]), sp*np.ones(Npe[ievt]))
            phe2_happens = rand(Npe[ievt]) < P_dphe
            phe2_areas[~phe2_happens] = 0.
            tot_areas = phe1_areas + phe2_areas
            S1[ievt] = tot_areas[tot_areas>spEff].sum()
            
            # account for the nFold requirement (greater than or equal to nFold PMTs must have a hit)
            ii = 0.
            while (ii<Npe[ievt]) and (nHits[ievt]<nFold):
                if (tot_areas[int(ii)] > spEff) and (rand() < ((numberPMTs-1.)/numberPMTs)**nHits[ievt]):
                    nHits[ievt] = nHits[ievt] + 1.
                ii += 1.
        S1 = S1 / (1. + P_dphe)
        cutNfold = nHits >= nFold
        S1[nHits<nFold] = 0.
    else: # if nFold == 0
        #Npe = Npe + fastbinornd(Npe, P_dphe)
        Npe = Npe + st.binom.rvs(np.int64(Npe), P_dphe,size=len(Npe))
        S1 = rGaus(Npe, sp*np.sqrt(Npe)) / (1. + P_dphe)
        sub_threshold = rand(*Nph.shape) > 0.5*erf((S1/posDep-erf_mu)/(erf_sg*np.sqrt(2.))) + 0.5
        S1[sub_threshold] = 0.
    """  HERE HERE HERE, IS WHERE I CANCEL THE S1 CORRECTION!!!! 
    S1c = S1 / posDep
    """
    # --------------- </S1 COINCIDENCE REQUIREMENT> --------------- #
    
    # --------------- <S2 EXTRACTION EFFICIENCY> --------------- #
    if verbose:
        print(" ... Accounting for S2 extraction efficiency ...\n")
    if af != 6000.:
        af = af / 1e3
        LiqExtField = af / (1.96/1.00126)
        ExtractEffA = -0.0012052 + (0.1638 * af) - (0.0063782 * (af**2)) # Aprile 2004 IEEE No. 5
        ExtractEffB = 1. - 1.3558*np.exp(-0.074312 * LiqExtField**2.4259) # Gushchin
        ee = ExtractEffB
        if af >= 12.:
            ee = 1.
        if (af <= 0.) or (LiqExtField < 0):
            ee = 0.
        if ee < 0.:
            ee = 0.
        if ee > 1.:
            ee = 1.
        af = af * 1e3
    """
    MUST DECOUPLE THE RUN-3 DT USED FOR S1 CORRECTIONS FROM THE REAL DT THAT WILL
    BE USED FOR S2 CORRECTIONS.
    """
    dt_run4 = rawCoords2measuredCoords(r_real, z_real, fmap='run4')[1]
    gasDensity = (gasRho / molarMass) * AVO
    nphPerElec = (elGain * (af / gasDensity) * (1e17)-elThld) * gasDensity * (1e-17) * gasGap
    #Nee = fastbinornd(Ne, ee*np.exp(-dt_run4/e_life)) # losses due to e-lifetime are done here
    Nee = st.binom.rvs(np.int64(Ne), ee*np.exp(-dt_run4/e_life),size=len(Ne)) # losses due to e-lifetime are done here
    ElecLumYld = np.round(np.floor(rGaus(nphPerElec*Nee,ff*np.sqrt(nphPerElec*Nee)) + 0.5))
    Npe = binomFluct(ElecLumYld, g2)
    Npe = Npe + binomFluct(Npe, P_dphe)
    S2t = rGaus(Npe, sp*np.sqrt(Npe)) / (1+P_dphe)
    S2b = rGaus(S2_rat*S2t, np.sqrt(S2_rat*S2t*(1.-S2_rat)))
    """  HERE HERE HERE, IS WHERE I CANCEL THE S2 CORRECTION!!!! 
    S2bc = S2b / np.exp(-dt/e_life)
    S2tc = S2t / np.exp(-dt/e_life)
    """
    # --------------- </S2 EXTRACTION EFFICIENCY> --------------- #
    
    if verbose:
        print(" ...  done!")
        print("{:s} Finished with PyNEST {:s}".format(
            "<"*np.int64((terminalWidth-22)/2),
            ">"*(terminalWidth-22-np.int64((terminalWidth-22)/2))))
    
    return S1, S2t, S2b, cutNfold

def pyNEST(ke, r_real, z_real, pt,verbose=False, g1=.11, sp=.37, ee=0.54, fmap='run4', g2=.0831/vuvFudgeFactor, 
    ff=2.5, df=181., af=6000., dt_min=38., dt_max=305., dtCntr=160., gasRho=15e-3):
    """
    This is function that puts all the other stuff together.
    Inputs: -----------------
                 ke: Energy of the recoils (numpy ndarray)
             r_real: The real radii of the events (in cm).
             z_real: The real z-positions of the events (in cm)
                 pt: The type of recoil, can be "NR" or "ER"
    Outputs: ----------------
      **NOTE: the following are the default outputs, if no 'outs' kwarg is passed.  On the
              other hand, input 'outs' can be used to select which variables the function will
              output.  'outs' must be a list of strings that give the variable names as they
              appear in the code of this function. **
              
             S1c_xyz: The corrected S1 value (in dph).
            S2tc_xyz: The corrected S2 total value (in dph)
            S2bc_xyz: The corrected S2 bottom value (in dph)
              S1_raw: The raw S1 size (before position correction), in dph.
             S2t_raw: The raw S2-total size in dph.
             S2b_raw: The raw S2-bottom size in dph.
       r_real_recons: The reconstructed real R coordinate of the events (in cm).
       z_real_recons: The reconstructed real Z coordinate of the events (in cm).
      r_surface_meas: The raw radius, i.e. the radius of the e-cloud as it exits the surface (in cm).
                  dt: The drift time, in microseconds.
                 Nph: The raw number of photons emitted.
                  Ne: The raw number of electrons emitted. 
            cutNfold: Boolean array, True for events passing the S1 nFold requirement.
    
    2014.12.01 -- A. Manalaysay
    """
    # Get field values
    ef = getFieldFromTrueCoords(r_real, z_real, fmap=fmap)
    
    # Get Nph and Ne
    Nph, Ne = Nph_Ne(pt, ef, ke)
    
    # Get the raw measured coordinates (r_surface_raw and dt).  They are "raw" because they have
    # yet to be smeared due to measurement resolution. Uncertainty is dominated by r, so we will
    # not distinguish between raw and fluctuated dt.
    r_surface_raw, dt = rawCoords2measuredCoords(r_real, z_real, fmap=fmap)
    
    # Get the uncorrected S1 and S2 values
    #S1_raw, S2t_raw, S2b_raw = S1S2RawFromNphNe(Nph, Ne, r_real, z_real)
    S1_raw, S2t_raw, S2b_raw, cutNfold = S1S2RawFromNphNe(Nph, Ne, r_real, z_real,verbose=verbose, g1=g1, 
            sp=sp, ee=ee, g2=g2, ff=ff, df=df, af=af, dt_min=dt_min, dt_max=dt_max, 
            dtCntr=dtCntr, gasRho=gasRho)
    
    # Smear the measured R at the surface based on the resolution vs. S2
    cutS2gt1 = S2t_raw > 1.
    r_res = 17./np.sqrt(S2t_raw)  # parametrization from Claudio's resolution plot
    r_res[~cutS2gt1] = 1e-6
    x_smear = r_surface_raw + (r_res/np.sqrt(2))*randn(*S2t_raw.shape)
    y_smear = (r_res/np.sqrt(2))*randn(*S2t_raw.shape)
    r_surface_meas = np.sqrt((x_smear**2) + (y_smear**2))
    
    # Correct the S2 values for the measured drift time
    #S2bc = S2b_raw / np.exp(-dt/e_life)
    #S2tc = S2t_raw / np.exp(-dt/e_life)
    S2bc_z = S2b_raw * np.exp(dt/e_life)
    S2tc_z = S2t_raw * np.exp(dt/e_life)
    S2_xyzCorrFactor = ip.griddata(np.vstack([KrypcalC['s2_xy_xbinsPTS'],KrypcalC['s2_xy_ybinsPTS']]).T,
               KrypcalC['norm_S2_xyPTS'],np.vstack([x_smear,y_smear]).T,method='cubic')
    S2bc_xyz = S2bc_z * S2_xyzCorrFactor
    S2tc_xyz = S2tc_z * S2_xyzCorrFactor
    S1c_z = S1_raw * np.polyval(KrypcalC['mean_fit_s1z'],(det_edge-4*(det_edge/324))/2) / \
            np.polyval(KrypcalC['mean_fit_s1z'],dt)
    S1c_xyz = S1c_z*ip.griddata(np.vstack([KrypcalC['s1_xy_xbinsPTS'],KrypcalC['s1_xy_ybinsPTS']]).T,
              KrypcalC['norm_S1PTS'], np.vstack([x_smear,y_smear]).T,method='cubic')
    
    # Reconstruct the real R and Z based on the measured R (raw) and dt: 
    r_real_recons, z_real_recons = measuredCoords2trueCoords(r_surface_meas, dt, fmap=fmap)
    """
    # Get the equivalent dt for run3 (needed for the S1 correction)
    dt_run3_recons = rawCoords2measuredCoords(r_real_recons, z_real_recons, fmap='run3')[1]
    
    # Calculate the S1 correction factor from the run3 dt.
    S1pol =  [0.84637 , 0.0016222 , -9.9224e-6 , 4.7508e-8 , -7.1692e-11]  #  the run3 S1(dt) map
    posDep = S1pol[0]+S1pol[1]*dt_run3_recons + S1pol[2]*dt_run3_recons**2 + \
             S1pol[3]*dt_run3_recons**3 + S1pol[4]*dt_run3_recons**4
    posDep = posDep/(S1pol[0]+S1pol[1]*dtCntr+S1pol[2]*dtCntr**2+S1pol[3]*dtCntr**3+S1pol[4]*dtCntr**4)
    S1c = S1_raw / posDep
    """
    return S1c_xyz, S2tc_xyz, S2bc_xyz, S1_raw, S2t_raw, S2b_raw, r_real_recons, z_real_recons, r_surface_meas, dt, Nph, Ne, cutNfold

def pythonNEST_MESS(pt, arg2, verbose=False, g1=.11, sp=.37, ee=0.54, g2=.0831/vuvFudgeFactor, 
    ff=2.5, df=181., af=6000., dt_min=38., dt_max=305., dtCntr=160., S1_dt_dep='run3', field_dt_dep_bool=False):
    """
    Fun info
    """
    # --------------- <INPUT HANDLING> --------------- #
    # --------------- </INPUT HANDLING> --------------- #
    
    # --------------- <HANDLE ARG2> --------------- #
    # print info
    if verbose:
        terminalWidth = eval(os.popen('stty size','r').read().split()[1])
        print("\n")
        print("{:s} Starting PyNEST {:s}".format(
            "<"*np.int64((terminalWidth-17)/2),
            ">"*(terminalWidth-17-np.int64((terminalWidth-17)/2))))
        print(" ... assigning energies...\n")
    
    if (pt == "NR") or (pt == "ER"):
        ke = arg2;
    elif (pt == "CH3T"):
        ke = np.zeros(arg2)
        QQ = 18.5898
        ZZ =2.
        xmax = QQ
        ymax = 1.1e7
        # Generate a random sampling of energies from a tritium spectrum
        for ievent in range(arg2):
            xTry = xmax*rand()
            yTry = ymax*rand()
            #tryTest = np.sqrt( (xTry**2) + 2*xTry*m_e) * (xTry + m_e) * ((QQ-xTry)**2) *
            #          (1+pi*aa*ZZ*(xTry+m_e)/np.sqrt((xTry**2)+2*xTry*m_e))
            tryTest = _tritSpectrum(xTry)
            while (yTry > tryTest):
                xTry = xmax*rand()
                yTry = ymax*rand()
                tryTest = _tritSpectrum(xTry)
            ke[ievent] = xTry
        
    elif (pt == "DD"):
        ke = VonNeumann(0., 100., 0., 5.3, arg2)
    # --------------- </HANDLE ARG2> --------------- #
    
    # --------------- <ASSIGN DT VALUES> --------------- #
    if verbose:
        print(" ... assigning dt values...\n")
    dt = dt_min + (dt_max - dt_min) * rand(*ke.shape)
    # --------------- </ASSIGN DT VALUES> --------------- #
    
    # --------------- <ASSIGN FIELD VALUES> --------------- #
    if verbose:
        print(" ... assigning field values...\n")
    dt = dt_min + (dt_max - dt_min) * rand(*ke.shape)
    ef = df * np.ones_like(dt)
    print("!!!!!!!!NO COMPLICATED FIELDS YET!!!!!!!!")
    # --------------- </ASSIGN FIELD VALUES> --------------- #
    
    # --------------- <LINDHARD> --------------- #
    if (pt=="NR") or (pt=="DD"):   # Lindhard only for nuclear recoils
        if verbose:
            print(" ... calculating Lindhard quantities...\n")
        
        # Calculate the energy function
        epsilon = 11.5*ke*(Z**(-7/3))
        lamb = 3*(epsilon**.15) + .7*(epsilon**.6) + epsilon
        # Choose the isotope
        isotope = rand(*ke.shape)*100
        A = 136.*np.ones_like(ke)
        A[(isotope>= 0.00)&(isotope< 0.09)] = 124.
        A[(isotope>= 0.09)&(isotope< 0.18)] = 126.
        A[(isotope>= 0.18)&(isotope< 2.10)] = 128.
        A[(isotope>= 2.10)&(isotope<28.54)] = 129.
        A[(isotope>=28.54)&(isotope<32.62)] = 130.
        A[(isotope>=32.62)&(isotope<53.80)] = 131.
        A[(isotope>=53.80)&(isotope<80.69)] = 132.
        A[(isotope>=80.69)&(isotope<91.13)] = 134.
        
        # Now put them together to get the Lindhard parameter
        LindkFactor = kappa*np.sqrt(A/molarMass)
        L = (LindkFactor*lamb)/(1+LindkFactor*lamb)
        L[L>1.] = 1.
        L[L<0.] = 0.
    # --------------- </LINDHARD> --------------- #
    
    # --------------- <NexNi> --------------- #
    if verbose:
        print(" ... Calculating Nex/Ni ...\n")
    if (pt=="NR") or (pt=="DD"):
        NexONi = 1.24 * (ef**-.0472) * (1-np.exp(-239.*epsilon))
    elif (pt=="ER") or (pt=="CH3T"):
        NexONi = 0.059813 + 0.031228*Xe_rho
    alf = 1 / (1 + NexONi)
    # --------------- </NexNi> --------------- #
    
    # --------------- <THOMAS-IMEL RECOMB.> --------------- #
    if verbose:
        print(" ... Calculating recombination via Thomas-Imel ...\n")
    if (pt=="NR") or (pt=="DD"):
        TIB      = (0.0554 * (ef**-.062)) * np.ones_like(ke)
        tibCurlZ = 0.
    elif (pt=="ER") or (pt=="CH3T"):
        L        = 2./3.
        tibMain  = 0.6347 * np.exp(-.00014*ef)
        tibEdep  = -0.373*np.exp(-ef*0.001/tibMain) + 1.5
        tibCurlA = 10.*(ef**-.04)*np.exp(18./ef)
        tibCurlB = 0.188*(ef**(1./3.))
        tibCurlZ = 1. - (ef**0.2147) + 3.
        TIB      = np.zeros_like(ke)
        c_ke     = ke >= tibCurlZ # regulate ke>tibCurlZ is positive, otherwise TIB gives NaN
        TIB[c_ke]= tibMain[c_ke]*(ke[c_ke]**(-tibEdep[c_ke])) * \
                   (1.-np.exp(-(((ke[c_ke]-tibCurlZ[c_ke])/tibCurlA[c_ke])**tibCurlB[c_ke])))
        #TIB      = tibMain*(ke**(-tibEdep))*(1.-np.exp(-(((ke-tibCurlZ)/tibCurlA)**tibCurlB)))
    TIB = TIB * (Xe_rho / 2.888)**0.3
    # --------------- </THOMAS-IMEL RECOMB.> --------------- #
    
    # --------------- <FLUCUATE INITIAL QUANTA> --------------- #
    if verbose:
        print(" ... Fluctuating initial quanta ...\n")
    if (pt=="NR") or (pt=="DD"):
        Ni  = modPoisRnd((L*alf*ke)/W, Fi)
        Nex = modPoisRnd((L*NexONi*alf*ke)/W, Fex)
        Nq  = Nex + Ni
    elif (pt=="ER") or (pt=="CH3T"):
        Nq  = binomFluct(st.poisson.rvs(ke/(L*W)), L)
        Ni  = binomFluct(Nq, alf)
    # --------------- </FLUCUATE INITIAL QUANTA> --------------- #
    
    # --------------- <RECOMBINATION FLUCTUATIONS> --------------- #
    if verbose:
        print(" ... recombination fluctuations ...\n")
    R              = np.zeros_like(ke)
    R[TIB>0]       = 1. - np.log(1. + (Ni[TIB>0]/4.) * TIB[TIB>0]) / ((Ni[TIB>0]/4.)*TIB[TIB>0])
    R[R>1.]        = 1.
    R[R<0.]        = 0.
    R[ke<tibCurlZ] = 0.
    
    Fr = 0.008 * Ni
    Fr[Fr<1e-6] = 1e-6
    
    Ne = modPoisRnd(Ni * (1.-R), Fr)
    Ne[Ne>Nq] = Nq[Ne>Nq]
    
    Nph = Nq - Ne
    Ne[Ne<0.] = 0.
    Nph[Nph<0.] = 0.
    Nph[Nph>Nq] = Nq[Nph>Nq]
    if (pt=="NR") or (pt=="DD"):
        Nph = fastbinornd(Nph, 1./(1.+3.32*epsilon**1.141))
    # --------------- </RECOMBINATION FLUCTUATIONS> --------------- #
    
    # --------------- <S1 YIELD> --------------- #
    # Using the values for run3 by default, with a kludge for run4 if requested in input S1_dt_dep
    if verbose:
        print(" ... Assigning Npe values ...\n")
    # we're going to assume the run3 S1(dt) correction holds for run4, with a
    # mapping (from the field model) between dt for run4 to dt for run3
    if S1_dt_dep=='run3':
        dtmapping = [0., 1. , 0.]
    elif S1_dt_dep=='run4_lowR':
        dtmapping = [-0.00037775 , 1.1196 , 0]
    elif S1_dt_dep=='run4_highR':
        dtmapping = [-0.00063668 , 1.1239 , 0]
    dt_scaled_toRun3     = np.polyval(dtmapping , dt); 
    dtCntr_scaled_toRun3 = np.polyval(dtmapping , dtCntr);
    S1pol =  [0.84637 , 0.0016222 , -9.9224e-6 , 4.7508e-8 , -7.1692e-11]  #  the run3 S1(dt) map
    posDep = S1pol[0] + S1pol[1]*dt_scaled_toRun3 + S1pol[2]*dt_scaled_toRun3**2 + \
             S1pol[3]*dt_scaled_toRun3**3 + S1pol[4]*dt_scaled_toRun3**4
    # normalize posDep to be 1 at dtCntr
    posDep = posDep /( S1pol[0] + S1pol[1]*dtCntr_scaled_toRun3 + S1pol[2]*dtCntr_scaled_toRun3**2 + \
             S1pol[3]*dtCntr_scaled_toRun3**3 + S1pol[4]*dtCntr_scaled_toRun3**4)
    
    # this variation step takes a loooong time.  Heaviest line of code in this script.
    Npe = fastbinornd(Nph, g1*posDep)
    # --------------- </S1 YIELD> --------------- #
    
    # --------------- <S1 COINCIDENCE REQUIREMENT> --------------- #
    if verbose:
        print(" ... S1 coincidence requirement ...\n")
    if (nFold > 0.):
        S1 = np.zeros_like(ke)
        nHits = np.zeros_like(ke)
        
        # loop over all events (this is slow... must be a better way!?!?)
        for ievt in range(len(ke)):
            phe1_areas = rGaus(np.ones(Npe[ievt]), sp*np.ones(Npe[ievt]))
            phe2_areas = rGaus(np.ones(Npe[ievt]), sp*np.ones(Npe[ievt]))
            phe2_happens = rand(Npe[ievt]) < P_dphe
            phe2_areas[~phe2_happens] = 0.
            tot_areas = phe1_areas + phe2_areas
            S1[ievt] = tot_areas[tot_areas>spEff].sum()
            
            # account for the nFold requirement (greater than or equal to nFold PMTs must have a hit)
            ii = 0.
            while (ii<Npe[ievt]) and (nHits[ievt]<nFold):
                if (tot_areas[ii] > spEff) and (rand() < ((numberPMTs-1.)/numberPMTs)**nHits[ievt]):
                    nHits[ievt] = nHits[ievt] + 1.
                ii += 1.
        S1 = S1 / (1. + P_dphe)
        S1[nHits<nFold] = 0.
    else: # if nFold == 0
        Npe = Npe + fastbinornd(Npe, P_dphe)
        S1 = rGaus(Npe, sp*np.sqrt(Npe)) / (1. + P_dphe)
        sub_threshold = rand(*ke.shape) > 0.5*erf((S1/posDep-erf_mu)/(erf_sg*np.sqrt(2.))) + 0.5
        S1[sub_threshold] = 0.
    S1c = S1 / posDep
    # --------------- </S1 COINCIDENCE REQUIREMENT> --------------- #
    
    # --------------- <S2 EXTRACTION EFFICIENCY> --------------- #
    if verbose:
        print(" ... Accounting for S2 extraction efficiency ...\n")
    if af != 6000.:
        af = af / 1e3
        LiqExtField = af / (1.96/1.00126)
        ExtractEffA = -0.0012052 + (0.1638 * af) - (0.0063782 * (af**2)) # Aprile 2004 IEEE No. 5
        ExtractEffB = 1. - 1.3558*np.exp(-0.074312 * LiqExtField**2.4259) # Gushchin
        ee = ExtractEffB
        if af >= 12.:
            ee = 1.
        if (af <= 0.) or (LiqExtField < 0):
            ee = 0.
        if ee < 0.:
            ee = 0.
        if ee > 1.:
            ee = 1.
        af = af * 1e3
    
    gasDensity = (gasRho / molarMass) * AVO
    nphPerElec = (elGain * (af / gasDensity) * (1e17)-elThld) * gasDensity * (1e-17) * gasGap
    Nee = fastbinornd(Ne, ee*np.exp(-dt/e_life))
    ElecLumYld = np.round(np.floor(rGaus(nphPerElec*Nee,ff*np.sqrt(nphPerElec*Nee)) + 0.5))
    Npe = binomFluct(ElecLumYld, g2)
    Npe = Npe + binomFluct(Npe, P_dphe)
    S2t = rGaus(Npe, sp*np.sqrt(Npe)) / (1+P_dphe)
    S2b = rGaus(S2_rat*S2t, np.sqrt(S2_rat*S2t*(1.-S2_rat)))
    S2bc = S2b / np.exp(-dt/e_life)
    S2tc = S2t / np.exp(-dt/e_life)
    # --------------- </S2 EXTRACTION EFFICIENCY> --------------- #
    
    if verbose:
        print(" ...  done!")
        print("{:s} Finished with PyNEST {:s}".format(
            "<"*np.int64((terminalWidth-22)/2),
            ">"*(terminalWidth-22-np.int64((terminalWidth-22)/2))))
    
    return S1c, S2tc, S2bc, Nph, Ne, ke














