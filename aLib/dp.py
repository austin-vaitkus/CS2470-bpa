"""
dp.py -- a module for helpful operations for analysis of the LUX RQ. Under construction, permanently.

Methods contained herein:
    evtWindow
    mask_S1inWindow
    mask_S2inWindow
    mask_SMinWindow
    NS2_InWindow
    NS2_InW_thr
    NS1_InWindow
    NSM_InWindow
    Nth_S1_Window
    Nth_S2_Window
    NthLargestS2
    Nth_SM_Window
    concatRQsWcuts

2014.02.26 -- A. Manalaysay
2014.03.24 -- Updated.
2014.03.27 -- Updated
"""
from __future__ import print_function
import numpy as np
import os
import aLib

tDrift0 = 35000.
tEv0 = 5e5


def evtWindow(d, tDrift=tDrift0, tEv=tEv0):
    """
    Return the limits in time of the event, within which one can consider S1 and S2 pulses as being
    part of that event.  This works on processed RQs.  This function was created to address the 
    differences in the way that the main-chain and alternate-chain DP methods chose which ten pulses
    to include in the RQ files.  The algorithm is as follows:
        * Find the largest S2 pulse (raw area)
        * Look in the time window 'tDrift' before the largest S2 and identify any S1 pulses in this
          window.  All other S1 pulses in the event record are ignored.
        * The beginning of the event window is defined by 'aft_t0_samples' of the first S1 pulse
          within the time 'tDrift' before the largest S2.
        * The end of the event window is defined by 'aft_t1_samples' of the last S1 plus 'tDrift'.
        * If no S1 pulses are found, the event window is defined as being 'tDrift' before the 
          largest S2 until 'tDrift' after the largest S2.
        * If no S2 pulses are found, the event window is defined as tWidth before and after the 
          largest S1 pulse.
        * If no S2 or S1 pulses are found, the event window is defined as 'tDrift' before the last 
          pulse until 'tDrift' after (in other words, it basically treats the last found pulse as 
          if it was the largest S2 with no S1 pulses around).
    
    ------Inputs:
           d: Dictionary of LUX data.  Must include keys 'pulse_classification', 'pulse_area_phe',
              'aft_t1_samples', and 'aft_t0_samples'
      tDrift: The maximum drift time for an event (i.e. dT of an event at the cathode... be sure
              to add a bit of time to this to be safe).
         tEv: The maximum possible time window covered by an event.
    -----Outputs:
     tBounds: A 2xN numpy array. N is the number of events in 'd'. The first row is the beginning 
              time of the event, the second row is the end time of the event.
    
    2014.03.14 -- A. Manalaysay
    """
    # Initialize dictionary to hold cuts (mainly for diagnostic output)
    c = aLib.S()
    
    # Figure out the dimensions of the data fields in the RQs. The default is 10 pulses, but
    # I don't want to hard-code that in. I also want to know the number of events.
    numPulsesPerEvt    = d['pulse_classification'].shape[0]
    numEvts            = d['pulse_classification'].shape[1]
    
    # Initialize the array that will contain the time-window boundaries
    tBounds = np.zeros([2, numEvts])
    
    # Figure out where the S1 and S2 pulses are in the whole event record (or, the ones that
    # were deemed worthy of being included in the 10 returned pulses).
    c.cutS1all         = d['pulse_classification'] == 1
    c.cutS2all         = d['pulse_classification'] == 2
    
    # It could be that no S2 pulses were included/found. Need to identify if/where that happens
    # so that we will be able to handle that specific case without choking.
    c.cutHasS2         = c.cutS2all.sum(0) > 0
    
    # Situations where S1 pulses are found, but no S2:
    c.cutS1NoS2        = (c.cutS1all.sum(0)>0) & (c.cutS2all.sum(0)==0)
    
    # Make a boolean array that identifies, in each event, where the largest S2 pulse is.
    S2areaRawLargest   = (c.cutS2all * d['pulse_area_phe']).max(0)
    c.cutS2Largest     = (d['pulse_area_phe'] == np.tile(S2areaRawLargest,[numPulsesPerEvt,1])) & \
                         c.cutS2all
    
    tCut_pcNEQzero     = d['pulse_classification'] != 0
    tCut_pcNEQzero_cm  = tCut_pcNEQzero.cumsum(0)
    c.cutLastPulse     = (tCut_pcNEQzero_cm == tCut_pcNEQzero_cm.max(0)) & tCut_pcNEQzero
    
    # For events w/o S2, set cutS2Largest to be the last pulse
    c.cutS2Largest[:,~c.cutHasS2] = c.cutLastPulse[:,~c.cutHasS2]
    
    # Adding and subtracting tEv because the aft times might span zero, in which case taking
    # the max value would not necessarily yield the last value.
    t_LargestS2 = ((d['aft_t1_samples']+tEv)*c.cutS2Largest).max(0) - tEv
    c.cutS1_tDriftBEFs2Largest = \
                ((d['aft_t1_samples'] < np.tile(t_LargestS2,[numPulsesPerEvt,1])) \
              &  (d['aft_t1_samples'] > (np.tile(t_LargestS2-tDrift,[numPulsesPerEvt,1])))) \
              & c.cutS1all
                #(((d['aft_t1_samples']-1e5) < np.tile(t_LargestS2,[numPulsesPerEvt,1])) \
              #&  ((d['aft_t1_samples']-1e5) > (np.tile(t_LargestS2-tDrift,[numPulsesPerEvt,1])))) \
              #& c.cutS1all
    
    # Determine number of S1s found in the tDrift window before the largest S2 and flag
    # events that have none
    numS1inPreWindow     = c.cutS1_tDriftBEFs2Largest.sum(0)
    c.cutNoS1inPreWindow = numS1inPreWindow == 0
    
    # Nail the first and last of the found S1 pulses in the window
    c.cutS1_tDriftBEFs2Largest_cm = c.cutS1_tDriftBEFs2Largest.cumsum(0)
    c.cut1stS1inPreWindow = (c.cutS1_tDriftBEFs2Largest_cm == 1) & c.cutS1_tDriftBEFs2Largest
    c.cutLstS1inPreWindow = (c.cutS1_tDriftBEFs2Largest_cm == \
            np.tile(c.cutS1_tDriftBEFs2Largest_cm.max(0),[numPulsesPerEvt,1])) \
            & c.cutS1_tDriftBEFs2Largest
    
    # Set the first tBound based on aft_t0_samples of the first S1
    tBounds[0,:] = ((d['aft_t0_samples']+tEv)*c.cut1stS1inPreWindow).max(0) - tEv
    # Set the last tBound based on aft_t1_samples of the last S1 + tDrift
    tBounds[1,:] = ((d['aft_t1_samples']+tEv)*c.cutLstS1inPreWindow).max(0) - tEv + tDrift
    
    
    # Handle the case where no S1 pulses were found
    tBounds[0,c.cutNoS1inPreWindow] = \
        (d['aft_t1_samples'][:,c.cutNoS1inPreWindow] * \
        c.cutS2Largest[:,c.cutNoS1inPreWindow] + tEv).max(0) - tEv - tDrift
    tBounds[1,c.cutNoS1inPreWindow] = \
        (d['aft_t1_samples'][:,c.cutNoS1inPreWindow] * \
        c.cutS2Largest[:,c.cutNoS1inPreWindow] + tEv).max(0) - tEv + tDrift
    
    # Handle the case where no S2 pulses were found
    tBounds[0,~c.cutHasS2] = (d['aft_t0_samples'][:,~c.cutHasS2] * \
        c.cutLastPulse[:,~c.cutHasS2]+tEv).max(0) - tEv - tDrift
    tBounds[1,~c.cutHasS2] = (d['aft_t1_samples'][:,~c.cutHasS2] * \
        c.cutLastPulse[:,~c.cutHasS2]+tEv).max(0) - tEv + tDrift
    
    # Handle the case where no S2 pulse was found, but S1 pulses exist
    S1areaRawLargest = (c.cutS1all*d['pulse_area_phe']).max(0)
    c.cutS1Largest = (d['pulse_area_phe'] == np.tile(S1areaRawLargest,[numPulsesPerEvt,1])) & \
                    c.cutS1all
    tBounds[0,c.cutS1NoS2] = ((d['aft_t1_samples']+tEv)*c.cutS1Largest).max(0)[c.cutS1NoS2] - tEv - tDrift
    tBounds[1,c.cutS1NoS2] = ((d['aft_t1_samples']+tEv)*c.cutS1Largest).max(0)[c.cutS1NoS2] - tEv + tDrift
    
    #return tBounds, c
    return tBounds

def mask_S2inWindow(d, tDrift=tDrift0, tEv=tEv0, forceWinRecalc=False):
    """
    Return a boolean mask that picks out the S2 pulses in the event window.
    ------Inputs:
                 d: Python dictionary containing LUX RQs.
            tDrift: (Optional) kw var for use in evtWindow (see corresponding documentation).
               tEv: (Optional) kw var for use in evtWindow (see corresponding documentation).
    forceWinRecalc: (Optional) Force recalculation of the time window, if it has already been 
                    calculated. Default: False
    -----Outputs:
         maskS2InW: A NxM boolean array, where N is the number of pulses per event in an RQ file, 
                    and M is the number of events in 'd'. Array is True for S2 pulses in the event
                    window.
    
    2014.03.24 -- A. Manalaysay
    """
    # Check if the time window has been calculated
    if forceWinRecalc or not ('tWindow_samples' in d.keys()):
        tempVar = evtWindow(d, tDrift=tDrift, tEv=tEv)
        if type(tempVar) == tuple:
            d['tWindow_samples'] = tempVar[0]
        else:
            d['tWindow_samples'] = tempVar
    cutS2 = d['pulse_classification'] == 2
    cutPulseInWindow =  (d['aft_t1_samples'] >= np.tile(d['tWindow_samples'][0,:],[cutS2.shape[0],1])) & \
                        (d['aft_t1_samples'] < np.tile(d['tWindow_samples'][1,:],[cutS2.shape[0],1]))
    maskS2InW = cutS2 & cutPulseInWindow
    return maskS2InW

def mask_S1inWindow(d, tDrift=tDrift0, tEv=tEv0, forceWinRecalc=False):
    """
    Return a boolean mask that picks out the S1 pulses in the event window before the largest S2
    pulse, if it exists.
    ------Inputs:
                 d: Python dictionary containing LUX RQs.
            tDrift: (Optional) kw var for use in evtWindow (see corresponding documentation).
               tEv: (Optional) kw var for use in evtWindow (see corresponding documentation).
    forceWinRecalc: (Optional) Force recalculation of the time window, if it has already been 
                    calculated. Default: False
    -----Outputs:
         maskS1InW: A NxM boolean array, where N is the number of pulses per event in an RQ file, 
                    and M is the number of events in 'd'. Array is True for S2 pulses in the event
                    window.
    
    2014.03.24 -- A. Manalaysay
    """
    # Check if the time window has been calculated
    if forceWinRecalc or not ('tWindow_samples' in d.keys()):
        tempVar = evtWindow(d, tDrift=tDrift, tEv=tEv)
        if type(tempVar) == tuple:
            d['tWindow_samples'] = tempVar[0]
        else:
            d['tWindow_samples'] = tempVar
    
    # Identify S1 and S2 pulses found by the pulse classifier
    cutS1 = d['pulse_classification'] == 1
    cutS2 = d['pulse_classification'] == 2
    
    # Calculate the [raw] area of the largest S2 pulse 
    S2areaLargest = (d['pulse_area_phe']*cutS2).max(0)
    
    # Create a cut that identifies the largest S2 pulse
    cutS2Largest  = cutS2 & (d['pulse_area_phe'] == np.tile(S2areaLargest,[cutS2.shape[0],1]))
    
    # Determine which pulses come before the largest S2
    cutPulsesBfLargestS2 = cutS2Largest.cumsum(0) == 0
    
    # Determine which pulses are in the time window
    cutPulseInWindow =  (d['aft_t1_samples'] >= np.tile(d['tWindow_samples'][0,:],[cutS1.shape[0],1])) & \
                        (d['aft_t1_samples'] < np.tile(d['tWindow_samples'][1,:],[cutS1.shape[0],1]))
    
    # Count the number of S1 that match the defined criteria
    maskS1InW = cutS1 & cutPulsesBfLargestS2 & cutPulseInWindow
    
    # Handle the situations where no S2 pulse was found
    maskS1InW[:,~(cutS2.sum(0)!=0)] = (cutS1 & cutPulseInWindow)[:, ~(cutS2.sum(0)!=0)]
    
    return maskS1InW

def mask_SMinWindow(d, M, tDrift=tDrift0, tEv=tEv0, forceWinRecalc=False):
    """
    Return a boolean mask that picks out the SM pulses in the event window, where M identifies the 
    pulse classification.
    ------Inputs:
                 d: Python dictionary containing LUX RQs.
                 M: Pulse identification number.
                    0: Not a pulse
                    1: S1
                    2: S2
                    3: Single photoelectron
                    4: Single-electron S2
                    5: Unidentified (ID failed)
                    6: (AC only) Baseline fluctuation.
                    7: (AC only) Cherenkov pulse.
                    8: (AC only) Part or all of an E-train.
            tDrift: (Optional) kw var for use in evtWindow (see corresponding documentation).
               tEv: (Optional) kw var for use in evtWindow (see corresponding documentation).
    forceWinRecalc: (Optional) Force recalculation of the time window, if it has already been 
                    calculated. Default: False
    -----Outputs:
         maskSMInW: A XxY boolean array, where X is the number of pulses per event in an RQ file, 
                    and Y is the number of events in 'd'. Array is True for SM pulses in the event
                    window.
    
    2014.03.24 -- A. Manalaysay
    """
    # Check if the time window has been calculated
    if forceWinRecalc or not ('tWindow_samples' in d.keys()):
        tempVar = evtWindow(d, tDrift=tDrift, tEv=tEv)
        if type(tempVar) == tuple:
            d['tWindow_samples'] = tempVar[0]
        else:
            d['tWindow_samples'] = tempVar
    cutSM = d['pulse_classification'] == M
    cutPulseInWindow =  (d['aft_t1_samples'] >= np.tile(d['tWindow_samples'][0,:],[cutSM.shape[0],1])) & \
                        (d['aft_t1_samples'] < np.tile(d['tWindow_samples'][1,:],[cutSM.shape[0],1]))
    maskSMInW = cutSM & cutPulseInWindow
    return maskSMInW

def NS2_InWindow(d, tDrift=tDrift0, tEv=tEv0, forceWinRecalc=False):
    """
    Return the number of S2 pulses for each event in the time window determined by function 
    "evtWindow" in this module.
    
    ------Inputs:
                 d: Python dictionary containing LUX RQs.
            tDrift: (Optional) kw var for use in evtWindow (see corresponding documentation).
               tEv: (Optional) kw var for use in evtWindow (see corresponding documentation).
    forceWinRecalc: (Optional) Force recalculation of the time window, if it has already been 
                    calculated. Default: False
    -----Outputs:
       NS2inWindow: A length-N numpy array, where N is the number of events in 'd', specifying the 
                    number of S2 pulses found in the event window.
    
    2014.03.24 -- A. Manalaysay
    """
    NS2inWindow = mask_S2inWindow(d, tDrift=tDrift, tEv=tEv, forceWinRecalc=forceWinRecalc).sum(0)
    return NS2inWindow

def NS2_InW_thr(d, thresh=100, tDrift=tDrift0, tEv=tEv0, forceWinRecalc=False):
    """
    Return the number of S2 pulses in the event window that are above a specified threshold, in raw 
    phe.  The event window is that returned by the function "evtWindow" in this module.
    
    ------Inputs:
                 d: Python dictionary containing LUX RQs.
            thresh: (Optional) Threshold above which to count S2 pulses, in raw phe.
                    Default: 100 phe.
            tDrift: (Optional) kw var for use in evtWindow (see corresponding documentation).
               tEv: (Optional) kw var for use in evtWindow (see corresponding documentation).
    forceWinRecalc: (Optional) Force recalculation of the time window, if it has already been 
                    calculated. Default: False
    -----Outputs:
       NS2inWindow: A length-N numpy array, where N is the number of events in 'd', specifying the 
                    number of S2 pulses found in the event window above threshold.
    
    2014.03.24 -- A. Manalaysay
    """
    maskS2 = mask_S2inWindow(d, tDrift=tDrift, tEv=tEv, forceWinRecalc=forceWinRecalc)
    maskPulseAboveThr = d['pulse_area_phe'] >= thresh
    NS2inWindow =  (maskS2 & maskPulseAboveThr).sum(0)
    return NS2inWindow

def NS1_InWindow(d, tDrift=tDrift0, tEv=tEv0, forceWinRecalc=False):
    """
    Return the number of S1 pulses for each event in the time window determined by function 
    "evtWindow" in this module, but only if they occur before the largest S2 (if it exists).  If no 
    S2 pulse exists, it just blindly returns the number of S1 pulses in the event window.
    
    ------Inputs:
                 d: Python dictionary containing LUX RQs.
            tDrift: (Optional) kw var for use in evtWindow (see corresponding documentation).
               tEv: (Optional) kw var for use in evtWindow (see corresponding documentation).
    forceWinRecalc: (Optional) Force recalculation of the time window, if it has already been 
                    calculated. Default: False
    -----Outputs:
       NS2inWindow: A length-N numpy array, where N is the number of events in 'd', specifying the 
                    number of S1 pulses found in the event window.
    
    2014.03.24 -- A. Manalaysay
    """
    NS1Window = mask_S1inWindow(d, tDrift=tDrift, tEv=tEv, forceWinRecalc=forceWinRecalc).sum(0)
    return NS1Window

def NSM_InWindow(d, M, tDrift=tDrift0, tEv=tEv0, forceWinRecalc=False):
    """
    Return the number of SM pulses for each event in the time window determined by function 
    "evtWindow" in this module. M indicates the pulse identification number.
    
    ------Inputs:
                 d: Python dictionary containing LUX RQs.
                 M: Pulse identification number.
                    0: Not a pulse
                    1: S1
                    2: S2
                    3: Single photoelectron
                    4: Single-electron S2
                    5: Unidentified (ID failed)
                    6: (AC only) Baseline fluctuation.
                    7: (AC only) Cherenkov pulse.
                    8: (AC only) Part or all of an E-train.
            tDrift: (Optional) kw var for use in evtWindow (see corresponding documentation).
               tEv: (Optional) kw var for use in evtWindow (see corresponding documentation).
    forceWinRecalc: (Optional) Force recalculation of the time window, if it has already been 
                    calculated. Default: False
    -----Outputs:
       NSMinWindow: A length-N numpy array, where N is the number of events in 'd', specifying the 
                    number of S2 pulses found in the event window.
    
    2014.03.24 -- A. Manalaysay
    """
    NSMinWindow = mask_SMinWindow(d, M, tDrift=tDrift, tEv=tEv, forceWinRecalc=forceWinRecalc).sum(0)
    return NSMinWindow

def Nth_S1_Window(d, N=1, quantity='xyz_corrected_pulse_area_all_phe', tDrift=tDrift0, tEv=tEv0, 
                  forceWinRecalc=False):
    """
    Return the Nth S1 (by time) in the event window, before the largest S2 (if it exists).  The event 
    window is defined by the boundaries returned by dp.evtWindow.
    ------Inputs:
                 d: Python dictionary containing LUX RQs.
                 N: (Optional) The number of the S1 pulse you want returned. Default: 1 (i.e. the first)
          quantity: (Optional) String specifying the name of the RQ to be returned.
                    Default: 'xyz_corrected_pulse_area_all_phe'
            tDrift: (Optional) kw var for use in evtWindow (see corresponding documentation).
               tEv: (Optional) kw var for use in evtWindow (see corresponding documentation).
    forceWinRecalc: (Optional) Force recalculation of the time window, if it has already been 
                    calculated. Default: False
    -----Outputs:
              qOut: The value of the quantity specified corresponding to the Nth S1 pulse in each event.
    
    2014.03.24 -- A. Manalaysay
    """
    S1Mask = mask_S1inWindow(d, tDrift=tDrift, tEv=tEv, forceWinRecalc=forceWinRecalc)
    NthS1Mask = (S1Mask.cumsum(0) == N) & S1Mask
    qOut = (d[quantity]*NthS1Mask).sum(0)
    return qOut

def Nth_S2_Window(d, N=1, quantity='xyz_corrected_pulse_area_all_phe', tDrift=tDrift0, tEv=tEv0, 
                  forceWinRecalc=False):
    """
    Return the Nth S2 (by time) in the event window.  The event window is defined by the boundaries 
    returned by dp.evtWindow.
    ------Inputs:
                 d: Python dictionary containing LUX RQs.
                 N: (Optional) The number of the S2 pulse you want returned. Default: 1 (i.e. the first)
          quantity: (Optional) String specifying the name of the RQ to be returned.
                    Default: 'xyz_corrected_pulse_area_all_phe'
            tDrift: (Optional) kw var for use in evtWindow (see corresponding documentation).
               tEv: (Optional) kw var for use in evtWindow (see corresponding documentation).
    forceWinRecalc: (Optional) Force recalculation of the time window, if it has already been 
                    calculated. Default: False
    -----Outputs:
              qOut: The value of the quantity specified corresponding to the Nth S1 pulse in each event.
    
    2014.03.24 -- A. Manalaysay
    """
    S2Mask = mask_S2inWindow(d, tDrift=tDrift, tEv=tEv, forceWinRecalc=forceWinRecalc)
    NthS2Mask = (S2Mask.cumsum(0) == N) & S2Mask
    qOut = (d[quantity]*NthS2Mask).sum(0)
    return qOut

def Nth_SM_Window(d, M, N=1, quantity='xyz_corrected_pulse_area_all_phe', tDrift=tDrift0, tEv=tEv0, 
                  forceWinRecalc=False):
    """
    Return the Nth SM (by time) in the event window, where M refers to the pulse classification integer.
    The event window is defined by the boundaries returned by dp.evtWindow.
    ------Inputs:
                 d: Python dictionary containing LUX RQs.
                 M: Pulse identification number.
                    0: Not a pulse
                    1: S1
                    2: S2
                    3: Single photoelectron
                    4: Single-electron S2
                    5: Unidentified (ID failed)
                    6: (AC only) Baseline fluctuation.
                    7: (AC only) Cherenkov pulse.
                    8: (AC only) Part or all of an E-train.
                 N: (Optional) The number of the SM pulse you want returned. Default: 1 (i.e. the first)
          quantity: (Optional) String specifying the name of the RQ to be returned.
                    Default: 'xyz_corrected_pulse_area_all_phe'
            tDrift: (Optional) kw var for use in evtWindow (see corresponding documentation).
               tEv: (Optional) kw var for use in evtWindow (see corresponding documentation).
    forceWinRecalc: (Optional) Force recalculation of the time window, if it has already been 
                    calculated. Default: False
    -----Outputs:
              qOut: The value of the quantity specified corresponding to the Nth SM pulse in each event.
    
    2014.03.24 -- A. Manalaysay
    """
    SMMask = mask_SMinWindow(d, M, tDrift=tDrift, tEv=tEv, forceWinRecalc=forceWinRecalc)
    NthSMMask = (SMMask.cumsum(0) == N) & SMMask
    qOut = (d[quantity]*NthSMMask).sum(0)
    return qOut

def NthLargestS2(d, N=1, quantity='xyz_corrected_pulse_area_all_phe', tDrift=tDrift0, tEv=tEv0, 
                 forceWinRecalc=False):
    """
    Return the Nth S2 (by raw area) in the event window.  The event window is defined by the boundaries 
    returned by dp.evtWindow.
    ------Inputs:
                 d: Python dictionary containing LUX RQs.
                 N: (Optional) The number of the S2 pulse you want returned. Default: 1 (i.e. the largest)
          quantity: (Optional) String specifying the name of the RQ to be returned.
                    Default: 'xyz_corrected_pulse_area_all_phe'
            tDrift: (Optional) kw var for use in evtWindow (see corresponding documentation).
               tEv: (Optional) kw var for use in evtWindow (see corresponding documentation).
    forceWinRecalc: (Optional) Force recalculation of the time window, if it has already been 
                    calculated. Default: False
    -----Outputs:
              qOut: The value of the quantity specified corresponding to the Nth S1 pulse in each event.
    
    2014.03.25 -- A. Manalaysay
    """
    S2Mask = mask_S2inWindow(d, tDrift=tDrift, tEv=tEv, forceWinRecalc=forceWinRecalc)
    S2Areas = d['pulse_area_phe'] * S2Mask
    while (N > 1):
        S2Largest = S2Areas.max(0)
        S2Areas[S2Areas==np.tile(S2Largest,[d['pulse_area_phe'].shape[0],1])] = 0
        N -= 1
    S2Largest = S2Areas.max(0)
    S2NthLargestMask = (d['pulse_area_phe'] == np.tile(S2Largest,[d['pulse_area_phe'].shape[0],1])) & S2Mask
    qOut = (d[quantity] * S2NthLargestMask).sum(0)
    return qOut

def concatRQsWcuts(rqBasePath_list, fieldsToKeep, cuts='none', fileList='all', printMultiple=5):
    """
    Take RQ files and concatenate them with some cuts applied, outputting the results in a dictionary
    subclass ("S", part of aLib).  The statistics of the cuts are recorded in a few added dict keys, 
    as well as information (on an event-by-event basis) of which file and data set a particular event
    came from.
    
    -------Inputs:
        rqBasePath_list: Python list containing strings.  Each string specifies a path from which
                         RQ files will be taken.  Must include a trailing '/'
           fieldsToKeep: Which fields in the RQ files to keep, given as a Python list with a 
                         string as each element.  IMPORTANT: This list MUST include 'file_number', 
                         and 'pulse_classification'.
                   cuts: Cuts to apply. This input must be given as a list of function handles (even 
                         if only one function is given, it must be a list of one element). The 
                         user-defined functions contained therein must take a LUX RQ dict as input 
                         and give a boolean ndarray as output. The name of the function given will
                         be used when identifying the efficiency of the cut (that is, the thing that 
                         comes out of func.__name__).
               fileList: List of files, if you don't want to concatenate all the files in the path.
                         This will choke if input 'rqBasePath_list' has more than one element 
                         (NOTE: even if rqBasePath_list has only one entry, it MUST still be a list)
          printMultiple: Integer, default: 5.  A progress notification will be printed after each 
                         multiple of this number of files has been loaded.
    ------Outputs:
                      d: Dictionary containing the desired fields, concatenated from the desired
                         datasets, with the desired cuts applied. 
                         **NOTE**: This is not the builtin python "dict" class; this is my dict
                         subclass (which is better than plain dict), located in the module "aLib".
                         The following keys are added to the dictionary:
                               RQ_paths: Copy of 'rqBasePath' that was input.
                                pathNum: Tells which element of RQ_paths the event fell in.
                                fileNum: Tells which file in the path element the event fell in.
                         numEventsTotal: Tells how many events were contained in the RQs total, 
                                         before cuts.
                            numCutsPass: Included if input 'cuts' is not 'none'. Gives the number of 
                                         events that passed each cut, separately.
    
    Example: (it's better to do this as a non-interactive script, but whatever).
        In [1]: from aLib import dp
        In [2]: rqBasePath_list = []
        In [3]: rqBasePath.append("/data/lux10_20131031T1605_cp07728/matfiles/")
        In [4]: rqBasePath.append("/data/lux10_20131031T1605_cp07728/matfiles/")
        In [5]: fields = [<some list of fields as strings>]
        In [6]: def SingleScatterCut(b):
           ...:     <cut logic>
           ...:    return CutArray_bool
        In [7]: def FiducialCut(b):
           ...:     <cut logic>
           ...:     return CutArray_bool
        In [8]: cuts = [SingleScatterCut, FiducialCut]
        In [9]: d = dp.concatRQsWcuts(rqBasePath_list, fields, cuts)
    
    2014.03.01 -- A. Manalaysay
    2014.03.23 -- A. Manalaysay - Changed the method by which the user passes cuts to the concatenation
                                  function, allowing for more sophisticated cuts to be implemented. 
    2014.03.27 -- A. Manalaysay - Added pre-allocation of memory for the final data dictionary.  Memory 
                                  size is estimated by applying cuts to the first 10 files, counting how 
                                  many events pass, adding 2*sigma, and then scaling by the total number 
                                  of files; if this memory is exceeded, the function still works, it just 
                                  appends new events onto the existing structure.  A warning is given if 
                                  the estimated memory usage will be more than 2 GB, in which case the 
                                  user has 10 seconds to kill the function via ctrl-c.  Additionally, 
                                  error handling has been added to handle cases where a file can't load, 
                                  or the file doesn't have the requisite RQs.
    2014.06.13 -- A. Manalaysay - Added error-checking code to make sure that certain required fields 
                                  are included in input 'fieldsToKeep'.
    """
    # Input checking
    if (type(cuts)==str):
        if cuts=="none":
            flag_cutsNone = True
        else:
            raise TypeError("I dont' recognize your input 'cuts'.")
    elif (type(cuts)==list):
        flag_cutsNone = False
        numCutsPass = [0]*len(cuts)
        cutNames = [cut.__name__ for cut in cuts]
    else:
        raise TypeError("Input 'cuts' must be either 'none' or a list of function handles.")
    numEventsTotal = 0
    fieldDimLengths = np.zeros(len(fieldsToKeep),dtype=np.uint8)
    requiredFields = ['file_number','pulse_classification']
    for reqField in requiredFields:
        if reqField not in fieldsToKeep:
            raise ValueError("Input 'fieldsToKeep' is missing required field '{:s}'".format(reqField))
    
    print("\n<<<<<<<<<<<<<<<<<<< dp.concatRQsWcuts >>>>>>>>>>>>>>>>>>>")
    print("<<<<                     STARTING                    >>>>")
    print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")
    
    #--------------------------<Determine the number of files>--------------------------#
    print("Determining total number of files.", end="", flush=True)
    NumFilesTotal = 0
    for kRQ, rqBasePath in enumerate(rqBasePath_list):
        print(".", end="", flush=True)
        if type(fileList)==str:
            if fileList=="all":
                rqFileNames = os.listdir(rqBasePath)
            else:
                raise TypeError("I don't recognize your fileList.")
        else:
            rqFileNames = fileList
        NumFilesTotal += len(rqFileNames)
    print("\t {:d}".format(NumFilesTotal))
    #--------------------------</Determine the number of files>--------------------------#
    
    #------------------------<Estimate the total passing events>------------------------#
    tStatMessage = "Estimating number of events that will pass cuts."
    print(tStatMessage, end="\r", flush=True)
    kRQ = 0
    rqBasePath = rqBasePath_list[0]
    numCutPassInit = 0
    
    if fileList=="all":
        rqFileNames = os.listdir(rqBasePath)
    else:
        rqFileNames = fileList
    
    bytesPerEvent = 0
    flag_NoInitYet = True
    #b = aLib.S()
    for k0 in range(min(10,len(rqFileNames))):
        print("{:s}{:s}".format(tStatMessage, k0*"."),end='\r',flush=True)
        FileName = rqFileNames[k0]
        try:
            b = aLib.mfile.mload(rqBasePath + FileName,variable_names=fieldsToKeep)
            if 'pulse_classification' in b.keys():
                flag_loadSuccess = True
            else:
                flag_loadSuccess = False
        except:
            flag_loadSuccess = False
            pass
        if flag_loadSuccess:
            if flag_NoInitYet:
                flag_NoInitYet = False
                for k1, FieldName in enumerate(fieldsToKeep):
                    fieldDims = np.ndim(b[FieldName])
                    if fieldDims==1:
                        bytesPerEvent += np.empty(1, dtype=b[FieldName].dtype).nbytes
                    elif fieldDims==2:
                        bytesPerEvent += np.empty([b[FieldName].shape[0],1], dtype=b[FieldName].dtype).nbytes
                    elif fieldDims==3:
                        bytesPerEvent += \
                            np.empty([b[FieldName].shape[0],b[FieldName].shape[1],1], \
                            dtype=b[FieldName].dtype).nbytes
            # Make cuts
            if flag_cutsNone:
                tCutsAll = np.array([True]*len(b['file_number']))
            else:
                tCuts = []
                for kCut, cut in enumerate(cuts):
                    tCutItem = cut(b)
                    tCuts.append(tCutItem)
                del cut
                tCutsAll = np.tile(True,len(tCuts[0]))
                for cut in tCuts:
                    tCutsAll = tCutsAll & cut
            numCutPassInit += tCutsAll.sum()
    # Scale the number that pass the cuts by the number of total files and add 10%.
    NumEstPass = np.int64(numCutPassInit * NumFilesTotal / min(10,len(rqFileNames)))
    NumEstPassExtra = np.int64((numCutPassInit+2*np.sqrt(numCutPassInit)) * \
        NumFilesTotal / min(10,len(rqFileNames)))
    print("{:s}{:s} Roughly {:d}".format(tStatMessage, k0*".", np.uint64(NumEstPass)))
    def repSize(x):
        byteRank = np.floor(np.log2(x)/10)
        decChar = ['B','KB','MB','GB','TB','PB','EB']
        return "{:0.1f} {:s}".format(x/(2**(byteRank*10)),decChar[np.uint16(byteRank)])
    if (bytesPerEvent*NumEstPassExtra) < (2**31): # 2GB warning flag is hard coded.
        print("\n{:s} Allocating {:s} {:s}".format(10*"*",repSize(bytesPerEvent*NumEstPassExtra), 10*"*"))
    else:
        print("******************* <WARNING> *******************")
        print("\n    {:s} Allocating {:s}!! {:s}".format(10*"*",repSize(bytesPerEvent*NumEstPassExtra), 10*"*"))
        print(" ( CTRL-C WITHIN 10 SECONDS IF THAT'S TOO MUCH )")
        print("\n******************* <\\WARNING> *******************")
        from time import sleep
        sleep(10)
        
    #------------------------</Estimate the total passing events>------------------------#
    
    
    
    #***************************************************************************************************
    # Initialize the data structure
    d = aLib.S()
    d.RQ_paths = rqBasePath_list
    k_CutStep = 0
    flag_pastAllocate = False
    flag_pastAllocateFirst = True
    str_pastAllocate = ""
    flag_NoInitYet = True
    for kRQ, rqBasePath in enumerate(rqBasePath_list):
        if fileList=="all":
            rqFileNames = os.listdir(rqBasePath)
        else:
            rqFileNames = fileList
        print('\n\n**** {:s} ****'.format(rqBasePath))
        for k0, FileName in enumerate(rqFileNames):
            FileName = rqFileNames[k0]
            if ((k0 % printMultiple) == 0) or (k0==(len(rqFileNames)-1)) :
                print("-----({:s}{:s}) {:d}/{:d} files\t{:s}".format(">"*np.int64((k0/(len(rqFileNames)-1))*40), 
                    " "*(40-np.int64((k0/(len(rqFileNames)-1))*40)), k0+1, len(rqFileNames), str_pastAllocate),
                    end="\r", flush=True)
            try:
                b = aLib.mfile.mload(rqBasePath + FileName,variable_names=fieldsToKeep)
                #   aLib.mfile.mload(rqBasePath + FileName, mdict=b, variable_names=fieldsToKeep)
                if 'pulse_classification' in b.keys():
                    flag_loadSuccess = True
                else:
                    flag_loadSuccess = False
            except:
                flag_loadSuccess = False
                pass
            if flag_loadSuccess:
                numEventsTotal += len(b['file_number'])
                # Initialize fields in the data structure and allocating memory
                if flag_NoInitYet:
                    flag_NoInitYet = False
                    d.pathNum = np.empty(NumEstPassExtra, dtype=np.uint8)
                    d.fileNum = np.empty(NumEstPassExtra,dtype=np.uint16)
                    for k1, FieldName in enumerate(fieldsToKeep):
                        fieldDims = np.ndim(b[FieldName])
                        fieldDimLengths[k1] = fieldDims
                        #if fieldDims == 0:
                        #    d[FieldName] = []
                        if fieldDims == 1:
                            d[FieldName] = np.empty(NumEstPassExtra, dtype=b[FieldName].dtype)
                        elif fieldDims == 2:
                            d[FieldName] = np.empty([b[FieldName].shape[0],NumEstPassExtra],
                                dtype=b[FieldName].dtype)
                        elif fieldDims == 3:
                            d[FieldName] = \
                                np.empty([b[FieldName].shape[0],b[FieldName].shape[1],NumEstPassExtra], \
                                dtype=b[FieldName].dtype)
                # Make cuts
                if flag_cutsNone:
                    tCutsAll = np.array([True]*len(b['file_number']))
                else:
                    tCuts = []
                    for kCut, cut in enumerate(cuts):
                        tCutItem = cut(b)
                        numCutsPass[kCut] += tCutItem.sum()
                        tCuts.append(tCutItem)
                    del cut
                    tCutsAll = np.tile(True,len(tCuts[0]))
                    for cut in tCuts:
                        tCutsAll = tCutsAll & cut
                tNumCutPass = tCutsAll.sum()
                if (k_CutStep+tNumCutPass) < NumEstPassExtra:  # The normal action
                    # Apply cuts and append to the final data structure
                    d.pathNum[k_CutStep:(k_CutStep+tNumCutPass)] = \
                        np.ones(tCutsAll.sum(),dtype=np.uint8)*np.uint8(kRQ)
                    d.fileNum[k_CutStep:(k_CutStep+tNumCutPass)] = \
                        np.ones(tCutsAll.sum(),dtype=np.uint16)*np.uint16(k0)
                    for k1, FieldName in enumerate(fieldsToKeep):
                        if fieldDimLengths[k1] == 1:
                            d[FieldName][k_CutStep:(k_CutStep+tNumCutPass)] = b[FieldName][tCutsAll]
                        elif fieldDimLengths[k1] == 2:
                            d[FieldName][:,k_CutStep:(k_CutStep+tNumCutPass)] = b[FieldName][:,tCutsAll]
                        elif fieldDimLengths[k1] == 3:
                            d[FieldName][:,:,k_CutStep:(k_CutStep+tNumCutPass)] = b[FieldName][:,:,tCutsAll]
                    k_CutStep += tNumCutPass
                else:   # Uh oh, this only runs if you've run past the initial mem allocation
                    if flag_pastAllocateFirst:
                        flag_pastAllocate = True
                        flag_pastAllocateFirst = False
                        str_pastAllocate = "(Past initial mem. allocation)"
                        for k2, FieldName in enumerate(fieldsToKeep):
                            if fieldDimLengths[k2] == 1:
                                d[FieldName] = d[FieldName][:k_CutStep]
                            if fieldDimLengths[k2] == 2:
                                d[FieldName] = d[FieldName][:,:k_CutStep]
                            if fieldDimLengths[k2] == 3:
                                d[FieldName] = d[FieldName][:,:,:k_CutStep]
                    # Apply cuts and append to the final data structure
                    d.pathNum = np.hstack([d.pathNum, np.ones(tCutsAll.sum(),dtype=np.uint8)+np.uint8(kRQ)])
                    d.fileNum = np.hstack([d.fileNum, np.ones(tCutsAll.sum(),dtype=np.uint16)+np.uint16(k0)])
                    for k1, FieldName in enumerate(fieldsToKeep):
                        if fieldDimLengths[k1] == 1:
                            d[FieldName] = np.concatenate((d[FieldName],b[FieldName][tCutsAll]), axis=0)
                        elif fieldDimLengths[k1] == 2:
                            d[FieldName] = np.concatenate((d[FieldName],b[FieldName][:,tCutsAll]), axis=1)
                        elif fieldDimLengths[k1] == 3:
                            d[FieldName] = np.concatenate((d[FieldName],b[FieldName][:,:,tCutsAll]), axis=2)
    
    if not flag_pastAllocate:
        # Truncate the RQs of the extra entries that weren't filled
        d.pathNum = d.pathNum[:k_CutStep]
        d.fileNum = d.fileNum[:k_CutStep]
        for k1, FieldName in enumerate(fieldsToKeep):
            if fieldDimLengths[k1] == 1:
                d[FieldName] = d[FieldName][:k_CutStep]
            elif fieldDimLengths[k1] == 2:
                d[FieldName] = d[FieldName][:,:k_CutStep]
            elif fieldDimLengths[k1] == 3:
                d[FieldName] = d[FieldName][:,:,:k_CutStep]
    
    d['numEventsTotal'] = numEventsTotal
    if not flag_cutsNone:
        d['numCutsPass'] = [[x,y] for x,y in zip(cutNames,numCutsPass)]
        print("\n\nNumber of events before cuts: {:d}".format(numEventsTotal))
        print("Number of events after  cuts: {:d}".format(d.pulse_classification.shape[1]))
        length_cutNameMax = max(len("Cut name"),np.array([len(cName) for cName in cutNames]).max())
        print("{{:>{:d}s}}: {{:s}}".format(length_cutNameMax).format("Cut name","Num passing"))
        print("{:s}: {:s}".format(length_cutNameMax*"-", len("Num passing")*"-"))
        for k in range(len(cutNames)):
            print("{{:>{:d}s}}: {{:d}}".format(length_cutNameMax).format(cutNames[k],numCutsPass[k]))
    return d












