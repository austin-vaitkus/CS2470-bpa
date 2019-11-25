"""
This module allows one to use background models and signal models to apply to Yellin's pmax method, whereby
an upper limit can be found in the presence of an unknown background.

There are three parts to coming up with a limit:

1.  Collecting all *known* spectra, both background, in order to form a mapping from observed variable to 
    uniform variable.  The uniform variable is simply the evaluated cumulative distribution of the known 
    spectra.  Note that this mapping must be *re*calculated at each value of sigma that is iterated over.

2.  Take input observed data, map them according to step 1, sort them, and calculate the Poisson p-value for 
    each interval; there are (N+1)*(N+2)/2 inteverals, given N observed events, take the max value.

3.  Compare the observed maximum p value from 2 to the 90% C.L. confidence interval from many repeated trials
    of a M.C. to determine the distribution of p_max given a certain expected number of events.  This is 
    essentially what is plotted in Yellin's Fig. 5, and I have found can be approximated with the analytic 
    function:
    
        p_max = 1 - 0.1815*(1 - st.gamma.cdf(log10(mu), 2.1513, scale=0.2199))
    
    where "st" is an alias for the pylab module 'scipy.stats', and st.gamma.cdf is the cumulative distribution 
    function for a gamma distribution.



16.01.06 -- A. Manalaysay
"""




