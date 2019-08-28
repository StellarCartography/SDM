import numpy as np
from scipy.stats import binned_statistic
# import pandas as pd


def fib(n):
    '''
    return n digits of the Fibonacci sequence
    very inefficiently b/c i'm basic
    '''
    out = np.zeros(n)
    out[1] = 1
    for k in range(2,n):
        out[k] = out[k-2] + out[k-1]
    return out
    
    
def fib_approx(n):
    '''
    https://stackoverflow.com/a/4936086
    but w/ numpy arrays b/c i'm basic
    '''
    return np.array(((1 + np.sqrt(5)) / 2) ** np.arange(n) / np.sqrt(5) + 0.5, dtype=np.int)


def _gaus(x, a, b, x0, sigma):
    """
    Simple Gaussian function

    Parameters
    ----------
    x : float or 1-d numpy array
        The data to evaluate the Gaussian over
    a : float
        the amplitude
    b : float
        the constant offset
    x0 : float
        the center of the Gaussian
    sigma : float
        the width of the Gaussian

    Returns
    -------
    Array or float of same type as input (x).
    """
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + b


def fib_gaus1(time, tau, amp, std, t0=0, Nfib=-1):
    '''
    Make a sequence of identical Gaussian pulses separated by a Fibonacci sequence
    
    Parameters
    ----------
    time : the input time array
    tau : the key Fibonacci scaling size
    amp : the height of the Gaussian
    std : the width of the Gaussian
    t0 : the start time of the Fib sequence 
        0 by default
    Nfib : how many Fibonacci steps to model
        if set to -1, code will pick the max that fits within the time array's
        start and stop range.
        
    Returns
    -------
    An array of fluxes evaluated at the input times
    '''

    flux = np.zeros_like(time)
    
    if Nfib == -1:
        Nfib = 1
        dur = (np.nanmax(time) - np.nanmin(time)) / tau
        k=0
        while k<1:
            Nfib += 1
            cdur = np.nanmax(np.cumsum(fib(int(Nfib))))
            if cdur >= dur:
                k=1

    if Nfib < 2:
        Nfib = 2
    
    fib_seq = fib(int(Nfib)) * tau
    
    for k in range(len(fib_seq)):
        flux = flux + _gaus(time, amp, 0., t0 + np.cumsum(fib_seq)[k], std)
    
    return flux


def fib_gaus2(time, tau, amp, std, t0=0, Nfib=-1):
    '''
    Make a sequence of Gaussian pulses separated and scaled by a Fibonacci sequence
    
    Parameters
    ----------
    time : the input time array
    tau : the key Fibonacci scaling size
    amp : the height of the Gaussian
    std : the width of the Gaussian
    t0 : the start time of the Fib sequence
        0 by default
    Nfib : how many Fibonacci steps to model
        if set to -1, code will pick the max that fits within the time array's
        start and stop range.
        
    Returns
    -------
    An array of fluxes evaluated at the input times
    '''

    flux = np.zeros_like(time)
    
    if Nfib == -1:
        Nfib = 1
        dur = (np.nanmax(time) - np.nanmin(time)) / tau
        k=0
        while k<1:
            cdur = np.nanmax(np.cumsum(fib(int(Nfib))))
            if cdur >= dur:
                k=1
            Nfib += 1

    if Nfib < 2:
        Nfib = 2
    
    fib_seq = fib(int(Nfib)) * tau
    
    for k in range(len(fib_seq)):
        flux = flux + _gaus(time, amp, 0., t0 + np.cumsum(fib_seq)[k], std * (1+fib_seq[k])/tau)
    
    return flux


def chisq(data, error, model):
    '''
    Compute the normalized chi square statistic:
    chisq =  1 / N * SUM(i) ( (data(i) - model(i))/error(i) )^2
    '''
    return np.nansum( ((data - model) / error)**2.0 ) / np.size(data)



def SDM(time, flux, error, tau, t0, Nfib=-1, Nbins=50, stretch=True):
    '''
    compute the Fibonacci Sequence Dispersion Minimization for a dataset,
    given the characteristic timescale (tau) and pattern start time (t0).
    
    '''
    x_i = []
    y_i = []
    e_i = []

    # automatically pick the Fibonacci number to use, based on the total baseline of the data
    if Nfib == -1:
        Nfib = 1
        dur = (np.nanmax(time) - np.nanmin(time)) / tau
        k=0
        while k<1:
            Nfib += 1
            cdur = np.nanmax(np.cumsum(fib(int(Nfib))))
            if cdur >= dur:
                k=1
        Nfib += 1

    # but make sure to use at least 2!
    if Nfib < 2:
        Nfib = 2

    fib_seq_i = fib(int(Nfib)) * tau
    
    # step thru each fibonacci spacing, and shift (& scale) the data
    for k in range(1,len(fib_seq_i)):
        if stretch:
            scale = (1 + fib_seq_i[k]) / tau
        else:
            scale = 1.

        pp = (time - (t0 + np.cumsum(fib_seq_i)[k-1])) / scale

        ok = (pp/tau >= -0.5) & (pp/tau <= 0.5)
        x_i.extend(pp[ok]/tau)
        y_i.extend(flux[ok])
        e_i.extend(error[ok])

        
    x_i = np.array(x_i)
    y_i = np.array(y_i)
    e_i = np.array(e_i)
    
#     bin_width = int(len(flux)/100) # for rolling mean
#     if bin_width < 10:
#         bin_width = 10
#     ss_i = np.argsort(x_i)
#     model = pd.Series(y_i[ss_i]).rolling(bin_width, center=True).mean().values

    # my test shows this to be marginally faster, and very similar in result
    bin_means, bin_edges, binnumber = binned_statistic(x_i, y_i, statistic='mean', 
                                                       bins=np.linspace(-0.5, 0.5, Nbins))
    bin_centers = (bin_edges[1:] + bin_edges[0:-1])/2
    model = np.interp(x_i, bin_centers, bin_means)

    resid = chisq(y_i, e_i, model)
    return resid


def SDM_search(time, flux, error, tau, t0, Nfib=-1, stretch=True):
    '''
    compute the Sequence Dispersion Minimization 
    over a grid of tau & t0 values...
    
    '''
    resid = np.zeros((len(tau), len(t0)))
    
    # I THINK this could be refactored with list comprehension to speed it up a bit
    for j in range(len(t0)):
        for i in range(len(tau)):
            resid[i,j] = SDM(time, flux, error, tau[i], t0[j], Nfib, stretch=stretch)
        
    return resid