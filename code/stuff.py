import numpy as np


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
    t0 : the start time of the Fib sequence relative to the start of the time array
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
        flux = flux + _gaus(time, amp, 0., np.nanmin(time) + t0 + np.cumsum(fib_seq)[k], std)
    
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
    t0 : the start time of the Fib sequence relative to the start of the time array
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
        flux = flux + _gaus(time, amp, 0., np.nanmin(time) + t0 + np.cumsum(fib_seq)[k], std * (1+fib_seq[k])/tau)
    
    return flux
