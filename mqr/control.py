"""
Statistical process control routines.

The routines are for Xbar and R charts. The routines produce control limits from
historical data (known processes) for use in future production to detect
deviations in the distributions of process outputs.
"""

from mqr.constants import Constants as C
import matplotlib.pyplot as plt
import numpy as np

def xbar_lcl(xbar, rbar, n):
    """
    Lower control limit on an X-bar chart.

    Arguments
    ---------
    xbar: float -- Historical mean of sample means.
    rbar: float -- Historical mean of smaple ranges.
    n: int -- Number of samples (each with multiple observations).

    Returns
    -------
    float -- Lower control limit
    """
    n = np.clip(0, C.N_MAX, n)
    return xbar - C.A2[n] * rbar

def xbar_ucl(xbar, rbar, n):
    """
    Upper control limit on an x-bar chart.

    Arguments
    ---------
    xbar: float -- Historical mean of sample means.
    rbar: float -- Historical mean of smaple ranges.
    n: int -- Number of samples (each with multiple observations).

    Returns
    -------
    float -- Upper control limit
    """
    n = np.clip(0, C.N_MAX, n)
    return xbar + C.A2[n] * rbar

def rbar_lcl(rbar, n):
    """
    Lower control limit on an R chart.

    Arguments
    ---------
    rbar: float -- Historical mean of smaple ranges.
    n: int -- Number of samples (each with multiple observations).

    Returns
    -------
    float -- Lower control limit.
    """
    n = np.clip(0, C.N_MAX, n)
    return C.D3[n] * rbar

def rbar_ucl(rbar, n):
    """
    Upper control limit on an R chart.

    Arguments
    ---------
    rbar: float -- Historical mean of smaple ranges.
    n: int -- Number of samples (each with multiple observations).

    Returns
    -------
    float -- Uppeer control limit.
    """
    n = np.clip(0, C.N_MAX, n)
    return C.D4[n] * rbar

def x_control_limits(samples, x_mean=None, r_mean=None, std_hist=None):
    """
    Calculate Xbar control limits from historical data.

    If the means and standard deviations are not known, they will be estimated
    from `samples`.

    Arguments
    ---------
    samples: numpy.ndarray -- Samples with periods/time on the first dimension
        and replicates/groups on the second.

    Optional
    --------
    x_mean: float -- The historical mean. (Default calculated from `samples`.)
    r_mean: float -- The historical range. (Default calculated from `samples`.)
    std_hist: float -- The historical standard deviation. (Default calculated
            from `samples`.)

    Returns
    -------
    x_lcl: float -- Lower limit.
    x_cl: float -- Target.
    x_ucl:  -- Upper limit.
    """

    if (r_mean is not None) and (std_hist is not None):
        raise ValueError('cannot specify more than one of historical mean/range or historical stddev')

    m, n = samples.shape
    
    if x_mean is None:
        x_bar = np.mean(samples, axis=1)
        x_mean = np.mean(x_bar)

    if r_mean is None and std_hist is None:
        r_bar = np.max(samples, axis=1) - np.min(samples, axis=1)
        r_mean = np.mean(r_bar)
    elif std_hist is not None:
        r_mean = std_hist * d2[m]

    x_ucl = xbar_ucl(x_mean, r_mean, n)
    x_cl = x_mean 
    x_lcl = xbar_lcl(x_mean, r_mean, n)

    return (x_lcl, x_cl, x_ucl)

def r_control_limits(samples, r_mean=None, std_hist=None):
    """
    Calculates R control limits from historical data.

    If the mean and standard deviations are not known, they will be estimated
    from `samples`.

    Arguments
    ---------
    samples: numpy.ndarray -- Samples with periods/time on the first dimension
        and replicates/groups on the second.

    Optional
    --------
    r_mean: float -- The historical range. (Default calculated from `samples`.)
    std_hist: float -- The historical standard deviation. (Default calculated
            from `samples`.)

    Returns
    -------
    r_lcl: float -- Lower limit.
    r_cl: float -- Target.
    r_ucl: float -- Upper limit.
    """

    assert r_mean is None or std_hist is None, \
        'Cannot specify more than one of historical mean/range or historical stddev.'

    m, n = samples.shape

    if r_mean is None and std_hist is None:
        r_bar = np.max(samples, axis=1) - np.min(samples, axis=1)
        r_mean = np.mean(r_bar)
    elif std_hist is not None:
        r_mean = std_hist * d2[m]

    r_ucl = rbar_ucl(r_mean, n)
    r_cl  = r_mean
    r_lcl = rbar_lcl(r_mean, n)

    return (r_lcl, r_cl, r_ucl)
