from mqr.inference.lib.util import bounded_error_msg
from mqr.utils import clip_where

import numpy as np
import scipy

import warnings

def confint_1sample_chi2(count, n, meas, conf, bounded):
    """
    Confidence interval for rate `count / n / meas` using the chi-squared
    method. See [1].

    Arguments
    ---------
    count (int) -- Number of events.
    n (int) -- Number of periods over which events were counted.
    meas (float) -- Extent of one period of observation. (Default 1.0.)
    conf (float) -- Confidence level that determines the width of the interval.
        (Default 0.95.)
    bounded (str) -- Which sides of the interval to close. One of "both",
        "below" or "above". (Default "both".)

    Returns
    -------
    mqr.confint.ConfidenceInterval

    References
    ----------
    [1] Patil, V. V., & Kulkarni, H. V. (2012).
        Comparison of confidence intervals for the Poisson mean: some new aspects.
        REVSTAT-Statistical Journal, 10(2), 211-22.
    """
    alpha = 1 - conf
    if bounded == 'both':
        lower = scipy.stats.chi2(2 * count).ppf(alpha / 2) / (2 * n * meas)
        upper = scipy.stats.chi2(2 * count + 2).ppf(1 - alpha / 2) / (2 * n * meas)
    elif bounded == 'below':
        lower = scipy.stats.chi2(2 * count).ppf(alpha) / (2 * n * meas)
        upper = np.inf
    elif bounded == 'above':
        lower = 0.0
        upper = scipy.stats.chi2(2 * count + 2).ppf(1 - alpha) / (2 * n * meas)
    else:
        raise ValueError(bounded_error_msg(bounded))
    return lower, upper

def confint_1sample_exact(count, n, meas, conf, bounded):
    """
    Confidence interval for rate `count / n / meas` using the exact method.

    Searches the Poisson cumulative distribution for a rate that that produces
    the given confidence when `count` and `n` have the specified values. See [1].

    Arguments
    ---------
    count (int) -- Number of events.
    n (int) -- Number of periods over which events were counted.
    meas (float) -- Extent of one period of observation. (Default 1.0.)
    conf (float) -- Confidence level that determines the width of the interval.
        (Default 0.95.)
    bounded (str) -- Which sides of the interval to close. One of "both",
        "below" or "above". (Default "both".)

    Returns
    -------
    mqr.confint.ConfidenceInterval

    References
    ----------
    [1] Barker, L. (2002).
        A comparison of nine confidence intervals for a Poisson parameter when
        the expected number of events is ≤ 5.
        The American Statistician, 56(2), 85-89.
    """
    if (count < 0) or (n < 0) or (meas < 0):
        raise ValueError(f'Arguments `count`, `n` and `meas` must all be non-negative.')
    alpha = (1 - conf) / 2 if (bounded == 'both') else (1 - conf)
    f_ineq_L = lambda mu: alpha - 1 + scipy.stats.poisson(mu*n).cdf(count)
    f_ineq_U = lambda mu: alpha - scipy.stats.poisson(mu*n).cdf(count)
    constraints_L = ({'type': 'ineq', 'fun': f_ineq_L})
    constraints_U = ({'type': 'ineq', 'fun': f_ineq_U})
    min_fun = lambda mu: mu
    max_fun = lambda mu: -mu
    if bounded == 'both':
        res_L = scipy.optimize.minimize(max_fun, count/n, constraints=constraints_L)
        res_U = scipy.optimize.minimize(min_fun, count/n, constraints=constraints_U)
        lower = res_L.x[0] if res_L.success else 0.0
        upper = res_U.x[0]
    elif bounded == 'below':
        res_L = scipy.optimize.minimize(max_fun, count/n, constraints=constraints_L)
        lower = res_L.x[0] if res_L.success else 0.0
        upper = np.inf
    elif bounded == 'above':
        res_U = scipy.optimize.minimize(min_fun, count/n, constraints=constraints_U)
        lower = 0.0
        upper = res_U.x[0]
    else:
        raise ValueError('nup')
    lower, upper = lower / meas, upper / meas
    return lower, upper

def confint_1sample_wald_cc(count, n, meas, conf, bounded):
    """
    Confidence interval for rate `count / n / meas` using the modified Wald method.

    The continuity correction is applied to both mean and variance. See [1] and [2].

    Arguments
    ---------
    count (int) -- Number of events.
    n (int) -- Number of periods over which events were counted.
    meas (float) -- Extent of one period of observation. (Default 1.0.)
    conf (float) -- Confidence level that determines the width of the interval.
        (Default 0.95.)
    bounded (str) -- Which sides of the interval to close. One of "both",
        "below" or "above". (Default "both".)

    Returns
    -------
    mqr.confint.ConfidenceInterval

    References
    ----------
    [1] Patil, V. V., & Kulkarni, H. V. (2012).
        Comparison of confidence intervals for the Poisson mean: some new aspects.
        REVSTAT-Statistical Journal, 10(2), 211-22.
    [2] Barker, L. (2002).
        A comparison of nine confidence intervals for a Poisson parameter when
        the expected number of events is ≤ 5.
        The American Statistician, 56(2), 85-89.
    """
    alpha = 1 - conf
    value = count / n / meas
    dist = scipy.stats.norm()
    if bounded == 'both':
        lower = (count - 0.5 + dist.ppf(alpha / 2) * np.sqrt(count - 0.5)) / (n * meas)
        upper = (count + 0.5 + dist.ppf(1 - alpha / 2) * np.sqrt(count + 0.5)) / (n * meas)
    elif bounded == 'below':
        lower = (count - 0.5 + dist.ppf(alpha) * np.sqrt(count - 0.5)) / (n * meas)
        upper = np.inf
    elif bounded == 'above':
        lower = 0.0
        upper = (count + 0.5 + dist.ppf(1 - alpha) * np.sqrt(count + 0.5)) / (n * meas)
    else:
        raise ValueError(bounded_error_msg(bounded))
    lower = np.clip(lower, 0.0, np.inf)
    return lower, upper

def confint_2sample_wald(count1, n1, count2, n2, meas1, meas2, conf, bounded):
    alpha = 1 - conf
    r1 = count1 / n1 / meas1
    r2 = count2 / n2 / meas2
    mu = r1 - r2
    sigma = np.sqrt(r1 / n1 / meas1 + r2 / n2 / meas2)
    dist = scipy.stats.norm(mu, sigma)
    if bounded == 'both':
        lower = dist.ppf(alpha / 2)
        upper = dist.ppf(1 - alpha / 2)
    elif bounded == 'below':
        lower = dist.ppf(alpha)
        upper = np.clip(lower, np.inf, np.inf)
    elif bounded == 'above':
        upper = dist.ppf(1 - alpha)
        lower = np.clip(upper, -np.inf, -np.inf)
    else:
        raise ValueError(bounded_error_msg(bounded))
    return lower, upper
