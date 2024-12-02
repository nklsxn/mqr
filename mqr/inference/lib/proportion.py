from mqr.utils import clip_where
import mqr.inference.lib.util as util

import numpy as np
import scipy

import warnings

def confint_1sample_agresti_coull(count, nobs, conf, bounded):
    """
    Confidence interval for proportion `count / nobs`.

    Implements the Agresti-Coull interval, which is not recommended for small
    sample sizes (small means nobs < 40), according to [1]_. In those cases [1]_
    recommends Wilson with continuity correction or Jeffreys.

    Parameters
    ----------
    count : int
        Number of "true" observations.
    nobs : int
        Total observations.
    conf : float
        Confidence level that determines the width of the interval.
    bounded : {'both', 'below', 'above'}
        Which sides of the interval to close.

    Returns
    -------
    (float, float)

    References
    ----------
    .. [1]  Brown, L. D. Cai, T. T. and DasGupta, A. (2001).
            "Interval estimation for a binomial proportion".
            Statistical Science, 16(2), 101-133.
    """
    alpha = 1 - conf
    if np.any(nobs < 40):
        msg = (
            'The "agresti-coull" method is recommended when `nobs >= 40`. '
            'Consider using "wilson" or "jeffreys" methods. See [1].')
        warnings.warn(msg)
    dist = scipy.stats.norm()
    if bounded == 'both':
        z = dist.ppf(1 - alpha / 2)
        p = (count + z**2 / 2) / (nobs + z**2)
        n = nobs + z**2
        lower = np.maximum(0.0, p - z * np.sqrt(p * (1 - p) / n))
        upper = np.minimum(1.0, p + z * np.sqrt(p * (1 - p) / n))
    elif bounded == 'below':
        z = dist.ppf(1 - alpha)
        p = (count + z**2 / 2) / (nobs + z**2)
        n = nobs + z**2
        lower = np.maximum(0.0, p - z * np.sqrt(p * (1 - p) / n))
        upper = np.clip(lower, 1.0, 1.0)
    elif bounded == 'above':
        z = dist.ppf(1 - alpha)
        p = (count + z**2 / 2) / (nobs + z**2)
        n = nobs + z**2
        upper = np.minimum(1.0, p + z * np.sqrt(p * (1 - p) / n))
        lower = np.clip(upper, 0.0, 0)
    else:
        raise ValueError(util.bounded_error_msg(bounded))
    return lower, upper

def confint_1sample_jeffreys(count, nobs, conf, bounded):
    """
    Confidence interval for proportion `count / nobs`.

    Implements the Jeffreys interval, which is a bayesian method that also has
    desirable frequentist properties (see [1]_).

    Parameters
    ----------
    count : int
        Number of "true" observations.
    nobs : int
        Total observations.
    conf : float
        Confidence level that determines the width of the interval.
    bounded : {'both', 'below', 'above'}
        Which sides of the interval to close.

    Returns
    -------
    (float, float)

    References
    ----------
    .. [1]  Brown, L. D. Cai, T. T. and DasGupta, A. (2001).
            "Interval estimation for a binomial proportion".
            Statistical Science, 16(2), 101-133.
    """
    alpha = 1 - conf
    dist = scipy.stats.beta(count + 1/2, nobs - count + 1/2)
    if bounded == 'both':
        lower = dist.ppf(alpha / 2)
        upper = dist.ppf(1 - alpha / 2)
    elif bounded == 'below':
        lower = dist.ppf(alpha)
        upper = np.clip(lower, 1.0, 1.0)
    elif bounded == 'above':
        upper = dist.ppf(1 - alpha)
        lower = np.clip(upper, 0.0, 0.0)
    else:
        raise ValueError(util.bounded_error_msg(bounded))
    lower = clip_where(lower, 0.0, 0.0, np.isclose(count, 0))
    upper = clip_where(upper, 1.0, 1.0, np.isclose(count, nobs))
    return lower, upper

def confint_1sample_wilson(count, nobs, conf, bounded):
    """
    Confidence interval for proportion `count / nobs`.

    Implements the Wilson method without continuity correction (method 3 in [1]_).

    Parameters
    ----------
    count : int
        Number of "true" observations.
    nobs : int
        Total observations.
    conf : float
        Confidence level that determines the width of the interval.
    bounded : {'both', 'below', 'above'}
        Which sides of the interval to close.

    Returns
    -------
    (float, float)

    References
    ----------
    .. [1]  Newcombe, R. G. (1998).
            Two‐sided confidence intervals for the single proportion:
            comparison of seven methods.
            Statistics in medicine, 17(8), 857-872.
    """
    alpha = 1 - conf
    dist = scipy.stats.norm()
    p = count / nobs

    if bounded == 'both':
        z = dist.ppf(1 - alpha / 2)
        centre = p + z**2 / 2 / nobs
        width = z * np.sqrt(4 * nobs * p * (1 - p) + z**2) / 2 / nobs
        scale = 1 + z**2 / nobs
        lower = (centre - width) / scale
        upper = (centre + width) / scale
    elif bounded == 'below':
        z = dist.ppf(1 - alpha)
        centre = p + z**2 / 2 / nobs
        width = z * np.sqrt(4 * nobs * p * (1 - p) + z**2) / 2 / nobs
        scale = 1 + z**2 / nobs
        lower = (centre - width) / scale
        upper = np.clip(lower, 1.0, 1.0)
    elif bounded == 'above':
        z = dist.ppf(1 - alpha)
        centre = p + z**2 / 2 / nobs
        width = z * np.sqrt(4 * nobs * p * (1 - p) + z**2) / 2 / nobs
        scale = 1 + z**2 / nobs
        upper = (centre + width) / scale
        lower = np.clip(upper, 0.0, 0.0)
    else:
        raise ValueError(util.bounded_error_msg(bounded))
    lower = clip_where(lower, 0.0, 0.0, np.isclose(p, 0))
    upper = clip_where(upper, 1.0, 1.0, np.isclose(p, 1))
    return lower, upper

def confint_1sample_wilson_cc(count, nobs, conf, bounded):
    """
    Confidence interval for proportion `count / nobs`.

    Implements the Wilson method with continuity correction (method 4 in [1]_).

    Parameters
    ----------
    count : int
        Number of "true" observations.
    nobs : int
        Total observations.
    conf : float
        Confidence level that determines the width of the interval.
    bounded : {'both', 'below', 'above'}
        Which sides of the interval to close.

    Returns
    -------
    (float, float)

    References
    ----------
    .. [1]  Newcombe, R. G. (1998).
            Two‐sided confidence intervals for the single proportion:
            comparison of seven methods. Statistics in medicine, 17(8), 857-872.
    """
    alpha = 1 - conf
    dist = scipy.stats.norm()
    p = count / nobs
    if bounded == 'both':
        z = dist.ppf(1 - alpha / 2)
        a = 2 * nobs * p + z**2
        b_lower = 1 + z * np.sqrt(z**2 + 4 * nobs * p * (1 - p) - 1 / nobs + 4 * p - 2)
        b_upper = 1 + z * np.sqrt(z**2 + 4 * nobs * p * (1 - p) - 1 / nobs - 4 * p + 2)
        lower = np.maximum(0, (a - b_lower) / 2 / (nobs + z**2))
        upper = np.minimum(1, (a + b_upper) / 2 / (nobs + z**2))
    elif bounded == 'below':
        z = dist.ppf(1 - alpha)
        a = 2 * nobs * p + z**2
        b_lower = 1 + z * np.sqrt(z**2 + 4 * nobs * p * (1 - p) - 1 / nobs + 4 * p - 2)
        lower = np.maximum(0, (a - b_lower) / 2 / (nobs + z**2))
        upper = np.clip(lower, 1.0, 1.0)
    elif bounded == 'above':
        z = dist.ppf(1 - alpha)
        a = 2 * nobs * p + z**2
        b_upper = 1 + z * np.sqrt(z**2 + 4 * nobs * p * (1 - p) - 1 / nobs - 4 * p + 2)
        upper = np.minimum(1, (a + b_upper) / 2 / (nobs + z**2))
        lower = np.clip(upper, 0.0, 0.0)
    else:
        raise ValueError(util.bounded_error_msg(bounded))
    lower = clip_where(lower, 0.0, 0.0, np.isclose(p, 0))
    upper = clip_where(upper, 1.0, 1.0, np.isclose(p, 1))
    return lower, upper

def confint_2sample_agresti_caffo(count1, nobs1, count2, nobs2, conf, bounded):
    """
    Confidence interval for difference between proportions `count1 / nobs1 - count2 / nobs2`.

    Implements the Agresti-Caffo method. Note that this method performs well
    over most parameters, but is not optimal over all parameters. For details
    see ([1]_).

    Parameters
    ----------
    count1 : int
        Number of "true" observations in first sample.
    nobs1 : int
        Total observations in first sample.
    count2 : int
        Number of "true" observations in second sample.
    nobs2 : int
        Total observations in second sample.
    conf : float
        Confidence level that determines the width of the interval.
    bounded : {'both', 'below', 'above'}
        Which sides of the interval to close.

    Returns
    -------
    :class:`mqr.inference.confint.ConfidenceInterval`

    References
    ----------
    .. [1]  Agresti, A., & Caffo, B. (2000).
            Simple and effective confidence intervals for proportions and differences
            of proportions result from adding two successes and two failures.
            The American Statistician, 54(4), 280-288.
    """
    alpha = 1 - conf
    p1 = (count1 + 1) / (nobs1 + 2)
    p2 = (count2 + 1) / (nobs2 + 2)
    mu = p1 - p2
    sigma = np.sqrt(p1 * (1 - p1) / (nobs1 + 2) + p2 * (1 - p2) / (nobs2 + 2))
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
        raise ValueError(util.bounded_error_msg(bounded))
    return lower, upper

def confint_2sample_newcomb(count1, nobs1, count2, nobs2, conf, bounded):
    """
    Confidence interval for difference between proportions `count1 / nobs1 - count2 / nobs2`.

    Implements method 10 in [1]_.

    Parameters
    ----------
    count1 : int
        Number of "true" observations in first sample.
    nobs1 : int
        Total observations in first sample.
    count2 : int
        Number of "true" observations in second sample.
    nobs2 : int
        Total observations in second sample.
    conf : float
        Confidence level that determines the width of the interval.
    bounded : {'both', 'below', 'above'}
        Which sides of the interval to close.

    Returns
    -------
    :class:`mqr.inference.confint.ConfidenceInterval`

    References
    ----------
    .. [1]  Newcombe, R. G. (1998).
            Interval estimation for the difference between independent proportions:
            comparison of eleven methods.
            Statistics in medicine, 17(8), 873-890.
    """
    alpha = 1 - conf
    theta_hat = count1 / nobs1 - count2 / nobs2
    l1, u1 = confint_1sample_wilson(count1, nobs1, conf, bounded)
    l2, u2 = confint_1sample_wilson(count2, nobs2, conf, bounded)

    if bounded == 'both':
        z = scipy.stats.norm().ppf(1 - alpha / 2)
        delta = z * np.sqrt(l1 * (1 - l1) / nobs1 + u2 * (1 - u2) / nobs2)
        epsilon = z * np.sqrt(u1 * (1 - u1) / nobs1 + l2 * (1 - l2) / nobs2)
        lower = theta_hat - delta
        upper = theta_hat + epsilon
    elif bounded == 'below':
        z = scipy.stats.norm().ppf(1 - alpha)
        delta = z * np.sqrt(l1 * (1 - l1) / nobs1 + u2 * (1 - u2) / nobs2)
        lower = theta_hat - delta
        upper = np.clip(lower, np.inf, np.inf)
    elif bounded == 'above':
        z = scipy.stats.norm().ppf(1 - alpha)
        epsilon = z * np.sqrt(u1 * (1 - u1) / nobs1 + l2 * (1 - l2) / nobs2)
        upper = theta_hat + epsilon
        lower = np.clip(upper, -np.inf, -np.inf)
    else:
        raise ValueError(util.bounded_error_msg(bounded))
    return lower, upper

def confint_2sample_newcomb_cc(count1, nobs1, count2, nobs2, conf, bounded):
    """
    Confidence interval for difference between proportions `count1 / nobs1 - count2 / nobs2`.

    Implements method 11 in [1]_. Further discussion and comparisons in [2]_.

    Parameters
    ----------
    count1 : int
        Number of "true" observations in first sample.
    nobs1 : int
        Total observations in first sample.
    count2 : int
        Number of "true" observations in second sample.
    nobs2 : int
        Total observations in second sample.
    conf : float
        Confidence level that determines the width of the interval.
    bounded : {'both', 'below', 'above'}
        Which sides of the interval to close.

    Returns
    -------
    :class:`mqr.inference.confint.ConfidenceInterval`

    References
    ----------
    .. [1]  Newcombe, R. G. (1998).
            Interval estimation for the difference between independent proportions:
            comparison of eleven methods.
            Statistics in medicine, 17(8), 873-890.
    .. [2]  Agresti, A., & Caffo, B. (2000).
            Simple and effective confidence intervals for proportions and differences
            of proportions result from adding two successes and two failures.
            The American Statistician, 54(4), 280-288.
    """
    p1 = count1 / nobs1
    p2 = count2 / nobs2
    if bounded == 'both':
        lower1, upper1 = confint_1sample_wilson_cc(count1, nobs1, conf, bounded)
        lower2, upper2 = confint_1sample_wilson_cc(count2, nobs2, conf, bounded)
        lower = (p1 - p2) - np.sqrt((p1 - lower1)**2 + (upper2 - p2)**2)
        upper = (p1 - p2) + np.sqrt((upper1 - p1)**2 + (p2 - lower2)**2)
    elif bounded == 'below':
        lower1, _ = confint_1sample_wilson_cc(count1, nobs1, conf, 'below')
        _, upper2 = confint_1sample_wilson_cc(count2, nobs2, conf, 'above')
        lower = (p1 - p2) - np.sqrt((p1 - lower1)**2 + (upper2 - p2)**2)
        upper = np.clip(lower, np.inf, np.inf)
    elif bounded == 'above':
        _, upper1 = confint_1sample_wilson_cc(count1, nobs1, conf, 'above')
        lower2, _ = confint_1sample_wilson_cc(count2, nobs2, conf, 'below')
        upper = (p1 - p2) + np.sqrt((upper1 - p1)**2 + (p2 - lower2)**2)
        lower = np.clip(upper, -np.inf, -np.inf)
    else:
        raise ValueError(util.bounded_error_msg(bounded))
    return lower, upper
