"""
Confidence intervals and hypothesis tests (parametric) for correlation.
"""

from mqr.inference.confint import ConfidenceInterval
from mqr.inference.hyptest import HypothesisTest

import numpy as np
import scipy

# From 100 statistical tests, Gopal Kanji

def confint(x, y, conf=0.95, bounded='both', method='fisher-z'):
    """
    Confidence interval on correlation (using the Pearson correlation coefficient).

    Uses Fisher's z-transform, see [1].

    Arguments
    ---------
    x, y (array[float]) -- Calculate correlation between these samples.

    Optional
    --------
    conf (float) -- Confidence level that determines the width of the interval.
        (Default 0.95.)
    bounded (str) -- Which sides of the interval to close (default "both"). One
        of "both", "above" or "below.
    method (str) -- Type of statistic. Only the default, "fisher-z" Fisher's
        z-transform, is implemented.

    Returns
    -------
    mqr.confint.ConfidenceInterval

    References
    ----------
    [1] Asuero, A. G., Sayago, A., & González, A. G. (2006).
        The correlation coefficient: An overview.
        Critical reviews in analytical chemistry, 36(1), 41-59.
    """
    if method != 'fisher-z':
        raise NotImplemented('Only `fisher-z` method is implemented.')

    if len(x) != len(y):
        raise ValueError(f'Lengths of x and y must be equal.')

    alpha = 1 - conf
    r = scipy.stats.pearsonr(x, y).statistic
    n = len(x)

    corr_mu = _fisher_z(r)
    corr_var = 1 / (n - 3)
    dist = scipy.stats.norm(corr_mu, np.sqrt(corr_var))

    if bounded == 'both':
        lower_z, upper_z = dist.ppf(np.array([alpha/2, 1-alpha/2]))
        lower, upper = _inv_fisher_z(np.array([lower_z, upper_z]))
    elif bounded == 'above':
        lower = -1.0
        upper = _inv_fisher_z(dist.ppf(1-alpha))
    elif bounded == 'below':
        lower = _inv_fisher_z(dist.ppf(alpha))
        upper = 1.0
    else:
        raise ValueError(f'Invalid alternative "{alternative}". Use "two-sided" (default), "less", or "greater".')

    return ConfidenceInterval(
        name='correlation',
        value=r,
        lower=lower,
        upper=upper,
        conf=conf)

def test(x, y, H0_corr=0.0, alternative='two-sided'):
    """
    Hypothesis test for correlation (Pearson) between two samples.

    Null-hypothesis: `corr(x, y) == H0_corr`.

    Uses Fisher's z-transform, see [1].

    Arguments
    ---------
    x, y (array[float]) -- Test correlation of these two equal-length samples.

    Optional
    --------
    H0_corr (float) -- Null-hypothesis correlation coefficient. (Default 0.)
    alternative (str) -- Sense of alternative hypothesis. One of "two-sided",
        "less" or "greater". (Default "two-sided".)
    method (str) -- Type of statistic; only the default "fisher-z" is implemented.

    Returns
    -------
    mqr.hyptest.HypothesisTest

    Raises
    ------
    ValueError -- Length of x and y are different.

    References
    ----------
    [1] Asuero, A. G., Sayago, A., & González, A. G. (2006).
        The correlation coefficient: An overview.
        Critical reviews in analytical chemistry, 36(1), 41-59.
    """
    if len(x) != len(y):
        raise ValueError(f'Lengths of x and y must be equal.')

    r = scipy.stats.pearsonr(x, y).statistic
    n = len(x)
    if np.isclose(H0_corr, 0.0):
        method = 'pearson'
        dist = scipy.stats.t(n-2)
        stat = r * np.sqrt((n - 2) / (1 - np.square(r)))
    else:
        method = 'fisher-z'
        mu = _fisher_z(H0_corr)
        var = 1 / (n - 3)
        dist = scipy.stats.norm(mu, np.sqrt(var))
        stat = _fisher_z(r)

    if alternative == 'two-sided':
        pvalue = 2 * np.minimum(dist.cdf(stat), 1-dist.cdf(stat))
    elif alternative == 'less':
        pvalue = dist.cdf(stat)
    elif alternative == 'greater':
        pvalue = 1 - dist.cdf(stat)
    else:
        raise ValueError(f'Invalid alternative "{alternative}". Use "two-sided" (default), "less", or "greater".')

    x_name = x.name if hasattr(x, 'name') else 'x'
    y_name = y.name if hasattr(y, 'name') else 'y'
    return HypothesisTest(
        description='correlation coefficient',
        alternative=alternative,
        method=method,
        sample_stat=f'corr({x_name}, {y_name})',
        sample_stat_target=H0_corr,
        sample_stat_value=r,
        stat=stat,
        pvalue=pvalue,)

def test_diff(x1, y1, x2, y2, H0_corr1=0.0, H0_corr2=0.0, alternative='two-sided', method='fisher-z'):
    """
    Hypothesis test on the difference of two correlations (Pearson).

    Null-hypothesis: `corr(x1, y1) - corr(x2, y2) == H0_corr1 - H0_corr2`.
    Note that the alternatives mean "less" or "greater" on the real number line,
    not in absolute value.

    Uses Fisher's z-transform, see [1].

    Arguments
    ---------
    x1, y1 (array[float]) -- First pair of samples.
    x2, y2 (array[float]) -- Second pair of samples whose correlation will be
        subtracted from the correlation of the first pair to form the difference.

    Optional
    --------
    H0_corr1, H0_corr2 (float) -- Null-hypothesis difference is
        `H0_corr1 - H0_corr2`. (Defaults 0 and 0.)
    alternative (str) -- Sense of alternative hypothesis. One of "two-sided",
        "less" or "greater". (Default "two-sided".)
    method (str) -- Type of statistic; only the default "fisher-z" is supported.

    Returns
    -------
    mqr.hyptest.HypothesisTest

    Raises
    ------
    ValueError -- Lengths of x1 and y1 are different.
               -- Lengths of x2 and y2 are different.

    References
    ----------
    [1] Asuero, A. G., Sayago, A., & González, A. G. (2006).
        The correlation coefficient: An overview.
        Critical reviews in analytical chemistry, 36(1), 41-59.
    """
    if method != 'fisher-z':
        raise ValueError(f'method "{method}" is not available')

    if len(x1) != len(y1):
        raise ValueError('Lengths of x1 and y1 must be equal.')
    if len(x2) != len(y2):
        raise ValueError('Lengths of x2 and y2 must be equal.')

    n1 = len(x1)
    n2 = len(x2)
    mu1 = _fisher_z(H0_corr1)
    mu2 = _fisher_z(H0_corr2)
    var1 = 1 / (n1 - 3)
    var2 = 1 / (n2 - 3)
    mu = mu1 - mu2
    var = var1 + var2
    dist = scipy.stats.norm(mu, np.sqrt(var))

    r1 = np.corrcoef(x1, y1)[0, 1]
    r2 = np.corrcoef(x2, y2)[0, 1]
    Z1 = _fisher_z(r1)
    Z2 = _fisher_z(r2)
    Z = Z1 - Z2
    stat = Z

    if alternative == 'two-sided':
        pvalue = 2 * np.minimum(dist.cdf(stat), 1-dist.cdf(stat))
    elif alternative == 'less':
        pvalue = dist.cdf(stat)
    elif alternative == 'greater':
        pvalue = 1 - dist.cdf(stat)
    else:
        raise ValueError(f'Invalid alternative "{alternative}". Use "two-sided" (default), "less", or "greater".')

    x1_name = x1.name if hasattr(x1, 'name') else 'x1'
    y1_name = y1.name if hasattr(y1, 'name') else 'y1'
    x2_name = x2.name if hasattr(x2, 'name') else 'x2'
    y2_name = y2.name if hasattr(y2, 'name') else 'y2'
    return HypothesisTest(
        description='difference between correlation coefficients',
        alternative=alternative,
        method=method,
        sample_stat=f'corr({x1_name}, {y1_name}) - corr({x2_name}, {y2_name})',
        sample_stat_target=H0_corr1-H0_corr2,
        sample_stat_value=r1-r2,
        stat=stat,
        pvalue=pvalue,)

def _fisher_z(r):
    return np.arctanh(r)

def _inv_fisher_z(z):
    return np.tanh(z)
