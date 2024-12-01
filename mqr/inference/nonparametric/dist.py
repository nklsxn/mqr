"""
Hypothesis tests (non-parametric) for distributions.
"""

from mqr.inference.hyptest import HypothesisTest

import mqr.inference.lib.util as util

import scipy
import statsmodels

def test_1sample(x, method='runs', cutoff='median'):
    """
    Hypothesis test on randomness characteristics of a sample.

    Null-hypotheses:
        method "runs" -- observations in `x` are independent and identically distributed.
        method "adf" -- observations in `x` are non-stationary (time series has a unit root).

    Calls `statsmodels.sandbox.stats.runs.runstest_1samp` (statsmodels.org).

    Arguments
    ---------
    x (array[float]) -- Test the hypothesis that `x` is random.

    Optional
    --------
    method (str) -- Type of test (default "runs"). One of:
        "runs" the runs test for independent and identical distribution,
        "adf" the Augmented Dickey-Fuller test for unit root (stationarity).
            This test is used when x is a time-series. Note the sense: a small
            p-value indicates a stationary processes.
    cutoff (str) -- The cutoff to group large and small values.

    Returns
    -------
    mqr.confit.HypothesisTest
    """
    if method == 'runs':
        description = 'randomness'
        sample_stat_target = 'iid'
        stat, pvalue = statsmodels.sandbox.stats.runs.runstest_1samp(x, cutoff=cutoff, correction=True)
    else:
        raise ValueError(util.method_error_msg(method, ['runs']))

    return HypothesisTest(
        description=description,
        alternative='two-sided',
        method=method,
        sample_stat='dist(x)',
        sample_stat_target=sample_stat_target,
        sample_stat_value=None,
        stat=stat,
        pvalue=pvalue,)

def test_2sample(x, y, alternative='two-sided', method='ks'):
    """
    Hypothesis test for distributions of two samples.

    Null-hypothesis: cdf(x) == cdf(y)

    Arguments
    ---------
    x, y (array[float]) -- Samples to compare.

    Optional
    --------
    alternative (str) -- Sense of alternative hypothesis. One of "two-sided",
        "less" or "greater". (Default "two-sided".)
    method (str) -- Type of test (default "ks"). One of
        "ks" for Kolmogorov-Smirnov (`scipy.stats.kstest`, scipy.org),
        "runs" for Wald-Wolfowitz (`statsmodels.sandbox.stats.runs.runstest_2samp`,
        statsmodels.org).

    Returns
    -------
    mqr.confit.HypothesisTest
    """
    if method == 'ks':
        method = 'kolmogorov-smirnov'
        stat, pvalue = scipy.stats.ks_2samp(
            data1=x,
            data2=y,
            alternative=alternative)
    elif method == 'runs':
        if alternative != 'two-sided':
            raise ValueError('Only "two-sided" alternative available.')
        stat, pvalue = statsmodels.sandbox.stats.runs.runstest_2samp(x, y, correction=True)
    else:
        raise ValueError(util.method_error_msg(method, ['ks', 'runs']))

    return HypothesisTest(
        description='sampling distribution',
        alternative=alternative,
        method=method,
        sample_stat='dist(x)',
        sample_stat_target='dist(y)',
        sample_stat_value=None,
        stat=stat,
        pvalue=pvalue)
