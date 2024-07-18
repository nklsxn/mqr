"""
Hypothesis tests (non-parametric) for distributions.
"""

from mqr.inference.hyptest import HypothesisTest

import scipy
import statsmodels

def test_1sample(x, method='runs'):
    """
    Hypothesis test on randomness of distribution.

    Null-hypothesis: observations in `x` are independent and identically
    distributed.

    Calls `statsmodels.sandbox.stats.runs.runstest_1samp` (statsmodels.org).

    Arguments
    ---------
    x (array[float]) -- Test the hypothesis that `x` is random.

    Optional
    --------
    method (str) -- Type of test. Only the default "runs" test is implemented.

    Returns
    -------
    mqr.confit.HypothesisTest
    """
    import mqr

    if method == 'runs':
        stat, pvalue = statsmodels.sandbox.stats.runs.runstest_1samp(x, cutoff='median', correction=True)
    else:
        raise ValueError(f'method {method} is not available')

    return HypothesisTest(
        description='randomness',
        alternative='two-sided',
        method=method,
        sample_stat='dist(x)',
        sample_stat_target='iid',
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
        method = 'Kolmogorov-Smirnov'
        stat, pvalue = scipy.stats.ks_2samp(
            data1=x,
            data2=y,
            alternative=alternative)
    elif method == 'runs':
        if alternative != 'two-sided':
            raise ValueError('Only "two-sided" alternative available')
        stat, pvalue = statsmodels.sandbox.stats.runs.runstest_2samp(x, y, correction=True)
    else:
        raise NotImplementedError(f'method {method} is not available')

    return HypothesisTest(
        description='sampling distribution',
        alternative=alternative,
        method=method,
        sample_stat='dist(x)',
        sample_stat_target='dist(y)',
        sample_stat_value=None,
        stat=stat,
        pvalue=pvalue)
