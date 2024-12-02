from mqr.inference.hyptest import HypothesisTest

import mqr.inference.lib.util as util

import scipy
import statsmodels

def test_1sample(x, method='runs', cutoff='median'):
    """
    Hypothesis test on randomness characteristics of a sample.

    Null-hypotheses by method
        | 'runs'
        |   observations in `x` are independent and identically distributed
        | 'adf'
        |   observations in `x` are non-stationary (time series has a unit root)

    Parameters
    ----------
    x : array_like
        Test the hypothesis that `x` is random.
    method : {'runs'}, optional
        | 'runs'
        |   Runs test for independent and identical distribution.
            Calls :func:`sm..runstest_1samp <statsmodels.sandbox.stats.runs.runstest_1samp>`.
    cutoff : str, optional
        The cutoff to group large and small values. Only used for method "runs".

    Returns
    -------
    :class:`mqr.inference.hyptest.HypothesisTest`
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

    Null-hypothesis
        cdf(`x`) == cdf(`y`)

    Parameters
    ----------
    x, y : array_like
        Samples to compare.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Sense of alternative hypothesis.
    method : {'ks', 'runs'}, optional
        | 'ks'
        |   Kolmogorov-Smirnov test.
            Calls :func:`scipy..kstest <scipy.stats.kstest>`.
        | 'runs'
        |   Wald-Wolfowitz test
            Calls :func:`sm..runstest_2samp <statsmodels.sandbox.stats.runs.runstest_2samp>`.

    Returns
    -------
    :class:`mqr.inference.hyptest.HypothesisTest`
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
