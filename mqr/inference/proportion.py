"""
Confidence intervals and hypothesis tests (parametric) for proportions.
"""

from mqr.inference.confint import ConfidenceInterval
from mqr.inference.hyptest import HypothesisTest
from mqr.inference.power import TestPower

import mqr.interop.inference as interop

import numpy as np
import scipy
import statsmodels

def size_1sample(p0, pa, alpha, beta, alternative='two-sided'):
    """
    Calculate sample size for test of proportion.

    Uses `statsmodels.stats.proportion.power_proportions_2indep` (statsmodels.org).

    Arguments
    ---------
    p0 (float) -- Null proportion.
    pa (float) -- Alternative proportion, forming the effect size with `p0`.
    alpha (float) -- Required significance.
    beta (float) -- Required beta (1 - power).

    Optional
    --------
    alternative (float) -- Sense of alternative hypothesis. One of "two-sided",
        "less" or "greater". (Default "two-sided".)

    Returns
    -------
    mqr.power.TestPower
    """
    alt = interop.alternative(alternative, lib='statsmodels')
    def power_fn(nobs):
        power = statsmodels.stats.proportion.power_proportions_2indep(
            diff=p0-pa,
            prop2=pa,
            nobs1=nobs,
            ratio=1,
            alpha=alpha,
            value=0,
            alternative=alt,
            return_results=False)
        return 1 - power - beta
    nobs_opt = scipy.optimize.fsolve(power_fn, 0)[0]

    return TestPower(
        name='proportion',
        alpha=alpha,
        beta=beta,
        effect=f'{p0:g} - {pa:g} = {p0-pa:g}',
        alternative=alternative,
        method='z',
        sample_size=nobs_opt)

def size_2sample(p1, p2, alpha, beta, alternative='two-sided'):
    """
    Calculate sample size to test equality of two proportions.

    Equivalent to `mqr.inference.proportion.size_1sample(p1, p2, ...)`.

    Arguments
    ---------
    p1 (float) -- First proportion.
    p2 (float) -- Second propotion, forming effect size with `p1`.
    alpha (float) -- Required significance.
    beta (float) -- Required beta (1 - power).

    Optional
    --------
    alternative (float) -- Sense of alternative hypothesis. One of "two-sided",
        "less" or "greater". (Default "two-sided".)

    Returns
    -------
    mqr.power.TestPower
    """
    res = size_1sample(p1, p2, alpha, beta, alternative=alternative)
    res.name = 'difference between proportions'
    return res

def confint_1sample(count, nobs, conf=0.95, method='beta'):
    """
    Confidence interval for proportion `count / nobs`.

    Calls `statsmodels.stats.proportion.proportion_confint` (statsmodels.org).

    Arguments
    ---------
    count (int) -- Number of "true" observations.
    nobs (int) -- Total observations.

    Optional
    --------
    conf (float) -- Confidence level that determines the width of the interval.
    method (str) -- Statistical test to use (default "beta"). One of:
        "normal",
        "agresti_coull",
        "beta" (default),
        "wilson",
        "binom_test".

    Returns
    -------
    mqr.confint.ConfidenceInterval
    """
    alpha = 1 - conf
    (lower, upper) = statsmodels.stats.proportion.proportion_confint(
        count=count,
        nobs=nobs,
        alpha=alpha,
        method=method)
    value = count / nobs

    return ConfidenceInterval(
        name='proportion',
        value=value,
        lower=lower,
        upper=upper,
        conf=conf)

def confint_2sample(count1, nobs1, count2, nobs2, conf=0.95, method='newcomb'):
    """
    Confidence interval for difference between proportions
    `count1 / nobs1 - count2 / nobs2`.

    Calls `statsmodels.stats.proportion.confint_proportions_2indep` (statsmodels.org).

    Arguments
    ---------
    count1 (int) -- Number of "true" observations in first sample.
    nobs1 (int) -- Total observations in first sample.
    count2 (int) -- Number of "true" observations in second sample.
    nobs2 (int) -- Total observations in second sample.

    Optional
    --------
    conf (float) -- Confidence level that determines the width of the interval.
        (Default 0.95.)
    method (str) -- Statistical test to use (default "wald"). See statsmodels
        docs for others.

    Returns
    -------
    mqr.confint.ConfidenceInterval
    """
    value = count1 / nobs1 - count2 / nobs2
    lower, upper = statsmodels.stats.proportion.confint_proportions_2indep(
        count1, nobs1,
        count2, nobs2,
        alpha=1-conf,
        compare='diff',
        method=method)
    return ConfidenceInterval(
        name='difference between proportions',
        value=value,
        lower=lower,
        upper=upper,
        conf=conf)

def test_1sample(count, nobs, H0_prop, alternative='two-sided', method='binom'):
    """
    Hypothesis test for the proportion of "true" elements in a sample.

    Null-hypothesis: `count / nobs == H0_prop`.

    Arguments
    ---------
    count (int) -- Number of "true" observations.
    nobs (int) -- Total number of observations.
    H0_prop (float) -- Null-hypothesis proportion.

    Optional
    --------
    alternative (str) -- Sense of alternative hypothesis. One of "two-sided",
        "less" or "greater". Valid for methods "binom" and "chi2".
        (Default "two-sided".)
    method (str) -- Type of test (default "binom"). One of
        "binom" (`statsmodels.stats.proportion.binom_test`, statsmodels.org),
        "chi2" (`statsmodels.stats.proportion.proportions_chisquare`, statsmodels.org),
        "z" (`statsmodels.stats.proportions_ztest`, statsmodels.org).

    Returns
    -------
    mqr.hyptest.HypothesisTest
    """
    alt = interop.alternative(alternative, lib='statsmodels')
    if method == 'binom':
        pvalue = statsmodels.stats.proportion.binom_test(
            count=count,
            nobs=nobs,
            prop=H0_prop,
            alternative=alt)
        stat = np.nan
    elif method == 'z':
        stat, pvalue = statsmodels.stats.proportion.proportions_ztest(
            count=count,
            nobs=nobs,
            value=H0_prop,
            alternative=alt)
    elif method == 'chi2':
        if alternative != 'two-sided':
            raise ValueError(f'invalid alternative "{alternative}"')
        stat, pvalue, _ = statsmodels.stats.proportion.proportions_chisquare(
            count=count,
            nobs=nobs,
            value=H0_prop)
    else:
        raise ValueError(f'method "{method}" is not available')

    return HypothesisTest(
        description='proportion of "true" elements',
        alternative=alternative,
        method=method,
        sample_stat=f'count / nobs',
        sample_stat_target=H0_prop,
        sample_stat_value=count/nobs,
        stat=stat,
        pvalue=pvalue,)

def test_2sample(count1, nobs1, count2, nobs2, H0_diff, alternative='two-sided', method='agresti-caffo'):
    """
    Hypothesis test for the difference between proportions of two samples.

    Null-hypothesis: `count1 / nobs1 - count2 / nobs2 == H0_diff`.

    Calls `statsmodels.stats.proportion.test_proportions_2indep` (statsmodels.org).

    Arguments
    ---------
    count1 (int) -- Number of "true" observations in first sample.
    nobs1 (int) -- Total number of observations in second sample.
    count2 (int) -- Number of "true" observations in second sample.
    nobs2 (int) -- Total number of observations in seconds sample.
    H0_diff (float) -- Null-hypothesis difference.

    Optional
    --------
    alternative (str) -- Sense of alternative hypothesis. One of "two-sided",
        "less" or "greater". (Default "two-sided".)
    method (str) -- Type of test (default "agresti-caffo"). One of
        "wald",
        "agresti-caffo", or
        "score".

    Returns
    -------
    mqr.hyptest.HypothesisTest
    """
    alt = interop.alternative(alternative, 'statsmodels')
    res = statsmodels.stats.proportion.test_proportions_2indep(
        count1, nobs1,
        count2, nobs2,
        alternative=alt,
        value=H0_diff,
        method=method)

    return HypothesisTest(
        description='difference between proportions of "true" elements',
        alternative=alternative,
        method=res.method,
        sample_stat=f'count1 / nobs1 - count2 / nobs2',
        sample_stat_target=H0_diff,
        sample_stat_value=count1/nobs1-count2/nobs2,
        stat=res.statistic,
        pvalue=res.pvalue,)
