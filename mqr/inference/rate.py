"""
Confidence intervals and hypothesis tests (parametric) for rates of events.
"""

from mqr.inference.confint import ConfidenceInterval
from mqr.inference.hyptest import HypothesisTest
from mqr.inference.power import TestPower

import mqr.interop.inference as interop

import numpy as np
import scipy
import statsmodels

def size_1sample(ra, hyp_rate, alpha, beta, alternative='two-sided'):
    """
    Calculate sample size for test of rate of events.

    Null-hypothesis: `ra / hyp_rate == 1`.

    Arguments
    ---------
    ra (float) -- Alternative hypothesis rate, forming effect size.
    hyp_rate (float) -- Null-hypothesis rate.
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
    n = 1
    r = ra / hyp_rate
    if alternative == 'less' or alternative == 'greater':
        NP1 = 1 - beta
        DP1 = alpha
    elif alternative == 'two-sided':
        NP1 = 1 - beta
        DP1 = alpha / 2
    else:
        raise ValueError(f'Invalid alternative "{alternative}". Use "two-sided" (default), "less", or "greater".')

    def ratio(n):
        num = scipy.stats.chi2.ppf(NP1, df=n)
        den = scipy.stats.chi2.ppf(DP1, df=n)
        return num / den

    while n < 10000:
        if ratio(n) < r: 
            break
        n += 1
    if n == 10000:
        raise ValueError('iteration stopped at n = 10000, without sufficient power')

    num = scipy.stats.chi2.ppf(NP1, df=n)
    nobs = num / 2.0 / np.maximum(ra, hyp_rate)

    return TestPower(
        name='rate of events',
        alpha=alpha,
        beta=beta,
        effect=f'{ra:g} / {hyp_rate:g} = {r:g}',
        alternative=alternative,
        method='chi2',
        sample_size=nobs,)

def size_2sample(r1, r2, alpha, beta, effect=0.0, alternative='two-sided'):
    """
    Calculate sample size for difference of two rates of events.

    Null-hypothesis: `r1 - r2 == effect`.

    Uses `statsmodels.stats.rates.power_poisson_diff_2indep` (statsmodels.org).

    Arguments
    ---------
    r1 (float) -- First rate.
    r2 (float) -- Second rate.
    alpha (float) -- Required significance.
    beta (float) -- Required beta (1 - power).

    Optional
    --------
    effect (float) -- Required effect size. (Default 0.0).
    alternative (float) -- Sense of alternative hypothesis. One of "two-sided",
        "less" or "greater". (Default "two-sided".)

    Returns
    -------
    mqr.power.TestPower
    """
    alt = interop.alternative(alternative, lib='statsmodels')
    def beta_fn(nobs):
        power = statsmodels.stats.rates.power_poisson_diff_2indep(
            rate1=r1,
            rate2=r2,
            nobs1=nobs,
            nobs_ratio=1,
            value=effect,
            alpha=alpha,
            alternative=alt,
            return_results=False)
        return 1 - power - beta
    nobs_opt = scipy.optimize.fsolve(beta_fn, 0)[0]

    return TestPower(
        name='difference between rates of events',
        alpha=alpha,
        beta=beta,
        effect=f'{r1:g} - {r2:g} = {effect:g}',
        alternative=alternative,
        method=None,
        sample_size=nobs_opt)

def confint_1sample(count, n, meas=1.0, conf=0.95, method='exact-c'):
    """
    Confidence interval for rate `count / n / meas`.

    Calls `statsmodels.stats.rates.confint_poisson` (statsmodels.org).

    Arguments
    ---------
    count (int) -- Number of events.
    n (int) -- Number of periods over which events were counted.

    Optional
    --------
    meas (float) -- Extent of one period of observation. (Default 1.0.)
    conf (float) -- Confidence level that determines the width of the interval.
        (Default 0.95.)
    method (str) -- Test method (default "exact-c"). See docs for
        `statsmodels.stats.rates.confint_poisson`.

    Returns
    -------
    mqr.confint.ConfidenceInterval
    """
    value = count / n / meas
    alpha = 1 - conf
    (lower, upper) = statsmodels.stats.rates.confint_poisson(
        count=count,
        exposure=n*meas,
        method=method,
        alpha=alpha)

    return ConfidenceInterval(
        name='rate of events',
        value=value,
        lower=lower,
        upper=upper,
        conf=conf)

def confint_2sample(count1, n1, count2, n2, meas1=1.0, meas2=1.0,
                    conf=0.95, method='wald', compare='diff'):
    """
    Confidence interval for:
    - difference of rates `count1 / n1 / meas1 - count2 / n2 / meas2`,
        if `compare` is "diff", or
    - ratio of rates `count1 / n1 / meas1 / (count2 / n2 / meas2)`,
        if `compare` is "ratio".

    Calls `statsmodels.stats.rates.confint_poisson_2indep` (statsmodels.org).

    Arguments
    ---------
    count1 (int) -- Number of events in first observation.
    n1 (int) -- Number of periods over which first events were counted.
    count2 (int) -- Number of events in second observation.
    n2 (int) -- Number of periods over which second events were counted.

    Optional
    --------
    meas1 (float) -- Extent of one period in first observation. (Default 1.)
    meas2 (float) -- Extent of one period in second observation. (Default 1.)
    conf (float) -- Confidence level that determines the width of the interval.
        (Default 0.95.)
    method (str) -- Test method, default "wald". See statsmodels docs for more.
        (Default "wald".)
    compare (str) -- Null-hypothesis: either "diff" or "ratio". (Default "diff".)

    Returns
    -------
    mqr.confint.ConfidenceInterval
    """
    value = count1 / n1 / meas1 - count2 / n2 / meas2
    alpha = 1 - conf

    (lower, upper) = statsmodels.stats.rates.confint_poisson_2indep(
        count1=count1,
        exposure1=n1*meas1,
        count2=count2,
        exposure2=n2*meas2,
        method=method,
        compare=compare,
        alpha=alpha)

    return ConfidenceInterval(
        name='difference between rates of events',
        value=value,
        lower=lower,
        upper=upper,
        conf=conf)

def test_1sample(count, n, meas=1.0, H0_rate=1.0, alternative='two-sided', method='exact-c'):
    """
    Hypothesis test for the rate of events.

    Null-hypothesis: `count / n / meas == H0_rate`.

    Calls `statsmodels.stats.rates.test_poisson` (statsmodels.org).

    Arguments
    ---------
    count (int) -- Number of events.
    n (int) -- Number of periods over which events were counted.

    Optional
    --------
    meas (float) -- Extent of one period of observation. (Default 1.)
    H0_rate (float) -- Null-hypothesis rate. (Default 1.)
    alternative (str) -- Sense of alternative hypothesis. One of "two-sided",
        "less" or "greater". (Default "two-sided".)
    method (str) -- Test method (default "exact-c"). See statsmodels docs for more.

    Returns
    -------
    mqr.hyptest.HypothesisTest
    """
    alt = interop.alternative(alternative, lib='statsmodels')
    res = statsmodels.stats.rates.test_poisson(
        count=count,
        nobs=n*meas,
        value=H0_rate,
        method=method,
        alternative=alt,)

    return HypothesisTest(
        description='rate of events',
        alternative=alternative,
        method=method,
        sample_stat=f'rate',
        sample_stat_target=H0_rate,
        sample_stat_value=count/n/meas,
        stat=res.statistic,
        pvalue=res.pvalue,)

def test_2sample(count1, n1, count2, n2, meas1=1.0, meas2=1.0,
                    H0_value=None, alternative='two-sided', method='wald', compare='diff'):
    """
    Hypothesis test for equality of rates.

    Null-hypothesis:
    - `count1 / n1 / meas1 - count2 / n2 / meas2 == H0_value`,
        if `compare` is "diff", or
    - `count1 / n1 / meas1 / (count2 / n2 / meas2) == H0_value`,
        if `compare` is "ratio".

    Calls `statsmodels.stats.rates.test_poisson_2indep` (statsmodels.org).

    Arguments
    ---------
    count1 (int) -- Number of events in first observation.
    n1 (int) -- Number of periods over which first events were counted.
    count2 (int) -- Number of events in second observation.
    n2 (int) -- Number of periods over which second events were counted.

    Optional
    --------
    meas1 (float) -- Extent of one period in first observation. (Default 1.)
    meas2 (float) -- Extent of one period in second observation. (Default 1.)
    H0_value (float) -- Null-hypothesis value. (Default 1 when compare is "ratio",
        0 when compare is "diff".)
    alternative (str) -- Sense of alternative hypothesis. One of "two-sided",
        "less" or "greater". (Default "two-sided".)
    method (str) -- Test method (default "wald"). See statsmodels docs for more.
    compare (str) -- Null-hypothesis (default "diff"): either "diff" or "ratio".

    Returns
    -------
    mqr.confint.ConfidenceInterval
    """
    alt = interop.alternative(alternative, lib='statsmodels')
    res = statsmodels.stats.rates.test_poisson_2indep(
        count1=count1,
        exposure1=n1*meas1,
        count2=count2,
        exposure2=n2*meas2,
        value=H0_value,
        method=method,
        compare=compare,
        alternative=alt)

    if compare == 'diff':
        desc = 'difference between'
        sample_stat_sym = '-'
        sample_stat_value = count1 / n1 / meas1 - count2 / n2 / meas2
        if H0_value is None:
            H0_value = 0.0
    elif compare == 'ratio':
        desc = 'ratio of'
        sample_stat_sym = '/'
        sample_stat_value = count1 * n2 * meas2 / (count2 * n1 * meas1)
        if H0_value is None:
            H0_value = 1.0
    else:
        raise ValueError(f'`compare` argument ({compare}) not recognised')

    return HypothesisTest(
        description=f'{desc} rates of events',
        alternative=alternative,
        method=method,
        sample_stat=f'rate1 {sample_stat_sym} rate2',
        sample_stat_target=H0_value,
        sample_stat_value=sample_stat_value,
        stat=res.statistic,
        pvalue=res.pvalue,)
