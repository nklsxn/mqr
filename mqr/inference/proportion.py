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

def power_1sample(pa, H0_prop, nobs, alpha=0.05, alternative='two-sided', method='norm-approx'):
    """
    Calculate power of a test of proportion in a sample.

    Null-hypothesis: `pa - H0_prop == 0`.

    Arguments
    ---------
    pa (float) -- Alternative hypothesis proportion, forming effect size.
    H0_prop (float) -- Null-hypothesis proportion.
    nobs (int) -- Number of observations.
    alpha (float) -- Required significance.

    Optional
    --------
    alternative (float) -- Sense of alternative hypothesis. One of "two-sided",
        "less" or "greater". (Default "two-sided".)
    method (str) -- Test method. Only "norm-approx", the normal approximation to
        the binomial distribution is implemented.

    Returns
    -------
    mqr.power.TestPower
    """
    if method == 'norm-approx':
        diff = H0_prop - pa
        mu = np.sqrt(H0_prop * (1 - H0_prop) / nobs)
        sigma = np.sqrt(pa * (1 - pa) / nobs)
        dist = scipy.stats.norm()

        if alternative == 'less':
            z = dist.ppf(1 - alpha)
            power = dist.cdf((diff - z * mu) / sigma)
        elif alternative == 'greater':
            z = dist.ppf(1 - alpha)
            power = 1 - dist.cdf((diff + z * mu) / sigma)
        elif alternative == 'two-sided':
            z = dist.ppf(1 - alpha/2)
            power = 1 - dist.cdf((diff + z * mu) / sigma) + dist.cdf((diff - z * mu) / sigma)
        else:
            raise ValueError(f'Invalid alternative {alternative}. Use one of "less", "greater" or "two-sided".')
    else:
        raise ValueError(f'Invalid method {method}. Use one of "less", "greater" or "two-sided".')

    return TestPower(
        name='proportion',
        alpha=alpha,
        beta=1-power,
        effect=f'{pa:g} - {H0_prop:g} = {pa-H0_prop:g}',
        alternative=alternative,
        method=method,
        sample_size=nobs)

def power_2sample(p1, p2, nobs, alpha=0.05, alternative='two-sided', method='norm-approx'):
    """
    Calculate power of a test of difference of two proportions in two samples.

    Null-hypothesis: `p1 - p2 == 0`.

    Arguments
    ---------
    p1 (float) -- First proportion.
    p2 (float) -- Second proportion.
    nobs (int) -- Number of observations.
    alpha (float) -- Required significance.

    Optional
    --------
    alternative (float) -- Sense of alternative hypothesis. One of "two-sided",
        "less" or "greater". (Default "two-sided".)
    method (str) -- Test method. Only "norm-approx", the normal approximation to
        the binomial distribution, is implemented.

    Returns
    -------
    mqr.power.TestPower
    """
    if method == 'norm-approx':
        diff = p2 - p1
        pavg = (p1 + p2) / 2
        pavg_scale = np.sqrt(2 * pavg * (1 - pavg) / nobs)
        p_scale = np.sqrt(p1 * (1 - p1) / nobs + p2 * (1 - p2) / nobs)
        dist = scipy.stats.norm()

        if alternative == 'less':
            z = dist.ppf(1 - alpha)
            num = diff - z * pavg_scale
            den = p_scale
            power = dist.cdf(num / den)
        elif alternative == 'greater':
            z = dist.ppf(1 - alpha)
            num = diff + z * pavg_scale
            den = p_scale
            power = 1 - dist.cdf(num / den)
        elif alternative == 'two-sided':
            z = dist.ppf(1 - alpha/2)
            num1 = diff + z * pavg_scale
            num2 = diff - z * pavg_scale
            den = p_scale
            power = 1 - dist.cdf(num1 / den) + dist.cdf(num2 / den)
        else:

            raise ValueError(f'Invalid alternative {alternative}. Use one of "less", "greater" or "two-sided".')
    else:
        raise ValueError(f'Invalid method {method}. Use one of "less", "greater" or "two-sided".')

    return TestPower(
        name='difference between proportions',
        alpha=alpha,
        beta=1-power,
        effect=f'{p1:g} - {p2:g} = {p1-p2:g}',
        alternative=alternative,
        method=method,
        sample_size=nobs)

def size_1sample(pa, H0_prop, alpha, beta, alternative='two-sided', method='norm-approx'):
    """
    Calculate sample size for test of proportion.

    Null-hypothesis: `pa - H0_prop == 0`.

    Arguments
    ---------
    pa (float) -- Alternative proportiond, forming the effect size with `p0`.
    p0 (float) -- Null proportion.
    alpha (float) -- Required significance.
    beta (float) -- Required beta (1 - power).

    Optional
    --------
    alternative (float) -- Sense of alternative hypothesis. One of "two-sided",
        "less" or "greater". (Default "two-sided".)
    method (str) -- One of
        'norm-approx' normal approximation to the binomial distribution,
        'arcsin' inverse-sine approximation to the binomial distribution.

    Returns
    -------
    mqr.power.TestPower
    """
    if method == 'norm-approx':
        def power_fn(nobs):
            return power_1sample(
                pa=pa,
                H0_prop=H0_prop,
                nobs=nobs,
                alpha=alpha,
                alternative=alternative,
                method=method).beta - beta
        nobs_opt = scipy.optimize.fsolve(power_fn, 1)[0]
    elif method == 'arcsin':
        if alternative == 'less' or alternative == 'greater':
            crit = alpha
        elif alternative == 'two-sided':
            crit = alpha / 2
        else:
            raise ValueError(f'Alternative {alternative} not valid. Use one of "less", "greater" or "two-sided".')
        dist = scipy.stats.norm()

        Zb = dist.ppf(1-beta)
        Za = -dist.ppf(crit)
        num = Za + Zb
        denom = np.arcsin(np.sqrt(pa)) - np.arcsin(np.sqrt(H0_prop))
        nobs_opt = num**2 / denom**2 / 4
    else:
        raise ValueError(f'Method {method} not supported.')

    return TestPower(
        name='proportion',
        alpha=alpha,
        beta=beta,
        effect=f'{pa:g} - {H0_prop:g} = {pa-H0_prop:g}',
        alternative=alternative,
        method=method,
        sample_size=nobs_opt)

def size_2sample(p1, p2, alpha, beta, alternative='two-sided', method='norm-approx'):
    """
    Calculate sample size to test equality of two proportions.

    Null-hypothesis: `p1 - p2 == 0`.

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
    method (str) -- Test method. One of
        'norm-approx': the normal approximation to the binomial distribution,
        'arcsin': the arcsin method.

    Returns
    -------
    mqr.power.TestPower
    """
    if method == 'norm-approx':
        def power_fn(nobs):
            return power_2sample(
                p1=p1,
                p2=p2,
                nobs=nobs,
                alpha=alpha,
                alternative=alternative,
                method=method).beta - beta
        nobs_opt = scipy.optimize.fsolve(power_fn, 1)[0]
    elif method == 'arcsin':
        if alternative == 'less' or alternative == 'greater':
            crit = alpha
        elif alternative == 'two-sided':
            crit = alpha / 2
        else:
            raise ValueError(f'Alternative {alternative} not valid. Use one of "less", "greater" or "two-sided".')
        dist = scipy.stats.norm()

        Zb = dist.ppf(1-beta)
        Za = -dist.ppf(crit)
        num = Za + Zb
        denom = np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2))
        nobs_opt = 2 * num**2 / denom**2 / 4
    else:
        raise ValueError(f'Method {method} not supported.')

    return TestPower(
        name='proportion',
        alpha=alpha,
        beta=beta,
        effect=f'{p1:g} - {p2:g} = {p1-p2:g}',
        alternative=alternative,
        method=method,
        sample_size=nobs_opt)

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
        method=method,
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
        method=method,
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

def test_2sample(count1, nobs1, count2, nobs2, H0_diff=0.0, alternative='two-sided', method='agresti-caffo'):
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
