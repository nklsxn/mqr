"""
Confidence intervals and hypothesis tests (parametric) for variance.
"""

from mqr.inference.confint import ConfidenceInterval
from mqr.inference.hyptest import HypothesisTest
from mqr.inference.power import TestPower

import mqr.interop.inference as interop

import numpy as np
import scipy
import statsmodels

def power_1sample(effect, nobs, alpha=0.05, alternative='two-sided'):
    """
    Calculate power for test of variance of a sample.

    Null-hypothesis: `effect = var / var_0 == 1` (two-sided).

    Arguments
    ---------
    effect (float) -- Effect size.
    nobs (float) -- Number of observations in sample.

    Optional
    --------
    alpha (float) -- Significance level. (Default 0.05.)
    alternative (str) -- Sense of alternative hypothesis. One of "two-sided",
        "less" or "greater". (Default "two-sided".)

    Returns
    -------
    mqr.power.TestPower
    """
    den = effect
    dist = scipy.stats.chi2(nobs-1)

    if alternative == 'less':
        num = dist.ppf(alpha)
        power = dist.cdf(num/den)
    elif alternative == 'greater':
        num = dist.ppf(1-alpha)
        power = 1 - dist.cdf(num/den)
    elif alternative == 'two-sided':
        num_1 = dist.ppf(1-alpha/2)
        num_2 = dist.ppf(alpha/2)
        power = 1 - (dist.cdf(num_1/den) - dist.cdf(num_2/den))
    else:
        raise ValueError(f'Invalid alternative "{alternative}". Use "two-sided" (default), "less", or "greater".')

    return TestPower(
        name='variance',
        alpha=alpha,
        beta=1-power,
        effect=effect,
        alternative=alternative,
        method='chi2',
        sample_size=nobs)

def power_2sample(var_ratio, nobs, alpha=0.05, alternative='two-sided'):
    """
    Calculate power for test of ratio of variances.

    Null-hypothesis: `effect = var_1 / var_2 == 1` (two-sided).

    Arguments
    ---------
    var_ratio (float) -- Effect size.
    nobs (float) -- Number of observations in sample.

    Optional
    --------
    alpha (float) -- Significance level. (Default 0.05.)
    alternative (str) -- Sense of alternative hypothesis. One of "two-sided",
        "less" or "greater". (Default "two-sided".)

    Returns
    -------
    mqr.power.TestPower
    """
    dist = scipy.stats.f(nobs-1, nobs-1)

    if alternative == 'less':
        num = dist.ppf(alpha)
        power = dist.cdf(num/var_ratio)
    elif alternative == 'greater':
        num = dist.ppf(1-alpha)
        power = 1 - dist.cdf(num/var_ratio)
    elif alternative == 'two-sided':
        num_1 = dist.ppf(1-alpha/2)
        num_2 = dist.ppf(alpha/2)
        power = 1 - (dist.cdf(num_1/var_ratio) - dist.cdf(num_2/var_ratio))
    else:
        raise ValueError(f'Invalid alternative "{alternative}". Use "two-sided" (default), "less", or "greater".')

    return TestPower(
        name='ratio of variances',
        alpha=alpha,
        beta=1-power,
        effect=var_ratio,
        alternative=alternative,
        method='f',
        sample_size=nobs)

def size_1sample(effect, alpha, beta, alternative='two-sided'):
    """
    Calculate sample size for test of variance of a sample.

    Null-hypothesis: `effect == var / var_0 == 1` (two-sided).

    Arguments
    ---------
    effect (float) -- Required effect size.
    alpha (float) -- Required significance.
    beta (float) -- Required beta (1 - power).

    Optional
    --------
    alternative (str) -- Sense of alternative hypothesis. One of "two-sided",
        "less" or "greater". (Default "two-sided".)

    Returns
    -------
    mqr.power.TestPower
    """
    if (effect < 1) and (alternative == 'greater'):
        raise ValueError('alternative "greater" not valid when effect < 1')
    elif (effect > 1) and (alternative == 'less'):
        raise ValueError('alternative "less" not valid when effect > 1')

    def beta_fn(n):
        power = power_1sample(
            effect=effect,
            nobs=n,
            alpha=alpha,
            alternative=alternative)
        return power.beta - beta
    nobs = scipy.optimize.fsolve(beta_fn, 2)[0]

    return TestPower(
        name='variance',
        alpha=alpha,
        beta=beta,
        effect=effect,
        alternative=alternative,
        method='chi2',
        sample_size=nobs)

def size_2sample(var_ratio, alpha, beta, alternative='two-sided'):
    """
    Calculate sample size for test of ratio of variances.

    Null-hypothesis: `var_ratio == var_1 / var_2 == 1` (two-sided).

    Arguments
    ---------
    var_ratio (float) -- Required effect size.
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
    if (alternative == 'less') and (var_ratio > 1.0):
            raise ValueError('diff must be > 1 for a greater-than alternative hypothesis')
    elif (alternative == 'greater') and (var_ratio < 1.0):
            raise ValueError('diff must be < 1 for a less-than alternative hypothesis')

    def beta_fn(n):
        power = power_2sample(
            var_ratio=var_ratio,
            nobs=n,
            alpha=alpha,
            alternative=alternative)
        return power.beta - beta
    nobs = scipy.optimize.fsolve(beta_fn, 2)[0]

    return TestPower(
        name='ratio of variances',
        alpha=alpha,
        beta=beta,
        effect=var_ratio,
        alternative=alternative,
        method='f',
        sample_size=nobs)

def confint_1sample(x, conf=0.95):
    """
    Confidence interval for the variance of a sample.

    Arguments
    ---------
    x (array[float]) -- Calcaulate interval for the variance of this sample.

    Optional
    --------
    conf (float) -- Confidence level that determines the width of the interval.
        (Default 0.95.)

    Returns
    -------
    mqr.confint.ConfidenceInterval
    """
    alpha = 1 - conf
    nobs = len(x)
    s2 = np.var(x, ddof=1)
    dof = nobs - 1
    lower = (nobs - 1) * s2 / scipy.stats.chi2.ppf(1 - alpha / 2, dof)
    upper = (nobs - 1) * s2 / scipy.stats.chi2.ppf(alpha / 2, dof)
    return ConfidenceInterval(
        name='variance',
        method='chi2',
        value=s2,
        lower=lower,
        upper=upper,
        conf=conf)

def confint_2sample(x, y, conf=0.95):
    """
    Confidence interval for the ratio of variances of two samples.

    Arguments
    ---------
    x, y (array[float]) -- Calcaulate interval for the ratio of variances of
        these two samples.

    Optional
    --------
    conf (float) -- Confidence level that determines the width of the interval.
        (Default 0.95.)

    Returns
    -------
    mqr.confint.ConfidenceInterval
    """
    alpha = 1 - conf
    nobsx = len(x)
    nobsy = len(y)
    dofx = nobsx - 1
    dofy = nobsy - 1
    s2x = np.var(x, ddof=1)
    s2y = np.var(y, ddof=1)
    upper = s2x / s2y * scipy.stats.f.ppf(1 - alpha / 2, dofy, dofx)
    lower = s2x / s2y * scipy.stats.f.ppf(alpha / 2, dofy, dofx)
    return ConfidenceInterval(
        name='ratio of variances',
        method='f',
        value=s2x/s2y,
        lower=lower,
        upper=upper,
        conf=conf)

def test_1sample(x, H0_var, alternative='two-sided'):
    """
    Hypothesis test for the varianve of a sample.

    Null hypothesis: `var(x) / H0_var == 1`.

    Arguments
    ---------
    x (array[float]) -- Test variance of this sample.
    H0_var (float) -- Null-hypothesis variance.

    Optional
    --------
    alternative (str) -- Sense of alternative hypothesis. One of "two-sided",
        "less" or "greater". (Default "two-sided".)

    Returns
    -------
    mqr.hyptest.HypothesisTest
    """
    dfx = len(x) - 1
    s2x = np.var(x, ddof=1)
    q = dfx * s2x / H0_var # Eqn 8-17, MR
    dist = scipy.stats.chi2(dfx)

    stat = s2x
    if alternative == 'less':
        pvalue = dist.cdf(q)
    elif alternative == 'greater':
        pvalue = 1 - dist.cdf(q)
    elif alternative == 'two-sided':
        pvalue = 2.0 * np.minimum(dist.cdf(q), 1-dist.cdf(q))
    else:
        assert False, f'Invalid alternative "{alternative}". Use "two-sided" (default), "less", or "greater".'

    x_name = x.name if hasattr(x, 'name') else 'x'
    return HypothesisTest(
        description='variance',
        alternative=alternative,
        method='chi2',
        sample_stat=f'var({x_name})',
        sample_stat_target=H0_var,
        sample_stat_value=s2x,
        stat=stat,
        pvalue=pvalue,)

def test_2sample(x, y, alternative='two-sided', method='f'):
    """
    Hypothesis test for the ratio of variances of two samples.

    Null hypothesis: `var(x) / var(y) == 1`.

    Arguments
    ---------
    x, y (array[float]) -- Test ratio of variances of these two samples.

    Optional
    --------
    alternative (str) -- Sense of alternative hypothesis. One of "two-sided",
        "less" or "greater". (Default "two-sided".)
    method (str) -- Type of test (default "f"):
        "f" for F-test (calculated directly from f-dist),
        "bartlett" for Bartlett's test (`scipy.stats.bartlett`, scipy.org).

    Returns
    -------
    mqr.hyptest.HypothesisTest
    """
    if method == 'f':
        description = 'ratio of variances'
        tgt = 1.0
        rel = '/'

        dfx = len(x) - 1
        dfy = len(y) - 1
        f = np.var(x, ddof=1) / np.var(y, ddof=1)
        dist = scipy.stats.f(dfx, dfy)

        stat = f
        if alternative == 'less':
            pvalue = dist.cdf(f)
        elif alternative == 'greater':
            pvalue = 1 - dist.cdf(f)
        elif alternative == 'two-sided':
            pvalue = 2.0 * np.minimum(1.0 - dist.cdf(f), dist.cdf(f))
        else:
            assert False, f'Invalid alternative "{alternative}". Use "two-sided" (default), "less", or "greater".'
    elif method == 'levene':
        if alternative != 'two-sided':
            raise ValueError('one-sided alternative not available in Levene test')
        description = 'equality of variances'
        tgt = 0.0
        rel = '-'
        (stat, pvalue) = scipy.stats.levene(x, y)
    elif method == 'bartlett':
        if alternative != 'two-sided':
            raise ValueError('one-sided alternative not available in Bartlett test')
        description = 'equality of variances'
        tgt = 0.0
        rel = '-'
        (stat, pvalue) = scipy.stats.bartlett(x, y)
    else:
        raise ValueError(f'invalid method "{method}"')

    x_name = x.name if hasattr(x, 'name') else 'x'
    y_name = y.name if hasattr(y, 'name') else 'y'
    return HypothesisTest(
        description=description,
        alternative=alternative,
        method=method,
        sample_stat=f'var({x_name}) {rel} var({y_name})',
        sample_stat_target=tgt,
        sample_stat_value=stat,
        stat=stat,
        pvalue=pvalue,)
