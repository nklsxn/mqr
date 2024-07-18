"""
Confidence intervals and hypothesis tests (parametric) for the mean.
"""

from mqr.inference.confint import ConfidenceInterval
from mqr.inference.hyptest import HypothesisTest
from mqr.inference.power import TestPower

import mqr.interop.inference as interop

import numpy as np
import scipy
import statsmodels

def size_1sample(effect, alpha, beta, alternative='two-sided'):
    """
    Calculate sample size for test of mean of sample.

    Calls stats.models.stats.power.tt_solve_power` (statsmodels.org).

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
    alt = interop.alternative(alternative, 'statsmodels')
    power = 1.0 - beta
    nobs = statsmodels.stats.power.tt_solve_power(
        alpha=alpha,
        power=power,
        effect_size=effect,
        alternative=alt)
    return TestPower(
        name='mean',
        alpha=alpha,
        beta=beta,
        effect=effect,
        alternative=alternative,
        method='t',
        sample_size=nobs)

def size_2sample(effect, alpha, beta, alternative='two-sided'):
    """
    Calculate sample size for test of difference of unpaired means.

    Calls `statsmodels.stats.power.tt_ind_solve_power` (statsmodels.org).

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
    alt = interop.alternative(alternative, 'statsmodels')
    power = 1.0 - beta
    nobs = statsmodels.stats.power.tt_ind_solve_power(
        alpha=alpha,
        power=power,
        effect_size=effect,
        ratio=1.0,
        alternative=alt)
    return TestPower(
        name='difference between means (independent)',
        alpha=alpha,
        beta=beta,
        effect=effect,
        alternative=alternative,
        method='t',
        sample_size=nobs)

def size_paired(effect, alpha, beta, alternative='two-sided'):
    """
    Calculate sample size for test of difference of unpaired means.

    Calls `statsmodels.stats.power.TTestPower` (statsmodels.org).

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
    alt = interop.alternative(alternative, 'statsmodels')
    power = 1.0 - beta
    nobs = statsmodels.stats.power.TTestPower().solve_power(
        alpha=alpha,
        power=power,
        effect_size=effect,
        alternative=alt)
    return TestPower(
        name='difference between means (paired)',
        alpha=alpha,
        beta=beta,
        effect=effect,
        alternative=alternative,
        method='t',
        sample_size=nobs)

def confint_1sample(x, conf=0.95, bounded='both'):
    """
    Confidence interval for mean.

    Uses `statsmodels.stats.api.DescrStatsW(...).tconfint_mean` (statsmodels.org).

    Arguments
    ---------
    x (array[float]) -- Calcaulate interval for mean of this sample.

    Optional
    --------
    conf (float) -- Confidence level that determines the width of the interval.
        (Default 0.95.)
    bounded (str) -- Which sides of the interval to close. (Default True.)

    Returns
    -------
    mqr.confint.ConfidenceInterval
    """
    value = np.mean(x)
    alt = interop.bounded(bounded, 'statsmodels')
    alpha = 1 - conf
    lower, upper = statsmodels.stats.api.DescrStatsW(x).tconfint_mean(alpha, alt)
    return ConfidenceInterval(
        name='mean',
        value=value,
        lower=lower,
        upper=upper,
        conf=conf)

def confint_2sample(x, y, conf=0.95, pooled=True, bounded='both'):
    """
    Confidence interval for difference of two unpaired means.

    Uses `DescrStatsW` and `CompareMeans` from `statsmodels.stats.api` (statsmodels.org).

    Arguments
    ---------
    x, y (array[float]) -- Calcaulate interval for difference between means of these samples.

    Optional
    --------
    conf (float) -- Confidence level that determines the width of the interval.
        (Default 0.95.)
    pooled (bool) -- When `True`, the samples have the same variance,
        `False` otherwise. (Default True.)
    bounded (str) -- Which sides of the interval to close. (Default "both".)

    Returns
    -------
    mqr.confint.ConfidenceInterval
    """
    value = np.mean(x) - np.mean(y)
    alt = interop.bounded(bounded, 'statsmodels')
    alpha = 1 - conf
    usevar = 'pooled' if pooled else 'unequal'
    xs = statsmodels.stats.api.DescrStatsW(x)
    ys = statsmodels.stats.api.DescrStatsW(y)
    comp = statsmodels.stats.api.CompareMeans(xs, ys)
    lower, upper = comp.tconfint_diff(
        alpha=alpha,
        usevar=usevar,
        alternative=alt)
    return ConfidenceInterval(
        name='difference between means (independent)',
        value=value,
        lower=lower,
        upper=upper,
        conf=conf)

def confint_paired(x, y, conf=0.95, bounded='both'):
    """
    Confidence interval for difference of two paired means.

    Equivalent to `mqr.inference.mean.confint_1sample(x-y)`.

    Arguments
    ---------
    x, y (array[float]) -- Calcaulate interval for difference between means of these samples.

    Optional
    --------
    conf (float) -- Confidence level that determines the width of the interval.
        (Default 0.95.)
    bounded (str) -- Which sides of the interval to close. (Default "both".)

    Returns
    -------
    mqr.confint.ConfidenceInterval
    """
    delta = x - y
    ci = confint_1sample(delta, conf, bounded)
    ci.name = 'difference between means (paired)'
    return ci

def test_1sample(x, H0_mean=0.0, alternative='two-sided', method='t'):
    """
    Hypothesis test for the mean of a sample.

    Null-hypothesis: `mean(x) == H0_mean`.

    Arguments
    ---------
    x (array[float]) -- Test mean of this sample.

    Optional
    --------
    H0_mean (float) -- Null-hypothesis mean. (Default 0.)
    alternative (str) -- Sense of alternative hypothesis. One of "two-sided",
        "less" or "greater". (Default "two-sided".)
    method (str) -- Type of test (default "t":
        "t" for student's t (`scipy.stats.ttest_1samp`, scipy.org),
        "z" for z-score (`statsmodels.stats.api.ztest`, statsmodels.org).

    Returns
    -------
    mqr.hyptest.HypothesisTest
    """
    if method == 't':
        statistic, pvalue = scipy.stats.ttest_1samp(
            a=x,
            popmean=H0_mean,
            alternative=alternative)
    elif method == 'z':
        alt = interop.alternative(alternative, lib='statsmodels')
        statistic, pvalue = statsmodels.stats.api.ztest(
            x1=x,
            value=H0_mean,
            alternative=alt)
    else:
        raise ValueError(f'method "{method}" is not available')

    x_name = x.name if hasattr(x, 'name') else 'x'
    return HypothesisTest(
        description='mean',
        alternative=alternative,
        method=method,
        sample_stat=f'mean({x_name})',
        sample_stat_target=H0_mean,
        sample_stat_value=np.mean(x),
        stat=statistic,
        pvalue=pvalue,
    )

def test_2sample(x, y, H0_diff=0.0, pooled=True, alternative='two-sided', method='t'):
    """
    Hypothesis test for the difference between means of two unpaired samples.

    Null-hypothesis: `mean(x) - mean(y) == H0_diff`.

    Arguments
    ---------
    x, y (array[float]) -- Test the difference between means of these samples.

    Optional
    --------
    H0_diff (float) -- Null-hypothesis difference. (Default 0.)
    pooled (bool) -- When `True`, the samples have the same variance,
        `False` otherwise. (Default True.)
    alternative (str) -- Sense of alternative hypothesis. One of "two-sided",
        "less" or "greater". (Default "two-sided".)
    method (str) -- Type of test (default "t"):
        "t" for student's t (`statsmodels.stats.api.ttest_ind`, statsmodels.org),
        "z" for z-score (`statsmodels.stats.api.ztest`, statsmodels.org).

    Returns
    -------
    mqr.hyptest.HypothesisTest
    """
    alt = interop.alternative(alternative, 'statsmodels')
    usevar = 'pooled' if pooled else 'unequal'
    if method == 't':
        statistic, pvalue, _dof = statsmodels.stats.api.ttest_ind(
            x1=x,
            x2=y,
            alternative=alt,
            usevar=usevar,
            value=H0_diff)
    elif method == 'z':
        statistic, pvalue = statsmodels.stats.api.ztest(
            x1=x,
            x2=y,
            value=H0_diff,
            alternative=alt,
            usevar=usevar)
    else:
        raise ValueError(f'method "{method}" is not available')
    
    x_name = x.name if hasattr(x, 'name') else 'x'
    y_name = y.name if hasattr(y, 'name') else 'y'
    return HypothesisTest(
        description='difference between means (independent)',
        alternative=alternative,
        method=method,
        sample_stat=f'mean({x_name}) - mean({y_name})',
        sample_stat_target=H0_diff,
        sample_stat_value=x.mean()-y.mean(),
        stat=statistic,
        pvalue=pvalue,)

def test_paired(x, y, alternative='two-sided', method='t'):
    """
    Hypothesis test for the difference between means of two paired samples.

    Null-hypothesis: `mean(x) == mean(y)`.

    Arguments
    ---------
    x, y (array[float]) -- Test the difference between means of these samples.

    Optional
    --------
    alternative (str) -- Sense of alternative hypothesis. One of "two-sided",
        "less" or "greater". (Default "two-sided".)
    method (str) -- Type of test, only the default:
        "t" for student's t (`scipy.stats.ttest_rel`, scipy.org).

    Returns
    -------
    mqr.hyptest.HypothesisTest
    """
    if method != 't':
        raise ValueError(f'method "{method}" is not available')

    result = scipy.stats.ttest_rel(x, y, alternative=alternative)

    x_name = x.name if hasattr(x, 'name') else 'x'
    y_name = y.name if hasattr(y, 'name') else 'y'
    return HypothesisTest(
        description='difference between means (paired)',
        alternative=alternative,
        method='t',
        sample_stat=f'mean({x_name}) - mean({y_name})',
        sample_stat_target=0,
        sample_stat_value=x.mean()-y.mean(),
        stat=result.statistic,
        pvalue=result.pvalue,)
