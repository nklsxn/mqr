"""
Confidence intervals and hypothesis tests (parametric) for standard deviation.
"""

from mqr.inference.confint import ConfidenceInterval
from mqr.inference.hyptest import HypothesisTest
from mqr.inference.power import TestPower

import mqr.inference.variance as variance

import numpy as np

def size_1sample(std_ratio, alpha, beta, alternative='two-sided'):
    """
    Calculate sample size for test of standard deviation of a sample.

    Null-hypothesis: `std_ratio = sigma / sigma0 == 1` (two-sided).

    Arguments
    ---------
    std_ratio (float) -- Required effect size.
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
    var = variance.size_1sample(np.square(std_ratio), alpha, beta, alternative)
    return TestPower(
        name='standard deviation',
        alpha=var.alpha,
        beta=var.beta,
        effect=std_ratio,
        alternative=alternative,
        method=var.method,
        sample_size=var.sample_size)

def size_2sample(std_ratio, alpha, beta, alternative='two-sided'):
    """
    Calculate sample size for test of ratio of standard deviations.

    Null-hypothesis: `std_ratio == sigma_1 / sigma_2 == 1` (two-sided).

    Arguments
    ---------
    std_ratio (float) -- Required effect size.
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
    var = variance.size_2sample(np.square(std_ratio), alpha, beta, alternative)
    return TestPower(
        name='ratio of standard deviations',
        alpha=var.alpha,
        beta=var.beta,
        effect=std_ratio,
        alternative=alternative,
        method=var.method,
        sample_size=var.sample_size)

def confint_1sample(x, conf=0.95):
    """
    Confidence interval for the standard deviation of a sample.

    Arguments
    ---------
    x (array[float]) -- Calcaulate interval for the standard deviation of this sample.

    Optional
    --------
    conf (float) -- Confidence level that determines the width of the interval.
        (Default 0.95.)

    Returns
    -------
    mqr.confint.ConfidenceInterval
    """
    var = variance.confint_1sample(x, conf)
    return ConfidenceInterval(
        name="standard deviation",
        value=np.sqrt(var.value),
        lower=np.sqrt(var.lower),
        upper=np.sqrt(var.upper),
        conf=conf)

def confint_2sample(x, y, conf=0.95):
    """
    Confidence interval for the ratio of standard deviations of two samples.

    Arguments
    ---------
    x, y (array[float]) -- Calcaulate interval for the ratio of standard
        deviations of these two samples.

    Optional
    --------
    conf (float) -- Confidence level that determines the width of the interval.
        (Default 0.95.)

    Returns
    -------
    mqr.confint.ConfidenceInterval
    """
    var = variance.confint_2sample(x, y, conf)
    return ConfidenceInterval(
        name="ratio of standard deviations",
        value=np.sqrt(var.value),
        lower=np.sqrt(var.lower),
        upper=np.sqrt(var.upper),
        conf=conf)

def test_1sample(x, H0_std, alternative='two-sided'):
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
    var = variance.test_1sample(x, np.square(H0_std), alternative)
    x_name = x.name if hasattr(x, 'name') else 'x'
    return HypothesisTest(
        description='standard deviation',
        alternative=alternative,
        method='chi2',
        sample_stat=f'std({x_name})',
        sample_stat_target=H0_std,
        sample_stat_value=np.sqrt(var.sample_stat_value),
        stat=var.stat,
        pvalue=var.pvalue)

def test_2sample(x, y, alternative='two-sided'):
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
    var = variance.test_2sample(x, y, alternative, 'f')
    x_name = x.name if hasattr(x, 'name') else 'x'
    y_name = y.name if hasattr(y, 'name') else 'y'
    return HypothesisTest(
        description='ratio of standard deviations',
        alternative=alternative,
        method=var.method,
        sample_stat=f'std({x_name}) / std({y_name})',
        sample_stat_target=np.sqrt(var.sample_stat_target),
        sample_stat_value=np.sqrt(var.sample_stat_value),
        stat=var.stat,
        pvalue=var.pvalue)
