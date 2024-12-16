from mqr.inference.confint import ConfidenceInterval
from mqr.inference.hyptest import HypothesisTest
from mqr.inference.power import TestPower

from mqr.inference.lib import util
import mqr.inference.variance as variance

import numpy as np

def size_1sample(std_ratio, alpha, beta, alternative='two-sided'):
    """
    Calculate sample size for test of standard deviation of a sample.

    Null-hypothesis
        `std_ratio` = sigma / sigma0 == 1

    Parameters
    ----------
    std_ratio : float
        Required effect size.
    alpha : float
        Required significance.
    beta : float
        Required beta (1 - power).
    alternative : {'two-sided', 'less', 'greater'}, optional
        Sense of alternative hypothesis.

    Returns
    -------
    :class:`mqr.inference.power.TestPower`
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

    Null-hypothesis
        `std_ratio` = sigma_1 / sigma_2 == 1

    Parameters
    ----------
    std_ratio : float
        Required effect size.
    alpha : float
        Required significance.
    beta : float
        Required beta (1 - power).
    alternative : {'two-sided', 'less', 'greater'}, optional
        Sense of alternative hypothesis.

    Returns
    -------
    :class:`mqr.inference.power.TestPower`
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

def confint_1sample(x, conf=0.95, bounded='both', method='chi2'):
    """
    Confidence interval for the standard deviation of a sample.

    Parameters
    ----------
    x : array_like
        Calcaulate interval for the standard deviation of this sample.
    conf : float, optional
        Confidence level that determines the width of the interval.
    bounded : {'both', 'below', 'above'}, optional
        Which sides of the interval to close.
    method : {'chi2'}, optional
        Only an interval based on the 'chi2' distribution is implemented.

    Returns
    -------
    :class:`mqr.inference.confint.ConfidenceInterval`
    """
    var = variance.confint_1sample(x, conf, bounded=bounded, method=method)
    return ConfidenceInterval(
        name="standard deviation",
        method=method,
        value=np.sqrt(var.value),
        lower=np.sqrt(var.lower),
        upper=np.sqrt(var.upper),
        conf=conf,
        bounded=bounded)

def confint_2sample(x, y, conf=0.95, bounded='both', method='f'):
    """
    Confidence interval for the ratio of standard deviations of two samples.

    Parameters
    ----------
    x, y : array_like
        Calculate interval for the ratio of standard deviations of these two samples.
    conf : float, optional
        Confidence level that determines the width of the interval.
    bounded : {'both', 'below', 'above'}, optional
        Which sides of the interval to close.
    method : {'f'}, optional
        Only 'f' is available, which calculates the standard deviation from an
        interval on variance.

    Returns
    -------
    :class:`mqr.inference.confint.ConfidenceInterval`
    """
    var = variance.confint_2sample(x, y, conf, bounded=bounded, method=method)
    lower = 0.0 if bounded == 'above' else np.sqrt(var.lower)
    upper = np.inf if bounded == 'below' else np.sqrt(var.upper)
    return ConfidenceInterval(
        name="ratio of standard deviations",
        method=method,
        value=np.sqrt(var.value),
        lower=lower,
        upper=upper,
        conf=conf,
        bounded=bounded)

def test_1sample(x, H0_std, alternative='two-sided'):
    """
    Hypothesis test for the varianve of a sample.

    Null hypothesis
        var(`x`) / `H0_var` == 1

    Parameters
    ----------
    x : array_like
        Test variance of this sample.
    H0_std : float
        Null-hypothesis standard deviation.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Sense of alternative hypothesis.

    Returns
    -------
    :class:`mqr.inference.hyptest.HypothesisTest`
    """
    var = variance.test_1sample(x, np.square(H0_std), alternative)
    x_name = util.var_name(x, 'x')
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

    Null hypothesis
        var(`x`) / var(`y`) == 1

    Parameters
    ----------
    x, y : array_like
        Test ratio of variances of these two samples.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Sense of alternative hypothesis.
    method : {'f', 'bartlett'}, optional
        | 'f'
        |   F-test on variances applied to the squares of sample standard deviations.
        | 'bartlett'
        |   Bartlett's test. Calls :func:`scipy..bartlett <scipy.stats.bartlett>`.

    Returns
    -------
    :class:`mqr.inference.hyptest.HypothesisTest`
    """
    var = variance.test_2sample(x, y, alternative, 'f')
    x_name = util.var_name(x, 'x')
    y_name = util.var_name(y, 'y')
    return HypothesisTest(
        description='ratio of standard deviations',
        alternative=alternative,
        method=var.method,
        sample_stat=f'std({x_name}) / std({y_name})',
        sample_stat_target=np.sqrt(var.sample_stat_target),
        sample_stat_value=np.sqrt(var.sample_stat_value),
        stat=var.stat,
        pvalue=var.pvalue)
