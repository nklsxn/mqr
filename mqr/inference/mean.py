from mqr.inference.confint import ConfidenceInterval
from mqr.inference.hyptest import HypothesisTest
from mqr.inference.power import TestPower

import mqr.inference.lib.util as util
import mqr.interop.inference as interop

import numbers
import numpy as np
import scipy
import statsmodels
import warnings

def size_1sample(effect, alpha, beta, alternative='two-sided'):
    """
    Calculate sample size for a t-test for the mean.

    Calls :func:`sm..tt_solve_power <statsmodels.stats.power.tt_solve_power>`.

    Parameters
    ----------
    effect : float
        Required effect size; Cohen's F.
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
    alt = interop.alternative(alternative, 'statsmodels')
    power = 1.0 - beta
    warnings.filterwarnings("error")
    try:
        nobs = statsmodels.stats.power.tt_solve_power(
            alpha=alpha,
            power=power,
            effect_size=effect,
            alternative=alt)
    finally:
        warnings.resetwarnings()
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

    Calls :func:`sm..tt_ind_solve_power <statsmodels.stats.power.tt_ind_solve_power>`.

    Parameters
    ----------
    effect : float
        Required effect size; Cohen's F.
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
    alt = interop.alternative(alternative, 'statsmodels')
    power = 1.0 - beta
    warnings.filterwarnings("error")
    try:
        nobs = statsmodels.stats.power.tt_ind_solve_power(
            alpha=alpha,
            power=power,
            effect_size=effect,
            ratio=1.0,
            alternative=alt)
    finally:
        warnings.resetwarnings()
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

    Calls :meth:`sm..solve_power <statsmodels.stats.power.TTestPower.solve_power>`.

    Parameters
    ----------
    effect : float
        Required effect size; Cohen's F.
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

def confint_1sample(x, conf=0.95, bounded='both', method='t'):
    """
    Confidence interval for mean.

    Parameters
    ----------
    x : array_like
        Interval for the mean of the population from which this data was sampled.
    conf : float, optional
        Confidence level that determines the width of the interval.
    bounded : {'both', 'below', 'above'}, optional
        Which sides of the interval to close.
    method : {'t', 'z'}, optional
        | 't'
        |   Interval from Student's t distribution.
            Calls :meth:`sm..tconfint_mean <statsmodels.stats.weightstats.DescrStatsW.tconfint_mean>`.
        | 'z'
        |   Interval from the z-scores.
            Calls :meth:`sm..zconfint_mean <statsmodels.stats.weightstats.DescrStatsW.zconfint_mean>`.

    Returns
    -------
    :class:`mqr.inference.confint.ConfidenceInterval`
    """
    value = np.mean(x)
    alt = interop.bounded(bounded, 'statsmodels')
    alpha = 1 - conf
    if method == 't':
        lower, upper = statsmodels.stats.api.DescrStatsW(x).tconfint_mean(alpha, alt)
    elif method == 'z':
        lower, upper = statsmodels.stats.api.DescrStatsW(x).zconfint_mean(alpha, alt)
    else:
        raise ValueError(util.method_error_msg(method, ['t', 'z']))

    return ConfidenceInterval(
        name='mean',
        method=method,
        value=value,
        lower=lower,
        upper=upper,
        conf=conf,
        bounded=bounded)

def confint_2sample(x, y, conf=0.95, pooled=True, bounded='both', method='t'):
    """
    Confidence interval for difference of two unpaired means.

    Parameters
    ----------
    x, y : array_like
        Calculate interval for difference between means of these samples.
    conf : float
        Confidence level that determines the width of the interval.
    pooled : bool
        When `True`, the samples have the same variance, `False` otherwise.
    bounded : {'both', 'below', 'above'}, optional
        Which sides of the interval to close.
    method : {'t', 'z'}, optional
        | 't'
        |   Interval based on Student's t distribution.
            Calls :meth:`sm..tconfint_diff <statsmodels.stats.weightstats.CompareMeans.tconfint_diff>`.
        | 'z'
        |   Interval based on z-scores.
            Calls :meth:`sm..zconfint_diff <statsmodels.stats.weightstats.CompareMeans.zconfint_diff>`.

    Returns
    -------
    :class:`mqr.inference.confint.ConfidenceInterval`
    """
    value = np.mean(x) - np.mean(y)
    alt = interop.bounded(bounded, 'statsmodels')
    alpha = 1 - conf
    usevar = 'pooled' if pooled else 'unequal'
    xs = statsmodels.stats.api.DescrStatsW(x)
    ys = statsmodels.stats.api.DescrStatsW(y)
    comp = statsmodels.stats.api.CompareMeans(xs, ys)

    if method == 't':
        lower, upper = comp.tconfint_diff(
            alpha=alpha,
            usevar=usevar,
            alternative=alt)
    elif method == 'z':
        lower, upper = comp.zconfint_diff(
            alpha=alpha,
            usevar=usevar,
            alternative=alt)
    else:
        raise ValueError(util.method_error_msg(method, ['t', 'z']))
    return ConfidenceInterval(
        name='difference between means (independent)',
        method=method,
        value=value,
        lower=lower,
        upper=upper,
        conf=conf,
        bounded=bounded)

def confint_paired(x, y, conf=0.95, bounded='both', method='t'):
    """
    Confidence interval for difference of two paired means.

    Parameters
    ----------
    x, y : array_like
        Calculate interval for difference between means of these samples.
    conf : float, optional
        Confidence level that determines the width of the interval.
    bounded : {'two-sided', 'less', 'greater'}, optional
        Which sides of the interval to close.
    method : {'t', 'z'}, optional
        | 't'
        |   Interval based on Student's t distribution.
            Calls :meth:`sm..tconfint_mean <statsmodels.stats.weightstats.DescrStatsW.tconfint_mean>`.
        | 'z'
        |   Interval based on z-score.
            Calls :meth:`sm..zconfint_mean <statsmodels.stats.weightstats.DescrStatsW.zconfint_mean>`.

    Returns
    -------
    :class:`mqr.inference.confint.ConfidenceInterval`
    """
    delta = x - y
    ci = confint_1sample(delta, conf, bounded, method)
    ci.name = 'difference between means (paired)'
    return ci

def test_1sample(x, H0_mean=0.0, alternative='two-sided', method='t'):
    """
    Hypothesis test for the mean of a sample.

    Null-hypothesis
        mean(`x`) == `H0_mean`

    Parameters
    ----------
    x : array_like
        Test mean of this sample.
    H0_mean : float, optional
        Null-hypothesis mean.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Sense of alternative hypothesis.
    method : {'t', 'z'}, optional
        | 't'
        |   Test based on Student's t distribution.
            Calls :func:`scipy..ttest_1samp <scipy.stats.ttest_1samp>`.
        | 'z'
        |   Test based on z-score.
            Calls :func:`sm..ztest <statsmodels.stats.weightstats.ztest>`.

    Returns
    -------
    :class:`mqr.inference.hyptest.HypothesisTest`
    """
    if method == 't':
        statistic, pvalue = scipy.stats.ttest_1samp(
            a=x,
            popmean=H0_mean,
            alternative=alternative)
    elif method == 'z':
        alt = interop.alternative(alternative, lib='statsmodels')
        statistic, pvalue = statsmodels.stats.weightstats.ztest(
            x1=x,
            value=H0_mean,
            alternative=alt)
    else:
        raise ValueError(util.method_error_msg(method, ['t', 'z']))

    x_name = util.var_name(x, 'x')
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

    Null-hypothesis
        mean(`x`) - mean(`y`) == `H0_diff`

    Parameters
    ----------
    x, y : array_like
        Test the difference between means of these samples.
    H0_diff : float, optional
        Null-hypothesis difference.
    pooled : bool, optional
        `True` when the samples are taken from the same population, `False` otherwise.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Sense of alternative hypothesis.
    method : {'t', 'z'}, optional
        | 't'
        |   Test based on Student's t distribution
            Calls :func:`sm..ttest_ind <statsmodels.stats.weightstats.ttest_ind>`.
        | 'z'
        |   Test based on z-score.
            Calls :func:`sm..ztest <statsmodels.stats.weightstats.ztest>`.

    Returns
    -------
    :class:`mqr.inference.hyptest.HypothesisTest`
    """
    alt = interop.alternative(alternative, 'statsmodels')
    usevar = 'pooled' if pooled else 'unequal'
    if method == 't':
        statistic, pvalue, _dof = statsmodels.stats.weightstats.ttest_ind(
            x1=x,
            x2=y,
            alternative=alt,
            usevar=usevar,
            value=H0_diff)
    elif method == 'z':
        statistic, pvalue = statsmodels.stats.weightstats.ztest(
            x1=x,
            x2=y,
            value=H0_diff,
            alternative=alt,
            usevar=usevar)
    else:
        raise ValueError(util.method_error_msg(method, ['t', 'z']))
    
    x_name = util.var_name(x, 'x')
    y_name = util.var_name(y, 'y')
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

    Null-hypothesis
        mean(`x`) == mean(`y`)

    Parameters
    ----------
    x, y : array_like
        Test the difference between means of these samples.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Sense of alternative hypothesis.
    method : 't', optional
        Only Student's t is currently available.
        Calls :func:`scipy..ttest_rel <scipy.stats.ttest_rel>`.

    Returns
    -------
    :class:`mqr.inference.hyptest.HypothesisTest`
    """
    if method != 't':
        raise ValueError(util.method_error_msg(method, ['t']))

    result = scipy.stats.ttest_rel(x, y, alternative=alternative)

    x_name = util.var_name(x, 'x')
    y_name = util.var_name(y, 'y')
    return HypothesisTest(
        description='difference between means (paired)',
        alternative=alternative,
        method='t',
        sample_stat=f'mean({x_name}) - mean({y_name})',
        sample_stat_target=0,
        sample_stat_value=x.mean()-y.mean(),
        stat=result.statistic,
        pvalue=result.pvalue,)
