import mqr.inference.lib.proportion as proportion

from mqr.inference.confint import ConfidenceInterval
from mqr.inference.hyptest import HypothesisTest
from mqr.inference.power import TestPower

import mqr.inference.lib.util as util
import mqr.interop.inference as interop
from mqr.utils import clip_where

import numpy as np
import scipy
import statsmodels

def power_1sample(pa, H0_prop, nobs, alpha=0.05, alternative='two-sided', method='norm-approx'):
    """
    Calculate power of a test of proportion in a sample.

    Null-hypothesis
        `pa` - `H0_prop` == 0

    Parameters
    ----------
    pa : float
        Alternative hypothesis proportion, forming effect size.
    H0_prop : float
        Null-hypothesis proportion.
    nobs : int
        Number of observations.
    alpha : float
        Required significance.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Sense of alternative hypothesis.
    method : {'norm-approx'}, optional
        Test method. Only 'norm-approx', the normal approximation to the
        binomial distribution is implemented.

    Returns
    -------
    :class:`mqr.inference.power.TestPower`
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
            raise ValueError(util.alternative_error_msg(alternative))
    else:
        raise ValueError(util.method_error_msg(method, ['norm-approx']))

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

    Null-hypothesis
        `p1` - `p2` == 0

    Parameters
    ----------
    p1 : float
        First proportion.
    p2 : float
        Second proportion.
    nobs : int
        Number of observations.
    alpha : float
        Required significance.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Sense of alternative hypothesis.
    method : {'norm-approx'}, optional
        Test method. Only 'norm-approx', the normal approximation to the
        binomial distribution, is implemented.

    Returns
    -------
    :class:`mqr.inference.power.TestPower`
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
            raise ValueError(util.alternative_error_msg(alternative))
    else:
        raise ValueError(util.method_error_msg(method, ['norm-approx']))

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

    Null-hypothesis
        `pa` - `H0_prop` == 0

    Parameters
    ----------
    pa : float
        Alternative proportion, forming the effect size with `p0`.
    p0 : float
        Null proportion.
    alpha : float
        Required significance.
    beta : float
        Required beta (1 - power).
    alternative : {'two-sided', 'less', 'greater'}, optional
        Sense of alternative hypothesis.
    method : {'norm-approx', 'arcsin'}, optional
        | 'norm-approx'
        |   Normal approximation to the binomial distribution.
            Solves :func:`power_1sample` equal to requested power by changing `nobs`.
        | 'arcsin'
        |   Inverse-sine approximation to the binomial distribution.

    Returns
    -------
    :class:`mqr.inference.power.TestPower`
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
    elif (method == 'arcsin') or (method == 'invsin-approx'):
        if alternative == 'less' or alternative == 'greater':
            crit = alpha
        elif alternative == 'two-sided':
            crit = alpha / 2
        else:
            raise ValueError(util.alternative_error_msg(alternative))
        dist = scipy.stats.norm()

        Zb = dist.ppf(1-beta)
        Za = -dist.ppf(crit)
        num = Za + Zb
        denom = np.arcsin(np.sqrt(pa)) - np.arcsin(np.sqrt(H0_prop))
        nobs_opt = num**2 / denom**2 / 4
    else:
        raise ValueError(util.method_error_msg(method, ['norm-approx', 'arcsin']))

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

    Null-hypothesis
        `p1` - `p2` == 0

    Parameters
    ----------
    p1 : float
        First proportion.
    p2 : float
        Second propotion, forming effect size with `p1`.
    alpha : float
        Required significance.
    beta : float
        Required beta (1 - power).
    alternative : {'two-sided', 'less', 'greater'}, optional
        Sense of alternative hypothesis.
    method : {'norm-approx', 'arcsin'}, optional
        | 'norm-approx'
        |   Numerically solves the normal approximation to the binomial
            distribution by searching over `nobs`.
        | 'arcsin'
        |   Inverse-sine method.

    Returns
    -------
    :class:`mqr.inference.power.TestPower`
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
    elif (method == 'arcsin') or (method == 'invsin-approx'):
        if alternative == 'less' or alternative == 'greater':
            crit = alpha
        elif alternative == 'two-sided':
            crit = alpha / 2
        else:
            raise ValueError(util.alternative_error_msg(alternative))
        dist = scipy.stats.norm()

        Zb = dist.ppf(1-beta)
        Za = -dist.ppf(crit)
        num = Za + Zb
        denom = np.arcsin(np.sqrt(p1)) - np.arcsin(np.sqrt(p2))
        nobs_opt = 2 * num**2 / denom**2 / 4
    else:
        raise ValueError(util.method_error_msg(method, ['norm-approx', 'arcsin']))

    return TestPower(
        name='proportion',
        alpha=alpha,
        beta=beta,
        effect=f'{p1:g} - {p2:g} = {p1-p2:g}',
        alternative=alternative,
        method=method,
        sample_size=nobs_opt)

def confint_1sample(count, nobs, conf=0.95, bounded='both', method='agresti-coull'):
    """
    Confidence interval for proportion `count / nobs`.

    Following [1]_, use the 'wilson-cc' or 'jeffreys' method for small sample
    size, and use the 'agresti-coull' or 'jeffreys' for larger sample sizes.
    The authors recommend n = 40 as the boundary between small and large.

    Parameters
    ----------
    count : int
        Number of "true" observations.
    nobs : int
        Total observations.
    conf : float, optional
        Confidence level that determines the width of the interval.
    bounded : {'both', 'below', 'above'}, optional
        Which sides of the interval to close.
    method : {'agresti-coull', 'jeffreys', 'wilson', 'wilson-cc'}, optional
        | 'agresti-coull'
        |   Agresti-Coull interval, see [1]_.
        | 'jeffreys'
        |   Jeffreys interval (a Bayesian method), see [1]_.
        | 'wilson'
        |   Wilson method without continuity correction (method 3 in [2]_).
        | 'wilson-cc'
        |   Wilson method with continuity correction (method 4 in [2]_).
        | (other)
        |   Everything else is passed to
            :func:`sm..proportion_confint <statsmodels.stats.proportion.proportion_confint>`.

    Returns
    -------
    :class:`mqr.inference.confint.ConfidenceInterval`

    References
    ----------
    .. [1]  Brown, L. D. Cai, T. T. and DasGupta, A. (2001).
            Interval estimation for a binomial proportion.
            Statistical Science, 16(2), 101-133.
    .. [2]  Newcombe, R. G. (1998).
            Two‐sided confidence intervals for the single proportion:
            comparison of seven methods.
            Statistics in medicine, 17(8), 857-872.
    """
    alpha = 1 - conf

    if method == 'agresti-coull':
        lower, upper = proportion.confint_1sample_agresti_coull(count, nobs, conf, bounded)
    elif method == 'jeffreys':
        lower, upper = proportion.confint_1sample_jeffreys(count, nobs, conf, bounded)
    elif method == 'wilson':
        lower, upper = proportion.confint_1sample_wilson(count, nobs, conf, bounded)
    elif method == 'wilson-cc':
        lower, upper = proportion.confint_1sample_wilson_cc(count, nobs, conf, bounded)
    else:
        if bounded == 'both':
            (lower, upper) = statsmodels.stats.proportion.proportion_confint(
                count=count,
                nobs=nobs,
                alpha=alpha,
                method=method)
        else:
            raise ValueError(
                f'Method "{method}" is passed to statsmodels which does not implement '
                'one-sided bounds. ' +
                util.method_error_msg(method, ['agresti-coull', 'jeffreys', 'wilson-cc']))
    value = count / nobs
    return ConfidenceInterval(
        name='proportion',
        method=method,
        value=value,
        lower=lower,
        upper=upper,
        conf=conf,
        bounded=bounded)

def confint_2sample(count1, nobs1, count2, nobs2, conf=0.95, bounded='both', method='newcomb-cc'):
    """
    Confidence interval for difference between proportions `count1 / nobs1 - count2 / nobs2`.

    Parameters
    ----------
    count1 : int
        Number of "true" observations in first sample.
    nobs1 : int
        Total observations in first sample.
    count2 : int
        Number of "true" observations in second sample.
    nobs2 : int
        Total observations in second sample.
    conf : float, optional
        Confidence level that determines the width of the interval.
    bounded : {'both', 'below', 'above'}, optional
        Which sides of the interval to close.
    method : {'agresti-caffo', 'newcomb', 'newcomb-cc'}, optional
        | 'agresti-caffo' (or 'adj-wald')
        |   Agresti-caffo method an adjusted normal approximation.
            See reference [1]_.
        | 'newcomb'
        |   A method presented by Newcomb as method 10 in [2]_.
        | 'newcomb-cc'
        |   Continuity-corrected version of 'newcomb'; method 11 in [2]_.
        | (other)
        |   Everything else is passed to
            :func:`sm..confint_proportions_2indep <statsmodels.stats.proportion.confint_proportions_2indep>`
            for comparison of the difference.

    Returns
    -------
    :class:`mqr.inference.confint.ConfidenceInterval`

    References
    ----------
    .. [1]  Agresti, A., & Caffo, B. (2000).
            Simple and effective confidence intervals for proportions and differences
            of proportions result from adding two successes and two failures.
            The American Statistician, 54(4), 280-288.
    .. [2]  Newcombe, R. G. (1998).
            Interval estimation for the difference between independent proportions:
            comparison of eleven methods.
            Statistics in medicine, 17(8), 873-890.
    """
    if (method == 'agresti-caffo') or (method == 'adj-wald'):
        lower, upper = proportion.confint_2sample_agresti_caffo(count1, nobs1, count2, nobs2, conf, bounded)
    elif method == 'newcomb':
        lower, upper = proportion.confint_2sample_newcomb(count1, nobs1, count2, nobs2, conf, bounded)
    elif method == 'newcomb-cc':
        lower, upper = proportion.confint_2sample_newcomb_cc(count1, nobs1, count2, nobs2, conf, bounded)
    else:
        if bounded == 'both':
            lower, upper = statsmodels.stats.proportion.confint_proportions_2indep(
                count1, nobs1,
                count2, nobs2,
                alpha=1-conf,
                compare='diff',
                method=method)
        else:
            raise ValueError(
                f'Method "{method}" is passed to statsmodels which does not implement '
                'one-sided bounds. ' +
                util.method_error_msg(method, ['agresti-caffo', 'newcomb-cc']))

    value = count1 / nobs1 - count2 / nobs2
    return ConfidenceInterval(
        name='difference between proportions',
        method=method,
        value=value,
        lower=lower,
        upper=upper,
        conf=conf,
        bounded=bounded)

def test_1sample(count, nobs, H0_prop, alternative='two-sided', method='binom'):
    """
    Hypothesis test for the proportion of "true" elements in a sample.

    Null-hypothesis
        `count` / `nobs` == `H0_prop`

    Parameters
    ----------
    count : int
        Number of "true" observations.
    nobs : int
        Total number of observations.
    H0_prop : float
        Null-hypothesis proportion.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Sense of alternative hypothesis. Valid for methods 'binom' and 'z';
        method 'chi2' supports only two-sided tests.
    method : {'binom', 'chi2', 'z'}, optional
        | 'binom'
        |   Calls :func:`sm..binom_test <statsmodels.stats.proportion.binom_test>`.
        | 'chi2'
        |   Calls :func:`sm..proportions_chisquare <statsmodels.stats.proportion.proportions_chisquare>`.
        | 'z'
        |   Calls :func:`sm..proportions_ztest <statsmodels.stats.proportion.proportions_ztest>`.

    Returns
    -------
    :class:`mqr.inference.hyptest.HypothesisTest`
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
            raise ValueError(
                f'Method "{method}" is passed to statsmodels which does not implement '
                'one-sided alternatives. ' +
                util.method_error_msg(method, ['binom', 'z']))
        stat, pvalue, _ = statsmodels.stats.proportion.proportions_chisquare(
            count=count,
            nobs=nobs,
            value=H0_prop)
    else:
        raise ValueError(util.method_error_msg(method, ['binom', 'chi2', 'z']))

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

    Null-hypothesis
        `count1` / `nobs1` - `count2` / `nobs2` == `H0_diff`

    Parameters
    ----------
    count1 : int
        Number of "true" observations in first sample.
    nobs1 : int
        Total number of observations in second sample.
    count2 : int
        Number of "true" observations in second sample.
    nobs2 : int
        Total number of observations in seconds sample.
    H0_diff : float
        Null-hypothesis difference.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Sense of alternative hypothesis.
    method : str, optional
        Passed to :func:`sm..test_proportions_2indep <statsmodels.stats.proportion.test_proportions_2indep>`.

    Returns
    -------
    :class:`mqr.inference.hyptest.HypothesisTest`
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
