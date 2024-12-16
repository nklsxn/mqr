from mqr.inference.confint import ConfidenceInterval
from mqr.inference.hyptest import HypothesisTest
from mqr.inference.power import TestPower

import mqr.inference.lib.util as util
import mqr.interop.inference as interop

import numpy as np
import scipy
import statsmodels

def power_1sample(effect, nobs, alpha=0.05, alternative='two-sided', method='chi2'):
    '''
    Calculate power for test of variance of a sample.

    Null-hypothesis
        `effect` = var / var_0 == 1

    Parameters
    ----------
    effect : float
        Effect size.
    nobs : float
        Number of observations in sample.
    alpha : float, optional
        Significance level.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Sense of alternative hypothesis.
    method : {'chi2'}, optional
        Only the 'chi2' is available. Calculates the power directly from the
        Chi-squared distribution.

    Returns
    -------
    :class:`mqr.inference.power.TestPower`
    '''
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
        raise ValueError(util.alternative_error_msg(alternative))

    return TestPower(
        name='variance',
        alpha=alpha,
        beta=1-power,
        effect=effect,
        alternative=alternative,
        method='chi2',
        sample_size=nobs)

def power_2sample(var_ratio, nobs, alpha=0.05, alternative='two-sided', method='f'):
    '''
    Calculate power for test of ratio of variances.

    Null-hypothesis
        `effect` = var_1 / var_2 == 1

    Parameters
    ----------
    var_ratio : float
        Effect size.
    nobs : float
        Number of observations in sample.
    alpha : float, optional
        Significance level.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Sense of alternative hypothesis.
    method : {'f'}, optional
        Only 'f' is available. Calculates the power directly from the
        F distribution.

    Returns
    -------
    :class:`mqr.inference.power.TestPower`
    '''
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
        raise ValueError(util.alternative_error_msg(alternative))

    return TestPower(
        name='ratio of variances',
        alpha=alpha,
        beta=1-power,
        effect=var_ratio,
        alternative=alternative,
        method='f',
        sample_size=nobs)

def size_1sample(effect, alpha, beta, alternative='two-sided', method='chi2'):
    '''
    Calculate sample size for test of variance of a sample.

    Null-hypothesis
        `effect` == var / var_0 == 1

    Parameters
    ----------
    effect : float
        Required effect size.
    alpha : float
        Required significance.
    beta : float
        Required beta (1 - power).
    alternative : {'two-sided', 'less', 'greater'}, optional
        Sense of alternative hypothesis.
    method : {'chi2'}, optional
        Numerically solves :func:`power_1sample` (passing `method`) equal to
        1 - 'beta' by changing `n`. Only 'chi2' is available.

    Returns
    -------
    :class:`mqr.inference.power.TestPower`
    '''
    if (effect < 1) and (alternative == 'greater'):
        raise ValueError('alternative "greater" not valid when effect < 1')
    elif (effect > 1) and (alternative == 'less'):
        raise ValueError('alternative "less" not valid when effect > 1')

    def beta_fn(n):
        power = power_1sample(
            effect=effect,
            nobs=n,
            alpha=alpha,
            alternative=alternative,
            method=method)
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

def size_2sample(var_ratio, alpha, beta, alternative='two-sided', method='f'):
    '''
    Calculate sample size for test of ratio of variances.

    Null-hypothesis
        `var_ratio` = var_1 / var_2 == 1

    Parameters
    ----------
    var_ratio : float
        Required effect size.
    alpha : float
        Required significance.
    beta : float
        Required beta (1 - power).
    alternative : {'two-sided', 'less', 'greater'}, optional
        Sense of alternative hypothesis.
    method : {'f'}, optional
        Numerically solves :func:`power_2sample` (passing `method`) equal to
        1 - 'beta' by changing `n`. Only 'f' is available.

    Returns
    -------
    :class:`mqr.inference.power.TestPower`
    '''
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

def confint_1sample(x, conf=0.95, bounded='both', method='chi2'):
    '''
    Confidence interval for the variance of a sample.

    Parameters
    ----------
    x : array_like
        Calcaulate interval for the variance of this sample.
    conf : float, optional
        Confidence level that determines the width of the interval.
    bounded : {'both', 'below', 'above'}, optional
        Which sides of the interval to close.
    method : {'chi2'}, optional
        Only 'chi2' for an exact interval based on the chi-squared distribution.

    Returns
    -------
    :class:`mqr.inference.confint.ConfidenceInterval`
    '''
    if method == 'chi2':
        alpha = 1 - conf
        nobs = len(x)
        dof = nobs - 1
        s2 = np.var(x, ddof=1)
        dist = scipy.stats.chi2(dof)
        if bounded == 'both':
            lower, upper = dof * s2 / dist.ppf([1 - alpha / 2, alpha / 2])
        elif bounded == 'below':
            lower = dof * s2 / dist.ppf(1 - alpha)
            upper = np.inf
        elif bounded == 'above':
            lower = 0.0
            upper = dof * s2 / dist.ppf(alpha)
        else:
            raise ValueError(util.bounded_error_msg(bounded))
    else:
        raise ValueError(util.method_error_msg(method, ['chi2']))

    return ConfidenceInterval(
        name='variance',
        method=method,
        value=s2,
        lower=lower,
        upper=upper,
        conf=conf,
        bounded=bounded)

def confint_2sample(x, y, conf=0.95, bounded='both', method='f'):
    '''
    Confidence interval for the ratio of variances of two samples.

    Parameters
    ----------
    x, y : array_like
        Calcaulate interval for the ratio of variances of these two samples.
    conf : float, optional
        Confidence level that determines the width of the interval.
    bounded : {'both', 'below', 'above'}, optional
        Which sides of the interval to close.
    method : {'f'}, optional
        Only 'f' for an exact interval based on the F distribution.

    Returns
    -------
    :class:`mqr.inference.confint.ConfidenceInterval`
    '''
    if method == 'f':
        alpha = 1 - conf
        s2x = np.var(x, ddof=1)
        s2y = np.var(y, ddof=1)
        nobsx = len(x)
        nobsy = len(y)
        dofx = nobsx - 1
        dofy = nobsy - 1
        ratio = s2x / s2y
        dist = scipy.stats.f(dofy, dofx)
        if bounded == 'both':
            lower, upper = ratio * dist.ppf([alpha / 2, 1 - alpha / 2])
        elif bounded == 'below':
            lower = ratio * dist.ppf(alpha)
            upper = np.inf
        elif bounded == 'above':
            lower = -np.inf
            upper = ratio * dist.ppf(1 - alpha)
        else:
            raise ValueError(util.bounded_error_msg(bounded))
    else:
        raise ValueError(util.method_error_msg(method, ['f']))

    return ConfidenceInterval(
        name='ratio of variances',
        method=method,
        value=ratio,
        lower=lower,
        upper=upper,
        conf=conf,
        bounded=bounded)

def test_1sample(x, H0_var, alternative='two-sided', method='chi2'):
    '''
    Hypothesis test for the varianve of a sample.

    Null hypothesis
        var(`x`) / `H0_var` == 1

    Parameters
    ----------
    x : array_like
        Test variance of this sample.
    H0_var : float
        Null-hypothesis variance.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Sense of alternative hypothesis.
    method : {'chi2'}, optional
        Only 'chi2' for an exact test based on the Chi-squared distribution.

    Returns
    -------
    :class:`mqr.inference.hyptest.HypothesisTest`
    '''
    if method == 'chi2':
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
            raise ValueError(util.alternative_error_msg(alternative))
    else:
        raise ValueError(util.method_error_msg(method, ['chi2']))

    x_name = util.var_name(x, 'x')
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
    '''
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
        |   F-test from the F distribution of the ratio of equal variances.
        | 'bartlett'
        |   Bartlett's test.
            Calls :func:`scipy..bartlett <scipy.stats.bartlett>`.

    Returns
    -------
    :class:`mqr.inference.hyptest.HypothesisTest`
    '''
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
            raise ValueError(util.alternative_error_msg(alternative))
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
        raise ValueError(util.method_error_msg(method, ['f', 'bartlett']))

    x_name = util.var_name(x, 'x')
    y_name = util.var_name(y, 'y')
    return HypothesisTest(
        description=description,
        alternative=alternative,
        method=method,
        sample_stat=f'var({x_name}) {rel} var({y_name})',
        sample_stat_target=tgt,
        sample_stat_value=stat,
        stat=stat,
        pvalue=pvalue,)
