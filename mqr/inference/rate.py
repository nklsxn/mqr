import mqr.inference.lib.rate as rate
import mqr.inference.lib.util as util

from mqr.inference.confint import ConfidenceInterval
from mqr.inference.hyptest import HypothesisTest
from mqr.inference.power import TestPower

import mqr.interop.inference as interop

import numpy as np
import scipy
import statsmodels

import warnings

def power_1sample(ra, H0_rate, nobs, alpha, meas=1.0, alternative='two-sided', method='norm-approx'):
    '''
    Calculate power of a test of rate of events.

    Null-hypothesis
        `ra` / `H0_rate` == 1

    Parameters
    ----------
    ra : float
        Alternative hypothesis rate, forming effect size.
    H0_rate : float
        Null-hypothesis rate.
    nobs : int
        Number of observations.
    alpha : float
        Required significance.
    meas : float, optional
        Extent of one period in observation.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Sense of alternative hypothesis.
    method : {'norm-approx'}, optional
        Test method. Only 'norm-approx', the normal approximation to
        the binomial distribution, is available.

    Returns
    -------
    :class:`mqr.inference.power.TestPower`
    '''
    if method == 'norm-approx':
        dist = scipy.stats.norm()
        var_a = ra / (nobs * meas)
        var_0 = H0_rate / (nobs * meas)
        den = np.sqrt(var_a)

        if alternative == 'less':
            z = dist.ppf(1-alpha)
            num = z * np.sqrt(var_0) + ra - H0_rate
            power = 1 - dist.cdf(num/den)
        elif alternative == 'greater':
            z = dist.ppf(1-alpha)
            num = z * np.sqrt(var_0) + H0_rate - ra
            power = 1 - dist.cdf(num/den)
        elif alternative == 'two-sided':
            z = dist.ppf(1-alpha/2)
            num_1 = z * np.sqrt(var_0) + H0_rate - ra
            num_2 = H0_rate - z * np.sqrt(var_0) - ra
            power = 1 - (dist.cdf(num_1/den) - dist.cdf(num_2/den))
        else:
            raise ValueError(util.alternative_error_msg(alternative))
    else:
        raise ValueError(util.method_error_msg(method, ['norm-approx']))

    return TestPower(
        name='rate of events',
        alpha=alpha,
        beta=1-power,
        effect=f'{ra:g} / {H0_rate:g} = {ra/H0_rate:g}',
        alternative=alternative,
        method=method,
        sample_size=nobs)

def power_2sample(r1, r2, nobs, alpha, H0_value=None, meas1=1.0, meas2=1.0,
                  alternative='two-sided', method='score', compare='diff'):
    '''
    Calculate power of a test of difference or ratio of two rates of events.

    Null-hypothesis by compare
        | 'diff'
        |   `r1` - `r2` == `H0_value`
        | 'ratio'
        |   `r1` / `r2` == `H0_value`

    Parameters
    ----------
    r1 : float
        First rate.
    r2 : float
        Second rate.
    nobs : int
        Number of observations.
    alpha : float
        Required significance.
    beta : float
        Required beta (1 - power).
    H0_value : float, optional
        Null-hypothesis rate. Default 0 for 'diff', 1 for 'ratio'.
    alternative : {'two-sided', 'less', 'greater'}
        Sense of alternative hypothesis.
    method : str, optional
        Test method. See statsmodels documentation.
    compare : {'diff', 'ratio'}, optional
        | 'diff'
        |   Compares the difference between rates.
            Calls :func:`sm..power_poisson_diff_2indep <statsmodels.stats.rates.power_poisson_diff_2indep>`.
        | 'ratio'
        |   Compares the ratio of two rates.
            Calls :func:`sm..power_poisson_ratio_2indep <statsmodels.stats.rates.power_poisson_ratio_2indep>`.

    Returns
    -------
    :class:`mqr.inference.power.TestPower`
    '''
    alt = interop.alternative(alternative, lib='statsmodels')
    if compare == 'diff':
        desc = 'difference between'
        sample_stat_sym = '-'
        sample_stat_value = r1 - r2
        if H0_value is None:
            H0_value = 0.0
        power = statsmodels.stats.rates.power_poisson_diff_2indep(
            rate1=r1,
            rate2=r2,
            nobs1=nobs,
            nobs_ratio=1,
            value=H0_value,
            alpha=alpha,
            alternative=alt,
            method_var=method,
            return_results=False)
    elif compare == 'ratio':
        desc = 'ratio of'
        sample_stat_sym = '/'
        sample_stat_value = r1 / r2
        if H0_value is None:
            H0_value = 1.0
        power = statsmodels.stats.rates.power_poisson_ratio_2indep(
            rate1=r1,
            rate2=r2,
            nobs1=nobs,
            nobs_ratio=1,
            value=H0_value,
            alpha=alpha,
            alternative=alt,
            method_var=method,
            return_results=False)
    else:
        raise ValueError(util.compare_error_msg('test'))
    return TestPower(
        name=f'{desc} rates of events',
        alpha=alpha,
        beta=1-power,
        effect=f'{r1/meas1:g} {sample_stat_sym} {r2/meas2:g} = {sample_stat_value:g}',
        alternative=alternative,
        method=method,
        sample_size=nobs)

def size_1sample(ra, H0_rate, alpha, beta, meas=1.0, alternative='two-sided', method='norm-approx'):
    '''
    Calculate sample size for test of rate of events.

    Null-hypothesis
        `ra` / `H0_rate` == 1

    Parameters
    ----------
    ra : float
        Alternative hypothesis rate, forming effect size.
    H0_rate : float
        Null-hypothesis rate.
    alpha : float
        Required significance.
    beta : float
        Required beta (1 - power).
    alternative : {'two-sided', 'less', 'greater'}, optional
        Sense of alternative hypothesis.
    method : {'norm-approx', 'chi2'}, optional
        | 'norm-approx'
        |   Approximation to the normal distribution. Numerically solves
            :func:`power_1sample` equal to 1 - `beta` by searching `nobs`.
        | 'chi2'
        |   Numerically finds the sample size that produces the requested power
            using a Chi-squared distribution.

    Returns
    -------
    :class:`mqr.inference.power.TestPower`
    '''
    if method == 'chi2':
        n = 1
        r = ra / H0_rate if ra > H0_rate else H0_rate / ra
        if alternative == 'less' or alternative == 'greater':
            NP1 = 1 - beta
            DP1 = alpha
        elif alternative == 'two-sided':
            NP1 = 1 - beta
            DP1 = alpha / 2
        else:
            raise ValueError(util.alternative_error_msg(alternative))
        def ratio(n):
            num = scipy.stats.chi2.ppf(NP1, df=n)
            den = scipy.stats.chi2.ppf(DP1, df=n)
            return num / den - r
        df_opt = scipy.optimize.fsolve(ratio, 1)[0]
        num = scipy.stats.chi2.ppf(NP1, df=df_opt)
        nobs_opt = num / 2.0 / np.maximum(ra, H0_rate)
    elif method == 'norm-approx':
        def beta_fn(n):
            return power_1sample(
                ra=ra,
                H0_rate=H0_rate,
                nobs=n,
                alpha=alpha,
                meas=meas,
                alternative=alternative).beta - beta
        nobs_opt = scipy.optimize.fsolve(beta_fn, 1)[0]
    else:
        raise ValueError(util.method_error_msg(method, ['chi2', 'norm-approx']))

    return TestPower(
        name='rate of events',
        alpha=alpha,
        beta=beta,
        effect=f'{ra:g} / {H0_rate:g} = {ra/H0_rate:g}',
        alternative=alternative,
        method=method,
        sample_size=nobs_opt,)

def size_2sample(r1, r2, alpha, beta, H0_value=None, alternative='two-sided', method='score', compare='diff'):
    '''
    Calculate sample size for difference of two rates of events.

    Null-hypothesis by compare
        | 'diff'
        |   `r1` - `r2` == `H0_value`
        | 'ratio'
        |   `r1` / `r2` == `H0_value`

    Parameters
    ----------
    r1 : float
        First rate.
    r2 : float
        Second rate.
    alpha : float
        Required significance.
    beta : float
        Required beta (1 - power).
    H0_value : float, optional
        Null-hypothesis rate. Default 0 for 'diff', 1 for 'ratio'.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Sense of alternative hypothesis.
    method : str, optional
        | 'z'
        |   Approximation based a normal approximation. For this method,
            `compare` must be 'diff' and `H0_value` must be 0.
        | (other)
        |   All other methods are passed as parameters to :func:`power_2sample`
            while that function is solved numerically equal to the requested
            power by varying `nobs`.
    compare : {'diff', 'ratio'}, optional
        | 'diff'
        |   Compare the difference between rates.
        | 'ratio'
        |   Compare the ratio of rates.

    Returns
    -------
    :class:`mqr.inference.power.TestPower`
    '''
    if H0_value is None:
        if compare == 'diff':
            H0_value = 0.0
        elif compare == 'ratio':
            H0_value = 1.0

    if method == 'z':
        if compare != 'diff':
            raise ValueError(f'comparison "{compare}" not available with `z` method')
        if not np.isclose(H0_value, 0):
            raise ValueError(f'H0_value "{H0_value}" must be 0 with `z` method')
        if alternative != 'two-sided':
            crit = alpha
        else:
            crit = alpha / 2
        Zb = scipy.stats.norm().ppf(1-beta)
        Za = -scipy.stats.norm().ppf(crit)
        num = Za * np.sqrt(r2) + Zb * np.sqrt(r1)
        nobs_opt = 2 * num**2 / (r1 - r2)**2
    else:
        def beta_fn(nobs):
            power = power_2sample(
                r1,
                r2,
                nobs,
                alpha,
                H0_value=H0_value,
                alternative=alternative,
                method=method,
                compare=compare)
            return power.beta - beta
        nobs_opt = scipy.optimize.fsolve(beta_fn, 2)[0]

    if compare == 'diff':
        desc = 'difference between'
        sample_stat_sym = '-'
        sample_stat_value = r1 - r2
    elif compare == 'ratio':
        desc = 'ratio of'
        sample_stat_sym = '/'
        sample_stat_value = r1 / r2
    else:
        raise ValueError(util.compare_error_msg(compare))

    return TestPower(
        name=f'{desc} rates of events',
        alpha=alpha,
        beta=beta,
        effect=f'{r1:g} {sample_stat_sym} {r2:g} = {sample_stat_value:g}',
        alternative=alternative,
        method=method,
        sample_size=nobs_opt)

def confint_1sample(count, n, meas=1.0, conf=0.95, bounded='both', method='wald-cc'):
    '''
    Confidence interval for rate `count / n / meas`.

    Parameters
    ----------
    count : int
        Number of events.
    n : int
        Number of periods over which events were counted.
    meas : float, optional
        Extent of one period of observation.
    conf : float, optional
        Confidence level that determines the width of the interval.
    method : {'chi2', 'exact', 'wald', 'wald-cc'}, optional
        | 'chi2'
        |   Chi2 interval, see [2]_.
        | 'exact'
        |   Exact method, recommended for small `count`.
            Implements method 9 in [3]_.
        | 'wald-cc'
        |   Wald method with continuity correction, recommended for small `count`.
            Implements method 5 in [1]_.
        | (other)
        |   Everything else is passed to
            :func:`sm..confint_poisson <statsmodels.stats.rates.confint_poisson>`,
            which supports only two-sided intervals.

    Notes
    -----
    For a discussion on the benefits and disadvantages of various intervals,
    including these, see [1]_ and [3]_.

    Returns
    -------
    :class:`mqr.inference.confint.ConfidenceInterval`

    References
    ----------
    .. [1]  Patil, V. V., & Kulkarni, H. V. (2012).
            Comparison of confidence intervals for the Poisson mean: some new aspects.
            REVSTAT-Statistical Journal, 10(2), 211-22.
    .. [2]  Garwood, F. (1936).
            Fiducial limits for the Poisson distribution.
            Biometrika, 28(3/4), 437-442.
    .. [3]  Barker, L. (2002).
            A comparison of nine confidence intervals for a Poisson parameter when
            the expected number of events is ≤ 5.
            The American Statistician, 56(2), 85-89.
    '''
    value = count / n / meas
    alpha = 1 - conf
    if method == 'chi2':
        if count <= 4:
            msg = (
                'This method is recommended when `count > 4`. '
                'Consider using method "exact" or "wald-cc". See [1].')
            warnings.warn(msg)
        (lower, upper) = rate.confint_1sample_chi2(count, n, meas, conf, bounded)
    elif method == 'exact':
        (lower, upper) = rate.confint_1sample_exact(count, n, meas, conf, bounded)
    elif method == 'wald-cc':
        (lower, upper) = rate.confint_1sample_wald_cc(count, n, meas, conf, bounded)
    else:
        (lower, upper) = statsmodels.stats.rates.confint_poisson(
            count=count,
            exposure=n*meas,
            method=method,
            alpha=alpha)
    return ConfidenceInterval(
        name='rate of events',
        method=method,
        value=value,
        lower=lower,
        upper=upper,
        conf=conf,
        bounded=bounded)

def confint_2sample(count1, n1, count2, n2, meas1=1.0, meas2=1.0,
                    conf=0.95, bounded='both', method='wald'):
    '''
    Confidence interval for difference of rates `count1 / n1 / meas1 - count2 / n2 / meas2`.

    Parameters
    ----------
    count1 : int
        Number of events in first observation.
    n1 : int
        Number of periods over which first events were counted.
    count2 : int
        Number of events in second observation.
    n2 : int
        Number of periods over which second events were counted.
    meas1 : float, optional
        Extent of one period in first observation.
    meas2 : float, optional
        Extent of one period in second observation.
    conf : float, optional
        Confidence level that determines the width of the interval.
    method : str, optional
        | 'wald'
        |   Wald's normal approximation, equation (9) in [1]_.
        | 'wald-moment'
        |   Based on Wald, equation (8) in [1]_. Use for small/zero counts.
        | (other)
        |   Everything else is passed to
            :func:`sm..confint_poisson_2indep <statsmodels.stats.rates.confint_poisson_2indep>`,
            which supports only two-sided intervals.

    Returns
    -------
    :class:`mqr.inference.confint.ConfidenceInterval`

    References
    ----------
    .. [1]  Krishnamoorthy, K., & Lee, M. (2013).
            New approximate confidence intervals for the difference between
            two Poisson means and comparison.
            Journal of Statistical Computation and Simulation, 83(12), 2232-2243.
    '''
    if method == 'wald':
        (lower, upper) = rate.confint_2sample_wald(
            count1, n1, count2, n2,
            meas1, meas2,
            conf, bounded)
    elif method == 'wald-moment':
        (lower, upper) = rate.confint_2sample_wald_moment(
            count1, n1, count2, n2,
            meas1, meas2,
            conf, bounded)
    else:
        if bounded != 'both':
            raise ValueError(
                f'Method "{method}" is passed to statsmodels which does not implement '
                'one-sided bounds. ' +
                util.method_error_msg(method, ['wald', 'wald-moment']))
        alpha = 1 - conf
        (lower, upper) = statsmodels.stats.rates.confint_poisson_2indep(
            count1=count1,
            exposure1=n1*meas1,
            count2=count2,
            exposure2=n2*meas2,
            method=method,
            compare='diff',
            alpha=alpha)

    value = count1 / n1 / meas1 - count2 / n2 / meas2
    return ConfidenceInterval(
        name='difference between rates of events',
        method=method,
        value=value,
        lower=lower,
        upper=upper,
        conf=conf,
        bounded=bounded)

def test_1sample(count, n, meas=1.0, H0_rate=1.0, alternative='two-sided', method='exact-c'):
    '''
    Hypothesis test for the rate of events.

    Null-hypothesis
        `count` / `n` / `meas` == `H0_rate`

    Calls :func:`sm..test_poisson <statsmodels.stats.rates.test_poisson>`.

    Parameters
    ----------
    count : int
        Number of events.
    n : int
        Number of periods over which events were counted.
    meas : float, optional
        Extent of one period of observation.
    H0_rate : float, optional
        Null-hypothesis rate.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Sense of alternative hypothesis.
    method : str, optional
        Test method. See statsmodels docs.

    Returns
    -------
    :class:`mqr.inference.hyptest.HypothesisTest`
    '''
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
                 H0_value=None, alternative='two-sided', method='score', compare='diff'):
    '''
    Hypothesis test for equality of rates.

    Null-hypothesis by compare
        | 'diff'
        |   `count1` / `n1` / `meas1` - `count2` / `n2` / `meas2` == `H0_value`
        | 'ratio'
        |   `count1` / `n1` / `meas1` / (`count2` / `n2` / `meas2`) == `H0_value`

    Calls :func:`sm..test_poisson_2indep <statsmodels.stats.rates.test_poisson_2indep>`.

    Parameters
    ----------
    count1 : int
        Number of events in first observation.
    n1 : int
        Number of periods over which first events were counted.
    count2 : int
        Number of events in second observation.
    n2 : int
        Number of periods over which second events were counted.
    meas1 : float, optional
        Extent of one period in first observation.
    meas2 : float, optional
        Extent of one period in second observation.
    H0_value : float, optional
        Null-hypothesis value. Default 1 when compare is 'ratio', 0 when compare
        is 'diff'.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Sense of alternative hypothesis.
    method : str, optional
        See statsmodels docs.
    compare : {'diff', 'ratio'}, optional
        | 'diff'
        |   Compare the difference between rates.
        | 'ratio'
        |   Compare the rate of ratios.

    Returns
    -------
    :class:`mqr.inference.confint.ConfidenceInterval`
    '''
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
        raise ValueError(util.compare_error_msg(compare))

    return HypothesisTest(
        description=f'{desc} rates of events',
        alternative=alternative,
        method=method,
        sample_stat=f'rate1 {sample_stat_sym} rate2',
        sample_stat_target=H0_value,
        sample_stat_value=sample_stat_value,
        stat=res.statistic,
        pvalue=res.pvalue,)
