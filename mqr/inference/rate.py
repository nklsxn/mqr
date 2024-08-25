"""
Confidence intervals and hypothesis tests (parametric) for rates of events.
"""

import mqr.inference.lib.rate as rate

from mqr.inference.confint import ConfidenceInterval
from mqr.inference.hyptest import HypothesisTest
from mqr.inference.power import TestPower

import mqr.interop.inference as interop

import numpy as np
import scipy
import statsmodels

import warnings

def power_1sample(ra, H0_rate, nobs, alpha, meas=1.0, alternative='two-sided', method='norm-approx'):
    """
    Calculate power of a test of rate of events.

    Null-hypothesis: `ra / H0_rate == 1`.

    Arguments
    ---------
    ra (float) -- Alternative hypothesis rate, forming effect size.
    H0_rate (float) -- Null-hypothesis rate.
    nobs (int) -- Number of observations.
    alpha (float) -- Required significance.

    Optional
    --------
    meas (float) -- Extent of one period in observation. (Default 1.)
    alternative (float) -- Sense of alternative hypothesis. One of "two-sided",
        "less" or "greater". (Default "two-sided".)
    method (str) -- Test method. Only "norm-approx", the normal approximation to
        the binomial distribution, is available.

    Returns
    -------
    mqr.power.TestPower
    """
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
            raise ValueError(f'Invalid alternative "{alternative}". Use "two-sided" (default), "less", or "greater".')
    else:
        raise ValueError(f'method {method} not available')

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
    """
    Calculate power of a test of difference or ratio of two rates of events.

    Null-hypothesis: `r1 - r2 == H0_value` (difference),
                     `r1 / r2 == H0_value` (ratio).

    Uses `statsmodels.stats.rates.power_poisson_diff_2indep`,
    and `statsmodels.stats.rates.power_poisson_ratio_2indep` (statsmodels.org).

    Arguments
    ---------
    r1 (float) -- First rate.
    r2 (float) -- Second rate.
    nobs (int) -- Number of observations.
    alpha (float) -- Required significance.
    beta (float) -- Required beta (1 - power).

    Optional
    --------
    H0_value (float) -- Null-hypothesis rate. (Default 0 for 'diff', 1 for 'ratio'.)
    alternative (float) -- Sense of alternative hypothesis. One of "two-sided",
        "less" or "greater". (Default "two-sided".)
    method (str) -- Test method (default "score"). See statsmodels documentation.

    Returns
    -------
    mqr.power.TestPower
    """
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
        raise ValueError(f'Invalid compare "{copmare}". Use "diff" (default), or "ratio".')
    return TestPower(
        name=f'{desc} rates of events',
        alpha=alpha,
        beta=1-power,
        effect=f'{r1/meas1:g} {sample_stat_sym} {r2/meas2:g} = {sample_stat_value:g}',
        alternative=alternative,
        method=method,
        sample_size=nobs)

def size_1sample(ra, H0_rate, alpha, beta, meas=1.0, alternative='two-sided', method='norm-approx'):
    """
    Calculate sample size for test of rate of events.

    Null-hypothesis: `ra / H0_rate == 1`.

    Arguments
    ---------
    ra (float) -- Alternative hypothesis rate, forming effect size.
    H0_rate (float) -- Null-hypothesis rate.
    alpha (float) -- Required significance.
    beta (float) -- Required beta (1 - power).

    Optional
    --------
    alternative (float) -- Sense of alternative hypothesis. One of "two-sided",
        "less" or "greater". (Default "two-sided".)
    method (str) -- Test method (default "norm-approx"): "norm-approx", or "chi2".

    Returns
    -------
    mqr.power.TestPower
    """
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
            raise ValueError(f'Invalid alternative "{alternative}". Use "two-sided" (default), "less", or "greater".')
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
        raise ValueError(f'Invalid method "{method}"')

    return TestPower(
        name='rate of events',
        alpha=alpha,
        beta=beta,
        effect=f'{ra:g} / {H0_rate:g} = {ra/H0_rate:g}',
        alternative=alternative,
        method=method,
        sample_size=nobs_opt,)

def size_2sample(r1, r2, alpha, beta, H0_value=None, alternative='two-sided', method='score', compare='diff'):
    """
    Calculate sample size for difference of two rates of events.

    Null-hypothesis: `r1 - r2 == H0_value` when `compare == 'diff'`,
                     `r1 / r2 == H0_value` when `compare == 'ratio'`.

    Uses `statsmodels.stats.rates.power_poisson_diff_2indep`,
    and `statsmodels.stats.rates.power_poisson_ratio_2indep` (statsmodels.org).

    Arguments
    ---------
    r1 (float) -- First rate.
    r2 (float) -- Second rate.
    alpha (float) -- Required significance.
    beta (float) -- Required beta (1 - power).

    Optional
    --------
    H0_value (float) -- Null-hypothesis rate. (Default 0 for `diff`, 1 for `ratio`.)
    alternative (float) -- Sense of alternative hypothesis. One of "two-sided",
        "less" or "greater". (Default "two-sided".)
    method (str) -- Test method (default "score"):
        "z",
        otherwise, all other methods are passed to statsmodels.
    compare (str) -- Form of the null-hypothesis. Either 'diff' (default) or 'ratio'.

    Returns
    -------
    mqr.power.TestPower
    """
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
        Za = -scipy.stats.norm().ppf(alpha)
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
        raise ValueError(f'`compare` argument ({compare}) not recognised')

    return TestPower(
        name=f'{desc} rates of events',
        alpha=alpha,
        beta=beta,
        effect=f'{r1:g} {sample_stat_sym} {r2:g} = {sample_stat_value:g}',
        alternative=alternative,
        method=method,
        sample_size=nobs_opt)

def confint_1sample(count, n, meas=1.0, conf=0.95, bounded='both', method='wald-cc'):
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
    method (str) -- Test method (default "wald-cc"). One of
        "chi2" the Chi2 interval, see [2], rationale in [1],
        "exact" the exact method, searches for rates that match the given values
            at the desired confidence level, see method 9 in [4], recommended
            for small `count`,
        "wald-cc" Modified wald method see [1], recommended for small `count`,
        (other) everything else is passed to `statsmodels.stats.rates.confint_poisson`,
            which supports only an interval, not one-sided bounds.

    Returns
    -------
    mqr.confint.ConfidenceInterval

    References
    ----------
    [1] Patil, V. V., & Kulkarni, H. V. (2012).
        Comparison of confidence intervals for the Poisson mean: some new aspects.
        REVSTAT-Statistical Journal, 10(2), 211-22.
    [2] Garwood, F. (1936).
        Fiducial limits for the Poisson distribution.
        Biometrika, 28(3/4), 437-442.
    [3] Schwertman, N. C., & Martinez, R. (1994).
        Approximate confidence intervals for the difference in two Poisson parameters.
        Journal of statistical computation and simulation, 50(3-4), 235-247.
    [4] Barker, L. (2002).
        A comparison of nine confidence intervals for a Poisson parameter when
        the expected number of events is ≤ 5.
        The American Statistician, 56(2), 85-89.
    """
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
        method=method,
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
    method (str) -- Test method (default "z"): "z", or "chi2".

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
                    H0_value=None, alternative='two-sided', method='score', compare='ratio'):
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
