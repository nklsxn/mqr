'''
Check call-throughs.
'''

import numbers
import numpy as np
import pytest

import mqr

def test_power_1sample():
    ra = 0.8
    hyp_rate = 0.5
    alpha = 0.05
    meas = 2.0
    nobs = 45

    alternative = 'two-sided'
    method = 'norm-approx'
    res = mqr.inference.rate.power_1sample(ra, hyp_rate, nobs, alpha, meas=meas, alternative=alternative, method=method)
    assert res.name == 'rate of events'
    assert res.alpha == alpha
    assert res.beta == pytest.approx(0.0512867)
    assert res.effect == '0.8 / 0.5 = 1.6'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_size == nobs

    alternative = 'greater'
    res = mqr.inference.rate.power_1sample(ra, hyp_rate, nobs, alpha, meas=meas, alternative=alternative, method=method)
    assert res.name == 'rate of events'
    assert res.alpha == alpha
    assert res.beta == pytest.approx(0.0299445)
    assert res.effect == '0.8 / 0.5 = 1.6'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_size == nobs

    ra = 0.3
    alternative = 'less'
    res = mqr.inference.rate.power_1sample(ra, hyp_rate, nobs, alpha, meas=meas, alternative=alternative, method=method)
    assert res.name == 'rate of events'
    assert res.alpha == alpha
    assert res.beta == pytest.approx(0.0900244)
    assert res.effect == '0.3 / 0.5 = 0.6'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_size == nobs

def test_power_2sample():
    r1 = 0.5
    r2 = 0.4
    nobs = 100
    alpha = 0.02

    compare = 'diff'
    H0_value = 0.0
    res = mqr.inference.rate.power_2sample(r1, r2, nobs, alpha, H0_value, compare=compare)
    assert res.name == 'difference between rates of events'
    assert res.alpha == alpha
    assert isinstance(res.beta, numbers.Number)
    assert res.effect == '0.5 - 0.4 = 0.1'
    assert res.alternative == 'two-sided'
    assert res.method == 'score'
    assert res.sample_size == nobs

    H0_value = None
    res = mqr.inference.rate.power_2sample(r1, r2, nobs, alpha, H0_value, compare=compare)
    assert res.name == 'difference between rates of events'
    assert res.alpha == alpha
    assert isinstance(res.beta, numbers.Number)
    assert res.effect == '0.5 - 0.4 = 0.1'
    assert res.alternative == 'two-sided'
    assert res.method == 'score'
    assert res.sample_size == nobs

    compare = 'ratio'
    H0_value = 1.0
    res = mqr.inference.rate.power_2sample(r1, r2, nobs, alpha, H0_value, compare=compare)
    assert res.name == 'ratio of rates of events'
    assert res.alpha == alpha
    assert isinstance(res.beta, numbers.Number)
    assert res.effect == '0.5 / 0.4 = 1.25'
    assert res.alternative == 'two-sided'
    assert res.method == 'score'
    assert res.sample_size == nobs

    H0_value = None
    res = mqr.inference.rate.power_2sample(r1, r2, nobs, alpha, H0_value, compare=compare)
    assert res.name == 'ratio of rates of events'
    assert res.alpha == alpha
    assert isinstance(res.beta, numbers.Number)
    assert res.effect == '0.5 / 0.4 = 1.25'
    assert res.alternative == 'two-sided'
    assert res.method == 'score'
    assert res.sample_size == nobs

def test_size_1sample():
    ra = 0.8
    hyp_rate = 0.5
    alpha = 0.05
    beta = 0.20

    alternative = 'two-sided'
    method = 'chi2'
    res = mqr.inference.rate.size_1sample(ra, hyp_rate, alpha, beta, alternative=alternative, method=method)
    assert res.name == 'rate of events'
    assert res.alpha == alpha
    assert res.beta == beta
    assert res.effect == '0.8 / 0.5 = 1.6'
    assert res.alternative == alternative
    assert res.method == method
    assert np.ceil(res.sample_size) == 55

    alternative = 'greater'
    res = mqr.inference.rate.size_1sample(ra, hyp_rate, alpha, beta, alternative=alternative, method=method)
    assert res.name == 'rate of events'
    assert res.alpha == alpha
    assert res.beta == beta
    assert res.effect == '0.8 / 0.5 = 1.6'
    assert res.alternative == alternative
    assert res.method == method
    assert np.ceil(res.sample_size) == 43

    ra = 0.3
    alternative = 'less'
    res = mqr.inference.rate.size_1sample(ra, hyp_rate, alpha, beta, alternative=alternative, method=method)
    assert res.name == 'rate of events'
    assert res.alpha == alpha
    assert res.beta == beta
    assert res.effect == '0.3 / 0.5 = 0.6'
    assert res.alternative == alternative
    assert res.method == method
    assert np.ceil(res.sample_size) == 59

    ra = 0.8
    alternative = 'two-sided'
    method = 'norm-approx'
    res = mqr.inference.rate.size_1sample(ra, hyp_rate, alpha, beta, alternative=alternative, method=method)
    assert res.name == 'rate of events'
    assert res.alpha == alpha
    assert res.beta == beta
    assert res.effect == '0.8 / 0.5 = 1.6'
    assert res.alternative == alternative
    assert res.method == method
    assert np.ceil(res.sample_size) == 51

    alternative = 'greater'
    res = mqr.inference.rate.size_1sample(ra, hyp_rate, alpha, beta, alternative=alternative, method=method)
    assert res.name == 'rate of events'
    assert res.alpha == alpha
    assert res.beta == beta
    assert res.effect == '0.8 / 0.5 = 1.6'
    assert res.alternative == alternative
    assert res.method == method
    assert np.ceil(res.sample_size) == 41

    ra = 0.3
    alternative = 'less'
    res = mqr.inference.rate.size_1sample(ra, hyp_rate, alpha, beta, alternative=alternative, method=method)
    assert res.name == 'rate of events'
    assert res.alpha == alpha
    assert res.beta == beta
    assert res.effect == '0.3 / 0.5 = 0.6'
    assert res.alternative == alternative
    assert res.method == method
    assert np.ceil(res.sample_size) == 66

def test_size_2sample():
    r1 = 0.8
    r2 = 0.5
    alpha = 0.05
    beta = 0.20
    effect = 0.2

    alternative = 'two-sided'
    res = mqr.inference.rate.size_2sample(r1, r2, alpha, beta, effect, alternative)
    assert res.name == 'difference between rates of events'
    assert res.alpha == alpha
    assert res.beta == beta
    assert res.effect == '0.8 - 0.5 = 0.3'
    assert res.alternative == alternative
    assert res.method == 'score'
    assert isinstance(res.sample_size, numbers.Number)

    alternative = 'two-sided'
    res = mqr.inference.rate.size_2sample(r1, r2, alpha, beta, effect, alternative)
    assert res.name == 'difference between rates of events'
    assert res.alpha == alpha
    assert res.beta == beta
    assert res.effect == '0.8 - 0.5 = 0.3'
    assert res.alternative == alternative
    assert res.method == 'score'
    assert isinstance(res.sample_size, numbers.Number)

    effect = 0.0
    method = 'z'
    alternative = 'less'
    res = mqr.inference.rate.size_2sample(r1, r2, alpha, beta, effect, alternative, method=method)
    assert res.name == 'difference between rates of events'
    assert res.alpha == alpha
    assert res.beta == beta
    assert res.effect == '0.8 - 0.5 = 0.3'
    assert res.alternative == alternative
    assert res.method == method
    assert isinstance(res.sample_size, numbers.Number)

    alternative = 'two-sided'
    res = mqr.inference.rate.size_2sample(r1, r2, alpha, beta, effect, alternative, method=method)
    assert res.name == 'difference between rates of events'
    assert res.alpha == alpha
    assert res.beta == beta
    assert res.effect == '0.8 - 0.5 = 0.3'
    assert res.alternative == alternative
    assert res.method == method
    assert isinstance(res.sample_size, numbers.Number)

    effect = 0.2
    method = 'score'
    res = mqr.inference.rate.size_2sample(r1, r2, alpha, beta, effect, alternative, method=method)
    assert res.name == 'difference between rates of events'
    assert res.alpha == alpha
    assert res.beta == beta
    assert res.effect == '0.8 - 0.5 = 0.3'
    assert res.alternative == alternative
    assert res.method == 'score'
    assert isinstance(res.sample_size, numbers.Number)

    res = mqr.inference.rate.size_2sample(r1, r2, alpha, beta, effect, alternative, method=method)
    assert res.name == 'difference between rates of events'
    assert res.alpha == alpha
    assert res.beta == beta
    assert res.effect == '0.8 - 0.5 = 0.3'
    assert res.alternative == alternative
    assert res.method == 'score'
    assert isinstance(res.sample_size, numbers.Number)

    compare = 'diff'
    res = mqr.inference.rate.size_2sample(r1, r2, alpha, beta, None, alternative, compare=compare)
    assert res.name == 'difference between rates of events'
    assert res.alpha == alpha
    assert res.beta == beta
    assert res.effect == '0.8 - 0.5 = 0.3'
    assert res.alternative == alternative
    assert res.method == 'score'
    assert isinstance(res.sample_size, numbers.Number)

    compare = 'ratio'
    res = mqr.inference.rate.size_2sample(r1, r2, alpha, beta, None, alternative, compare=compare)
    assert res.name == 'ratio of rates of events'
    assert res.alpha == alpha
    assert res.beta == beta
    assert res.effect == '0.8 / 0.5 = 1.6'
    assert res.alternative == alternative
    assert res.method == 'score'
    assert isinstance(res.sample_size, numbers.Number)

def test_confint_1sample():
    count = 5
    n = 20
    meas = 2.0
    conf = 0.90
    bounded = 'both'
    method = 'score'

    res = mqr.inference.rate.confint_1sample(count, n, meas, conf, bounded, method=method)
    assert res.name == 'rate of events'
    assert res.method == 'score'
    assert res.value == count / n / meas
    assert res.conf == conf
    assert res.bounded == bounded
    assert isinstance(res.lower, numbers.Number)
    assert isinstance(res.upper, numbers.Number)

    count = 10
    n = 10
    meas = 2.0
    conf = 0.95

    bounded = 'both'
    method = 'chi2'
    res = mqr.inference.rate.confint_1sample(count, n, meas, conf, bounded, method)
    assert list(res) == pytest.approx([0.2398, 0.9195], abs=1e-4)
    method = 'exact'
    res = mqr.inference.rate.confint_1sample(count, n, meas, conf, bounded, method)
    assert list(res) == pytest.approx([0.2746, 0.9195], abs=1e-4)
    method = 'wald-cc'
    res = mqr.inference.rate.confint_1sample(count, n, meas, conf, bounded, method)
    assert list(res) == pytest.approx([0.1729, 0.8426], abs=1e-4)

    bounded = 'below'
    method = 'chi2'
    res = mqr.inference.rate.confint_1sample(count, n, meas, conf, bounded, method)
    assert list(res) == pytest.approx([0.2713, np.inf], abs=1e-4)
    method = 'exact'
    res = mqr.inference.rate.confint_1sample(count, n, meas, conf, bounded, method)
    assert list(res) == pytest.approx([0.3085, np.inf], abs=1e-4)
    method = 'wald-cc'
    res = mqr.inference.rate.confint_1sample(count, n, meas, conf, bounded, method)
    assert list(res) == pytest.approx([0.2215, np.inf], abs=1e-4)

    bounded = 'above'
    method = 'chi2'
    res = mqr.inference.rate.confint_1sample(count, n, meas, conf, bounded, method)
    assert list(res) == pytest.approx([0.0, 0.8481], abs=1e-4)
    method = 'exact'
    res = mqr.inference.rate.confint_1sample(count, n, meas, conf, bounded, method)
    assert list(res) == pytest.approx([0.0, 0.8481], abs=1e-4)
    method = 'wald-cc'
    res = mqr.inference.rate.confint_1sample(count, n, meas, conf, bounded, method)
    assert list(res) == pytest.approx([0.0, 0.7915], abs=1e-4)

def test_confint_2sample():
    count1 = 5
    n1 = 20
    meas1 = 2.0
    count2 = 8
    n2 = 18
    meas2 = 3
    conf = 0.90
    bounded = 'both'
    method = 'score'

    res = mqr.inference.rate.confint_2sample(count1, n1, count2, n2, meas1, meas2, conf, method='score')
    assert res.name == 'difference between rates of events'
    assert res.method == 'score'
    assert res.value == count1 / n1 / meas1 - count2 / n2 / meas2
    assert isinstance(res.lower, numbers.Number)
    assert isinstance(res.upper, numbers.Number)
    assert res.conf == conf
    assert res.bounded == bounded

    count1, n1, meas1 = 1234, 20, 2
    count2, n2, meas2 = 2345, 20, 1
    conf = 0.95

    bounded = 'both'
    method = 'wald'
    res = mqr.inference.rate.confint_2sample(count1, n1, count2, n2, meas1, meas2, conf, bounded=bounded, method=method)
    assert list(res) == pytest.approx([-91.4481, -81.3519], abs=1e-4)
    method = 'wald-moment'
    res = mqr.inference.rate.confint_2sample(count1, n1, count2, n2, meas1, meas2, conf, bounded=bounded, method=method)
    assert list(res) == pytest.approx([-91.4963, -81.3997], abs=1e-4)

    bounded = 'below'
    method = 'wald'
    res = mqr.inference.rate.confint_2sample(count1, n1, count2, n2, meas1, meas2, conf, bounded=bounded, method=method)
    assert list(res) == pytest.approx([-90.6365, np.inf], abs=1e-4)
    method = 'wald-moment'
    res = mqr.inference.rate.confint_2sample(count1, n1, count2, n2, meas1, meas2, conf, bounded=bounded, method=method)
    assert list(res) == pytest.approx([-90.6705, np.inf], abs=1e-4)

    bounded = 'above'
    method = 'wald'
    res = mqr.inference.rate.confint_2sample(count1, n1, count2, n2, meas1, meas2, conf, bounded=bounded, method=method)
    assert list(res) == pytest.approx([-np.inf, -82.1635], abs=1e-4)
    method = 'wald-moment'
    res = mqr.inference.rate.confint_2sample(count1, n1, count2, n2, meas1, meas2, conf, bounded=bounded, method=method)
    assert list(res) == pytest.approx([-np.inf, -82.1972], abs=1e-4)

def test_confint_2sample_krishnamoorthy_lee():
    """
    Checks calculations against the values in Table 7 in [1]_.

    References
    ----------
    .. [1] Krishnamoorthy, K., & Lee, M. (2013).
       New approximate confidence intervals for the difference between
       two Poisson means and comparison.
       Journal of Statistical Computation and Simulation, 83(12), 2232-2243.
    """
    lower, upper = mqr.inference.lib.rate.confint_2sample_wald(
        3, 310, 7, 3500,
        1.0, 1.0,
        0.95, 'both')
    assert lower == pytest.approx(-0.0034, abs=1e-4)
    assert upper == pytest.approx(0.0187, abs=1e-4)

    lower, upper = mqr.inference.lib.rate.confint_2sample_wald(
        3, 310, 7, 3500,
        1.0, 1.0,
        0.95, 'below')
    assert lower == pytest.approx(-0.00160, abs=1e-5)
    assert upper == np.inf

    lower, upper = mqr.inference.lib.rate.confint_2sample_wald(
        3, 310, 7, 3500,
        1.0, 1.0,
        0.95, 'above')
    assert lower == -np.inf
    assert upper == pytest.approx(0.01695, abs=1e-5)

    lower, upper = mqr.inference.lib.rate.confint_2sample_wald_moment(
        3, 310, 7, 3500,
        1.0, 1.0,
        0.95, 'both')
    assert lower == pytest.approx(0.0009, abs=1e-4)
    assert upper == pytest.approx(0.02573, abs=1e-5)

    lower, upper = mqr.inference.lib.rate.confint_2sample_wald_moment(
        3, 310, 7, 3500,
        1.0, 1.0,
        0.95, 'below')
    assert lower == pytest.approx(0.00156, abs=1e-5)
    assert upper == np.inf

    lower, upper = mqr.inference.lib.rate.confint_2sample_wald_moment(
        3, 310, 7, 3500,
        1.0, 1.0,
        0.95, 'above')
    assert lower == -np.inf
    assert upper == pytest.approx(0.02175, abs=1e-5)

def test_test_1sample():
    count = 20
    n = 30
    meas = 3
    H0_rate = 0.3
    alternative = 'two-sided'
    method = 'exact-c'

    res = mqr.inference.rate.test_1sample(count, n, meas, H0_rate, alternative)
    assert res.description == 'rate of events'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_stat == 'rate'
    assert res.sample_stat_target == H0_rate
    assert res.sample_stat_value == pytest.approx(count / n / meas)
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)

def test_test_2sample():
    count1 = 3
    n1 = 10
    meas1 = 6
    count2 = 4
    n2 = 15
    meas2 = 5
    H0_value = -0.1
    alternative = 'less'
    method = 'wald'

    compare = 'diff'
    res = mqr.inference.rate.test_2sample(count1, n1, count2, n2, meas1, meas2, H0_value, alternative, method, compare)
    assert res.description == 'difference between rates of events'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_stat == 'rate1 - rate2'
    assert res.sample_stat_target == H0_value
    assert res.sample_stat_value == pytest.approx(count1 / n1 / meas1 - count2 / n2 / meas2)
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)

    compare = 'ratio'
    res = mqr.inference.rate.test_2sample(count1, n1, count2, n2, meas1, meas2, H0_value, alternative, method, compare)
    assert res.description == 'ratio of rates of events'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_stat == 'rate1 / rate2'
    assert res.sample_stat_target == H0_value
    assert res.sample_stat_value == pytest.approx(count1 / n1 / meas1 / (count2 / n2 / meas2))
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)

    compare = 'diff'
    res = mqr.inference.rate.test_2sample(count1, n1, count2, n2, meas1, meas2, None, alternative, method, compare)
    assert res.description == 'difference between rates of events'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_stat == 'rate1 - rate2'
    assert res.sample_stat_target == 0.0
    assert res.sample_stat_value == pytest.approx(count1 / n1 / meas1 - (count2 / n2 / meas2))
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)

    compare = 'ratio'
    res = mqr.inference.rate.test_2sample(count1, n1, count2, n2, meas1, meas2, None, alternative, method, compare)
    assert res.description == 'ratio of rates of events'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_stat == 'rate1 / rate2'
    assert res.sample_stat_target == 1.0
    assert res.sample_stat_value == pytest.approx(count1 / n1 / meas1 / (count2 / n2 / meas2))
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)
