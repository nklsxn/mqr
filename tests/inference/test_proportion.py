'''
Check call-throughs.
'''

import numbers
import numpy as np
import pytest

import mqr

def test_power_1sample():
    p0 = 0.4
    alpha = 0.05
    nobs = 200

    pa = 0.5
    alternative = 'two-sided'
    method = 'norm-approx'
    res = mqr.inference.proportion.power_1sample(pa, p0, nobs, alpha, alternative, method)
    assert res.name == 'proportion'
    assert res.alpha == 0.05
    assert res.beta == pytest.approx(0.181922, abs=1e-6)
    assert res.effect == '0.5 - 0.4 = 0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_size == nobs

    alternative = 'greater'
    res = mqr.inference.proportion.power_1sample(pa, p0, nobs, alpha, alternative, method)
    assert res.name == 'proportion'
    assert res.alpha == 0.05
    assert res.beta == pytest.approx(0.111839, abs=1e-6)
    assert res.effect == '0.5 - 0.4 = 0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_size == nobs

    pa = 0.3
    alternative = 'less'
    res = mqr.inference.proportion.power_1sample(pa, p0, nobs, alpha, alternative, method)
    assert res.name == 'proportion'
    assert res.alpha == 0.05
    assert res.beta == pytest.approx(0.0921478, abs=1e-6)
    assert res.effect == '0.3 - 0.4 = -0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_size == nobs

def test_power_2sample():
    p2 = 0.4
    alpha = 0.05
    nobs = 200

    p1 = 0.5
    alternative = 'two-sided'
    method = 'norm-approx'
    res = mqr.inference.proportion.power_2sample(p1, p2, nobs, alpha, alternative, method)
    assert res.name == 'difference between proportions'
    assert res.alpha == 0.05
    assert res.beta == pytest.approx(0.479882, abs=1e-6)
    assert res.effect == '0.5 - 0.4 = 0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_size == nobs

    alternative = 'greater'
    res = mqr.inference.proportion.power_2sample(p1, p2, nobs, alpha, alternative, method)
    assert res.name == 'difference between proportions'
    assert res.alpha == 0.05
    assert res.beta == pytest.approx(0.356779, abs=1e-6)
    assert res.effect == '0.5 - 0.4 = 0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_size == nobs

    p1 = 0.3
    alternative = 'less'
    res = mqr.inference.proportion.power_2sample(p1, p2, nobs, alpha, alternative, method)
    assert res.name == 'difference between proportions'
    assert res.alpha == 0.05
    assert res.beta == pytest.approx(0.324836, abs=1e-6)
    assert res.effect == '0.3 - 0.4 = -0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_size == nobs

def test_size_1sample():
    p0 = 0.4
    alpha = 0.05
    beta = 0.05

    pa = 0.5
    alternative = 'two-sided'
    method = 'norm-approx'
    res = mqr.inference.proportion.size_1sample(pa, p0, alpha, beta, alternative, method)
    assert res.name == 'proportion'
    assert res.alpha == 0.05
    assert res.beta == 0.05
    assert res.effect == '0.5 - 0.4 = 0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert np.ceil(res.sample_size) == 318

    alternative = 'greater'
    res = mqr.inference.proportion.size_1sample(pa, p0, alpha, beta, alternative, method)
    assert res.name == 'proportion'
    assert res.alpha == 0.05
    assert res.beta == 0.05
    assert res.effect == '0.5 - 0.4 = 0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert np.ceil(res.sample_size) == 266

    pa = 0.3
    alternative = 'less'
    res = mqr.inference.proportion.size_1sample(pa, p0, alpha, beta, alternative, method)
    assert res.name == 'proportion'
    assert res.alpha == 0.05
    assert res.beta == 0.05
    assert res.effect == '0.3 - 0.4 = -0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert np.ceil(res.sample_size) == 244

    pa = 0.5
    alternative = 'two-sided'
    method = 'arcsin'
    res = mqr.inference.proportion.size_1sample(pa, p0, alpha, beta, alternative, method)
    assert res.name == 'proportion'
    assert res.alpha == 0.05
    assert res.beta == 0.05
    assert res.effect == '0.5 - 0.4 = 0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert np.ceil(res.sample_size) == 321

    alternative = 'greater'
    res = mqr.inference.proportion.size_1sample(pa, p0, alpha, beta, alternative, method)
    assert res.name == 'proportion'
    assert res.alpha == 0.05
    assert res.beta == 0.05
    assert res.effect == '0.5 - 0.4 = 0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert np.ceil(res.sample_size) == 267

    pa = 0.3
    alternative = 'less'
    res = mqr.inference.proportion.size_1sample(pa, p0, alpha, beta, alternative, method)
    assert res.name == 'proportion'
    assert res.alpha == 0.05
    assert res.beta == 0.05
    assert res.effect == '0.3 - 0.4 = -0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert np.ceil(res.sample_size) == 246

def test_size_2sample():
    p2 = 0.4
    alpha = 0.05
    beta = 0.05

    p1 = 0.5
    alternative = 'two-sided'
    method = 'norm-approx'
    res = mqr.inference.proportion.size_2sample(p1, p2, alpha, beta, alternative, method)
    assert res.name == 'proportion'
    assert res.alpha == 0.05
    assert res.beta == 0.05
    assert res.effect == '0.5 - 0.4 = 0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert np.ceil(res.sample_size) == 641

    alternative = 'greater'
    res = mqr.inference.proportion.size_2sample(p1, p2, alpha, beta, alternative, method)
    assert res.name == 'proportion'
    assert res.alpha == 0.05
    assert res.beta == 0.05
    assert res.effect == '0.5 - 0.4 = 0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert np.ceil(res.sample_size) == 533

    p1 = 0.3
    alternative = 'less'
    res = mqr.inference.proportion.size_2sample(p1, p2, alpha, beta, alternative, method)
    assert res.name == 'proportion'
    assert res.alpha == 0.05
    assert res.beta == 0.05
    assert res.effect == '0.3 - 0.4 = -0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert np.ceil(res.sample_size) == 490

    p1 = 0.5
    alternative = 'two-sided'
    method = 'arcsin'
    res = mqr.inference.proportion.size_2sample(p1, p2, alpha, beta, alternative, method)
    assert res.name == 'proportion'
    assert res.alpha == 0.05
    assert res.beta == 0.05
    assert res.effect == '0.5 - 0.4 = 0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert np.ceil(res.sample_size) == 642

    alternative = 'greater'
    res = mqr.inference.proportion.size_2sample(p1, p2, alpha, beta, alternative, method)
    assert res.name == 'proportion'
    assert res.alpha == 0.05
    assert res.beta == 0.05
    assert res.effect == '0.5 - 0.4 = 0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert np.ceil(res.sample_size) == 534

    p1 = 0.3
    alternative = 'less'
    res = mqr.inference.proportion.size_2sample(p1, p2, alpha, beta, alternative, method)
    assert res.name == 'proportion'
    assert res.alpha == 0.05
    assert res.beta == 0.05
    assert res.effect == '0.3 - 0.4 = -0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert np.ceil(res.sample_size) == 491

def test_confint_1sample():
    count = 5
    nobs = 10
    conf = 0.90

    res = mqr.inference.proportion.confint_1sample(count, nobs, conf)
    assert res.name == 'proportion'
    assert res.method == 'beta'
    assert res.value == count / nobs
    assert isinstance(res.lower, numbers.Number)
    assert isinstance(res.upper, numbers.Number)
    assert res.conf == conf

def test_confint_2sample():
    count1 = 5
    nobs1 = 10
    count2 = 15
    nobs2 = 30
    conf = 0.90

    res = mqr.inference.proportion.confint_2sample(count1, nobs1, count2, nobs2, conf)
    assert res.name == 'difference between proportions'
    assert res.method == 'newcomb'
    assert res.value == count1 / nobs1 - count2 / nobs2
    assert isinstance(res.lower, numbers.Number)
    assert isinstance(res.upper, numbers.Number)
    assert res.conf == conf

def test_test_1sample():
    count = 5
    nobs = 10
    H0_prop = 0.6
    alternative = 'two-sided'

    res = mqr.inference.proportion.test_1sample(count, nobs, H0_prop, alternative, 'binom')
    assert res.description == 'proportion of "true" elements'
    assert res.alternative == alternative
    assert res.method == 'binom'
    assert res.sample_stat == 'count / nobs'
    assert res.sample_stat_target == H0_prop
    assert res.sample_stat_value == count / nobs
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)

    res = mqr.inference.proportion.test_1sample(count, nobs, H0_prop, alternative, 'z')
    assert res.description == 'proportion of "true" elements'
    assert res.alternative == alternative
    assert res.method == 'z'
    assert res.sample_stat == 'count / nobs'
    assert res.sample_stat_target == H0_prop
    assert res.sample_stat_value == count / nobs
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)

    res = mqr.inference.proportion.test_1sample(count, nobs, H0_prop, alternative, 'chi2')
    assert res.description == 'proportion of "true" elements'
    assert res.alternative == alternative
    assert res.method == 'chi2'
    assert res.sample_stat == 'count / nobs'
    assert res.sample_stat_target == H0_prop
    assert res.sample_stat_value == count / nobs
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)

def test_test_2sample():
    count1 = 5
    nobs1 = 10
    count2 = 15
    nobs2 = 60
    H0_diff = 0.5 - 0.25
    alternative = 'two-sided'

    res = mqr.inference.proportion.test_2sample(count1, nobs1, count2, nobs2, H0_diff, alternative)
    assert res.description == 'difference between proportions of "true" elements'
    assert res.alternative == alternative
    assert res.method == 'agresti-caffo'
    assert res.sample_stat == 'count1 / nobs1 - count2 / nobs2'
    assert res.sample_stat_target == H0_diff
    assert res.sample_stat_value == count1 / nobs1 - count2 / nobs2
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)
