'''
Check call-throughs.
'''

import numbers
import numpy as np
import pytest

import mqr

def test_size_1sample():
    ra = 0.8
    hyp_rate = 0.5
    alpha = 0.05
    beta = 0.20

    alternative = 'two-sided'
    res = mqr.inference.rate.size_1sample(ra, hyp_rate, alpha, beta, alternative)
    assert res.name == 'rate of events'
    assert res.alpha == alpha
    assert res.beta == beta
    assert res.effect == '0.8 / 0.5 = 1.6'
    assert res.alternative == alternative
    assert res.method == 'chi2'
    assert isinstance(res.sample_size, numbers.Number)

    alternative = 'less'
    res = mqr.inference.rate.size_1sample(ra, hyp_rate, alpha, beta, alternative)
    assert res.name == 'rate of events'
    assert res.alpha == alpha
    assert res.beta == beta
    assert res.effect == '0.8 / 0.5 = 1.6'
    assert res.alternative == alternative
    assert res.method == 'chi2'
    assert isinstance(res.sample_size, numbers.Number)

    alternative = 'greater'
    res = mqr.inference.rate.size_1sample(ra, hyp_rate, alpha, beta, alternative)
    assert res.name == 'rate of events'
    assert res.alpha == alpha
    assert res.beta == beta
    assert res.effect == '0.8 / 0.5 = 1.6'
    assert res.alternative == alternative
    assert res.method == 'chi2'
    assert isinstance(res.sample_size, numbers.Number)

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
    assert res.effect == '0.8 - 0.5 = 0.2'
    assert res.alternative == alternative
    assert res.method == None
    assert isinstance(res.sample_size, numbers.Number)

def test_confint_1sample():
    count = 5
    n = 20
    meas = 2.0
    conf = 0.90

    res = mqr.inference.rate.confint_1sample(count, n, meas, conf)
    assert res.name == 'rate of events'
    assert res.value == count / n / meas
    assert res.conf == conf
    assert isinstance(res.lower, numbers.Number)
    assert isinstance(res.upper, numbers.Number)

def test_confint_2sample():
    count1 = 5
    n1 = 20
    meas1 = 2.0
    count2 = 8
    n2 = 18
    meas2 = 3
    conf = 0.90

    res = mqr.inference.rate.confint_2sample(count1, n1, count2, n2, meas1, meas2, conf)
    assert res.name == 'difference between rates of events'
    assert res.value == count1 / n1 / meas1 - count2 / n2 / meas2
    assert isinstance(res.lower, numbers.Number)
    assert isinstance(res.upper, numbers.Number)
    assert res.conf == conf

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
