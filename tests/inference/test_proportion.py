'''
Check call-throughs.
'''

import numbers
import numpy as np
import pytest

import mqr

def test_size_1sample():
    p0 = 0.5
    pa = 0.4
    alpha = 0.05
    beta = 0.20
    alternative = 'two-sided'

    res = mqr.inference.proportion.size_1sample(p0, pa, alpha, beta, alternative)
    assert res.name == 'proportion'
    assert res.alpha == 0.05
    assert res.beta == 0.20
    assert res.effect == '0.5 - 0.4 = 0.1'
    assert res.alternative == 'two-sided'
    assert res.method == 'z'
    assert isinstance(res.sample_size, numbers.Number)

def test_size_2sample():
    p1 = 0.5
    p2 = 0.4
    alpha = 0.05
    beta = 0.20
    alternative = 'two-sided'

    res = mqr.inference.proportion.size_2sample(p1, p2, alpha, beta, alternative)
    assert res.name == 'difference between proportions'
    assert res.alpha == 0.05
    assert res.beta == 0.20
    assert res.effect == '0.5 - 0.4 = 0.1'
    assert res.alternative == 'two-sided'
    assert res.method == 'z'
    assert isinstance(res.sample_size, numbers.Number)

def test_confint_1sample():
    count = 5
    nobs = 10
    conf = 0.90

    res = mqr.inference.proportion.confint_1sample(count, nobs, conf)
    assert res.name == 'proportion'
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
