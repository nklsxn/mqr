'''
Check call-throughs.
'''

import numbers
import numpy as np
import pytest

import mqr

def test_size_1sample():
    effect = 1.0
    alpha = 0.01
    beta = 0.01
    alternative = 'two-sided'

    res = mqr.inference.mean.size_1sample(effect, alpha, beta, alternative)
    assert res.name == 'mean'
    assert res.alpha == alpha
    assert res.beta == beta
    assert res.effect == effect
    assert res.method == 't'
    assert res.alternative == alternative
    assert isinstance(res.sample_size, numbers.Number)

def test_size_2sample():
    effect = 1.0
    alpha = 0.01
    beta = 0.01
    alternative = 'two-sided'

    res = mqr.inference.mean.size_2sample(effect, alpha, beta, alternative)
    assert res.name == 'difference between means (independent)'
    assert res.alpha == alpha
    assert res.beta == beta
    assert res.effect == effect
    assert res.method == 't'
    assert res.alternative == alternative
    assert isinstance(res.sample_size, numbers.Number)

def test_size_paired():
    effect = 1.0
    alpha = 0.01
    beta = 0.01
    alternative = 'two-sided'

    res = mqr.inference.mean.size_paired(effect, alpha, beta, alternative)
    assert res.name == 'difference between means (paired)'
    assert res.alpha == alpha
    assert res.beta == beta
    assert res.effect == effect
    assert res.method == 't'
    assert res.alternative == alternative
    assert isinstance(res.sample_size, numbers.Number)

def test_confint_1sample():
    x = np.array([0.9, 1.1])
    conf = 0.90
    bounded = 'both'
    alternative = 'both'

    res = mqr.inference.mean.confint_1sample(x, conf, alternative)
    assert res.name == 'mean'
    assert res.conf == conf
    assert res.value == np.mean(x)
    assert isinstance(res.lower, numbers.Number)
    assert isinstance(res.upper, numbers.Number)

def test_confint_2sample():
    x = np.array([0.9, 1.1])
    y = np.array([1.9, 2.1])
    conf = 0.90
    pooled = True
    alternative = 'both'

    res = mqr.inference.mean.confint_2sample(x, y, conf, pooled, alternative)
    assert res.name == 'difference between means (independent)'
    assert conf == conf
    assert res.value == np.mean(x) - np.mean(y)
    assert isinstance(res.lower, numbers.Number)
    assert isinstance(res.upper, numbers.Number)

def test_confint_paired():
    x = np.array([0.9, 1.1])
    y = np.array([1.9, 2.1])
    conf = 0.90
    alternative = 'both'

    res = mqr.inference.mean.confint_paired(x, y, conf, alternative)
    assert res.name == 'difference between means (paired)'
    assert conf == conf
    assert res.value == np.mean(x) - np.mean(y)
    assert isinstance(res.lower, numbers.Number)
    assert isinstance(res.upper, numbers.Number)

def test_test_1sample():
    x = np.array([0.9, 1.1])
    H0_mean = 1.1
    alternative = 'two-sided'

    res = mqr.inference.mean.test_1sample(x, H0_mean, alternative, 'z')
    assert res.description == 'mean'
    assert res.alternative == alternative
    assert res.method == 'z'
    assert res.sample_stat == 'mean(x)'
    assert res.sample_stat_target == H0_mean
    assert res.sample_stat_value == np.mean(x)
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)

    res = mqr.inference.mean.test_1sample(x, H0_mean, 'two-sided', 't')
    assert res.description == 'mean'
    assert res.alternative == alternative
    assert res.method == 't'
    assert res.sample_stat == 'mean(x)'
    assert res.sample_stat_target == H0_mean
    assert res.sample_stat_value == np.mean(x)
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)

def test_test_2sample():
    x = np.array([0.9, 1.1])
    y = np.array([1.9, 2.1])
    H0_diff = 1.1
    pooled = True
    alternative = 'two-sided'

    res = mqr.inference.mean.test_2sample(x, y, H0_diff, pooled, alternative, 't')
    assert res.description == 'difference between means (independent)'
    assert res.alternative == alternative
    assert res.method == 't'
    assert res.sample_stat == 'mean(x) - mean(y)'
    assert res.sample_stat_target == H0_diff
    assert res.sample_stat_value == np.mean(x) - np.mean(y)
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)

    res = mqr.inference.mean.test_2sample(x, y, H0_diff, pooled, alternative, 't')
    assert res.description == 'difference between means (independent)'
    assert res.alternative == alternative
    assert res.method == 't'
    assert res.sample_stat == 'mean(x) - mean(y)'
    assert res.sample_stat_target == H0_diff
    assert res.sample_stat_value == np.mean(x) - np.mean(y)
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)

def test_test_paired():
    x = np.array([0.9, 1.1])
    y = np.array([1.8, 2.2])
    alternative = 'two-sided'
    method = 't'

    res = mqr.inference.mean.test_paired(x, y, alternative, method)
    assert res.description == 'difference between means (paired)'
    assert res.alternative == alternative
    assert res.method == 't'
    assert res.sample_stat == 'mean(x) - mean(y)'
    assert res.sample_stat_target == 0.0
    assert res.sample_stat_value == np.mean(x) - np.mean(y)
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)
