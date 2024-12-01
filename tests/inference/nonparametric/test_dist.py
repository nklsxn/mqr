'''
Check call-throughs.
'''

import numbers
import numpy as np
import pytest
import scipy

import mqr

def test_test_1sample():
    x = np.array([1, 6, 2, 3, 4, 9, 7, 3, 3 ,4])
    alternative = 'two-sided'
    
    method = 'runs'
    res = mqr.inference.nonparametric.dist.test_1sample(x, method)
    assert res.description == 'randomness'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_stat == 'dist(x)'
    assert res.sample_stat_target == 'iid'
    assert res.sample_stat_value == None
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)

def test_test_2sample():
    x = np.array([1, 2, 7, 3, 4, 5, 5, 4, 7, 9], dtype=float)
    y = np.array([9, 3, 6, 4, 1, 7, 7, 4, 5, 3], dtype=float)

    alternative = 'less'
    method = 'ks'
    res = mqr.inference.nonparametric.dist.test_2sample(x, y, alternative, method)
    assert res.description == 'sampling distribution'
    assert res.alternative == alternative
    assert res.method == 'kolmogorov-smirnov'
    assert res.sample_stat == 'dist(x)'
    assert res.sample_stat_target == 'dist(y)'
    assert res.sample_stat_value == None
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)

    alternative = 'two-sided'
    method = 'runs'
    res = mqr.inference.nonparametric.dist.test_2sample(x, y, alternative, method)
    assert res.description == 'sampling distribution'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_stat == 'dist(x)'
    assert res.sample_stat_target == 'dist(y)'
    assert res.sample_stat_value == None
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)
