'''
Check call-throughs.
'''

import numbers
import numpy as np
import pytest
import scipy

import mqr

def test_test():
    x = np.array([1, 2, 3, 5, 5])
    y = np.array([1, 2, 2, 3, 5])
    alternative = 'two-sided'

    method = 'spearman'
    res = mqr.inference.nonparametric.correlation.test(x, y, alternative, method)
    assert res.description == 'correlation coefficient'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_stat == 'corr(x, y)'
    assert res.sample_stat_target == 0.0
    assert isinstance(res.sample_stat_value, numbers.Number)
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)

    method = 'kendall'
    res = mqr.inference.nonparametric.correlation.test(x, y, alternative, method)
    assert res.description == 'correlation coefficient'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_stat == 'corr(x, y)'
    assert res.sample_stat_target == 0.0
    assert isinstance(res.sample_stat_value, numbers.Number)
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)
