'''
Check call-throughs.
'''

import numbers
import numpy as np
import pytest
import scipy

import mqr

def test_test_nsample():
    x, y, z = np.array([
       [0.36522057, 0.37377119, 0.86150726, 0.96718967, 0.47966424,
        0.74979588, 0.35591634, 0.75272332, 0.17909715, 0.66216129],
       [0.30579918, 0.01280289, 0.2218879 , 0.77332064, 0.18087063,
        0.85439362, 0.41027609, 0.96153719, 0.94262441, 0.88509784],
       [0.1558813 , 0.25968253, 0.32619182, 0.89433078, 0.51197809,
        0.5814941 , 0.60785036, 0.32729355, 0.68637953, 0.9616885 ]], dtype=float)
    alternative = 'two-sided'

    method = 'levene'
    res = mqr.inference.nonparametric.variance.test_nsample(x, y, z, alternative=alternative, method=method)
    assert res.description == 'equality of variances'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_stat == 'var(x_i)'
    assert res.sample_stat_target == 'var(x_j)'
    assert isinstance(res.sample_stat_value, numbers.Number)
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)

    method = 'fligner-killeen'
    res = mqr.inference.nonparametric.variance.test_nsample(x, y, z, alternative=alternative, method=method)
    assert res.description == 'equality of variances'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_stat == 'var(x_i)'
    assert res.sample_stat_target == 'var(x_j)'
    assert isinstance(res.sample_stat_value, numbers.Number)
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)
