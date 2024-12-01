'''
Check call-throughs.
'''

import numbers
import numpy as np
import pytest
import scipy

import mqr

def test_test_1sample():
    x = np.array([0.88612542, 0.67327485, 0.8307321 , 0.46658372, 0.62968216,
        0.67137016, 0.14714929, 0.02982314, 0.45770049, 0.70468434,
        0.97795913, 0.78578141, 0.14394705, 0.73653195, 0.40538661,])
    H0_median = 0.5

    alternative = 'two-sided'
    method = 'sign'
    res = mqr.inference.nonparametric.median.test_1sample(x, H0_median, alternative, method)
    assert res.description == 'median'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_stat == 'median(x)'
    assert res.sample_stat_target == H0_median
    assert isinstance(res.sample_stat_value, numbers.Number)
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)

    alternative = 'less'
    method = 'wilcoxon'
    res = mqr.inference.nonparametric.median.test_1sample(x, H0_median, alternative, method)
    assert res.description == 'median'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_stat == 'median(x)'
    assert res.sample_stat_target == H0_median
    assert isinstance(res.sample_stat_value, numbers.Number)
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)

def test_test_nsample():
    x, y, z = np.array([
       [0.36522057, 0.37377119, 0.86150726, 0.96718967, 0.47966424,
        0.74979588, 0.35591634, 0.75272332, 0.17909715, 0.66216129],
       [0.30579918, 0.01280289, 0.2218879 , 0.77332064, 0.18087063,
        0.85439362, 0.41027609, 0.96153719, 0.94262441, 0.88509784],
       [0.1558813 , 0.25968253, 0.32619182, 0.89433078, 0.51197809,
        0.5814941 , 0.60785036, 0.32729355, 0.68637953, 0.9616885 ]])

    alternative = 'two-sided'
    method = 'kruskal-wallis'
    res = mqr.inference.nonparametric.median.test_nsample(x, y, z, alternative=alternative, method=method)
    assert res.description == 'equality of medians'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_stat == 'median(x_i)'
    assert res.sample_stat_target == 'median(x_j)'
    assert np.isnan(res.sample_stat_value)
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)

    alternative = 'greater'
    method = 'mann-whitney'
    res = mqr.inference.nonparametric.median.test_nsample(x, y, alternative=alternative, method=method)
    assert res.description == 'equality of medians'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_stat == 'median(x) - median(y)'
    assert res.sample_stat_target == 0.0
    assert np.isnan(res.sample_stat_value)
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)
