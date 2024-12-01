'''
Check call-throughs.
'''

import numbers
import numpy as np
import pytest
import scipy

import mqr

def test_test_1_sample():
    x = np.array([1, 2, 3, 3, 4, 4, 4, 5, 5, 7, 9])

    test = 'ad-norm'
    res = mqr.inference.dist.test_1sample(x, test)
    assert res.description == 'non-normality'
    assert res.alternative == 'two-sided'
    assert res.method == 'anderson-darling'
    assert res.sample_stat == 'distribution'
    assert res.sample_stat_target == 'normal'
    assert res.sample_stat_value == None
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)

    test = 'ks-norm'
    res = mqr.inference.dist.test_1sample(x, test)
    assert res.description == 'non-normality'
    assert res.alternative == 'two-sided'
    assert res.method == 'kolmogorov-smirnov'
    assert res.sample_stat == 'distribution'
    assert res.sample_stat_target == 'normal'
    assert res.sample_stat_value == None
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)
