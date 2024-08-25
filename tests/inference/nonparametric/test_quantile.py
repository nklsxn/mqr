'''
Check call-throughs.
'''

import numbers
import numpy as np
import pytest
import scipy

import mqr

def test_confint_1sample():
    x = np.array([0.36522057, 0.37377119, 0.86150726, 0.96718967, 0.47966424,
        0.74979588, 0.35591634, 0.75272332, 0.17909715, 0.66216129])
    q = 0.2
    conf = 0.90
    bounded = 'below'

    res = mqr.inference.nonparametric.quantile.confint_1sample(x, q, conf, bounded)
    assert res.name == 'quantile (20th percentile)'
    assert res.method == 'binom'
    assert isinstance(res.value, numbers.Number)
    assert isinstance(res.lower, numbers.Number)
    assert res.upper == np.inf
    assert res.conf == conf
    assert res.bounded == bounded

def test_test_1sample():
    x = np.array([0.36522057, 0.37377119, 0.86150726, 0.96718967, 0.47966424,
        0.74979588, 0.35591634, 0.75272332, 0.17909715, 0.66216129])
    H0_quant = np.quantile(x, 0.2) + 0.1
    q = 0.2
    alternative = 'greater'

    res = mqr.inference.nonparametric.quantile.test_1sample(x, H0_quant, q, alternative)
    assert res.description == 'quantile'
    assert res.alternative == alternative
    assert res.method == 'binom'
    assert res.sample_stat == 'quantile(x, 0.2)'
    assert res.sample_stat_target == H0_quant
    assert res.sample_stat_value == np.quantile(x, q)
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)

    H0_quant = None
    res = mqr.inference.nonparametric.quantile.test_1sample(x, H0_quant, q, alternative)
    assert res.description == 'quantile'
    assert res.alternative == alternative
    assert res.method == 'binom'
    assert res.sample_stat == 'quantile(x, 0.2)'
    assert res.sample_stat_target == np.quantile(x, q)
    assert res.sample_stat_value == np.quantile(x, q)
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)
