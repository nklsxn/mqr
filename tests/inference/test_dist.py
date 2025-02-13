'''
Check call-throughs.
'''

import numbers
import numpy as np
import pytest
import scipy

import mqr

def test_test_1_sample():
    np.random.seed(1234)
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

    test = 'lf-norm'
    res = mqr.inference.dist.test_1sample(x, test)
    assert res.description == 'non-normality'
    assert res.alternative == 'two-sided'
    assert res.method == 'lilliefors'
    assert res.sample_stat == 'distribution'
    assert res.sample_stat_target == 'normal'
    assert res.sample_stat_value == None
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)

    test = 'sw-norm'
    res = mqr.inference.dist.test_1sample(x, test)
    assert res.description == 'non-normality'
    assert res.alternative == 'two-sided'
    assert res.method == 'shapiro-wilk'
    assert res.sample_stat == 'distribution'
    assert res.sample_stat_target == 'normal'
    assert res.sample_stat_value == None
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)

    test = 'dp-norm'
    x_large = scipy.stats.norm().rvs(30) # Uses kurtosistest, which warns for n<20
    res = mqr.inference.dist.test_1sample(x_large, test)
    assert res.description == 'non-normality'
    assert res.alternative == 'two-sided'
    assert res.method == 'dagostino-pearson'
    assert res.sample_stat == 'distribution'
    assert res.sample_stat_target == 'normal'
    assert res.sample_stat_value == None
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)

    test = 'jb-norm'
    with pytest.warns(UserWarning):
        res = mqr.inference.dist.test_1sample(x, test)
        assert res.pvalue > 0.05 # Only a small-sample warning
        assert res.description == 'non-normality'
        assert res.alternative == 'two-sided'
        assert res.method == 'jarque-bera'
        assert res.sample_stat == 'distribution'
        assert res.sample_stat_target == 'normal'
        assert res.sample_stat_value == None
        assert isinstance(res.stat, numbers.Number)
        assert isinstance(res.pvalue, numbers.Number)

    x_non_norm = scipy.stats.gamma(2).rvs(500)
    with pytest.warns(UserWarning):
        res = mqr.inference.dist.test_1sample(x_non_norm, test)
        assert res.pvalue < 0.05 # Only a small-pvalue warning, since n=500
        assert res.description == 'non-normality'
        assert res.alternative == 'two-sided'
        assert res.method == 'jarque-bera'
        assert res.sample_stat == 'distribution'
        assert res.sample_stat_target == 'normal'
        assert res.sample_stat_value == None
        assert isinstance(res.stat, numbers.Number)
        assert isinstance(res.pvalue, numbers.Number)

    x_large = scipy.stats.norm().rvs(500)
    res = mqr.inference.dist.test_1sample(x_large, test)
    assert res.description == 'non-normality'
    assert res.alternative == 'two-sided'
    assert res.method == 'jarque-bera'
    assert res.sample_stat == 'distribution'
    assert res.sample_stat_target == 'normal'
    assert res.sample_stat_value == None
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)

    test = 'cvm-norm'
    res = mqr.inference.dist.test_1sample(x, test)
    assert res.description == 'non-normality'
    assert res.alternative == 'two-sided'
    assert res.method == 'cramer-vonmises'
    assert res.sample_stat == 'distribution'
    assert res.sample_stat_target == 'normal'
    assert res.sample_stat_value == None
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)

    with pytest.raises(Exception) as e_info:
        mqr.inference.dist.test_1sample(x, 'not-a-test')
    assert e_info.value.args[0] == 'Method "not-a-test" is not available. Use ad-norm, ks-norm, lf-norm, sw-norm, dp-norm, jb-norm or cvm-norm.'
