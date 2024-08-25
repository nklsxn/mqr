import numbers
import numpy as np
import pytest

import mqr

def test_size_1sample():
    alpha = 0.10
    beta = 0.10

    effect = 1.5
    alternative = 'two-sided'
    res = mqr.inference.variance.size_1sample(effect, alpha, beta, alternative)
    assert res.name == 'variance'
    assert res.alpha == alpha
    assert res.beta == beta
    assert res.effect == effect
    assert res.alternative == alternative
    assert res.method == 'chi2'
    assert np.ceil(res.sample_size) == 105

    alternative = 'greater'
    res = mqr.inference.variance.size_1sample(effect, alpha, beta, alternative)
    assert res.name == 'variance'
    assert res.alpha == alpha
    assert res.beta == beta
    assert res.effect == effect
    assert res.alternative == alternative
    assert res.method == 'chi2'
    assert np.ceil(res.sample_size) == 82

    effect = 0.8
    alternative = 'less'
    res = mqr.inference.variance.size_1sample(effect, alpha, beta, alternative)
    assert res.name == 'variance'
    assert res.alpha == alpha
    assert res.beta == beta
    assert res.effect == effect
    assert res.alternative == alternative
    assert res.method == 'chi2'
    assert np.ceil(res.sample_size) == 266

def test_size_2sample():
    alpha = 0.10
    beta = 0.10

    var_ratio = 1.5
    alternative = 'two-sided'
    res = mqr.inference.variance.size_2sample(var_ratio, alpha, beta, alternative)
    assert res.name == 'ratio of variances'
    assert res.alpha == alpha
    assert res.beta == beta
    assert res.effect == var_ratio
    assert res.alternative == alternative
    assert res.method == 'f'
    assert np.ceil(res.sample_size) == 211

    alternative = 'greater'
    res = mqr.inference.variance.size_2sample(var_ratio, alpha, beta, alternative)
    assert res.name == 'ratio of variances'
    assert res.alpha == alpha
    assert res.beta == beta
    assert res.effect == var_ratio
    assert res.alternative == alternative
    assert res.method == 'f'
    assert np.ceil(res.sample_size) == 162

    var_ratio = 0.8
    alternative = 'less'
    res = mqr.inference.variance.size_2sample(var_ratio, alpha, beta, alternative)
    assert res.name == 'ratio of variances'
    assert res.alpha == alpha
    assert res.beta == beta
    assert res.effect == var_ratio
    assert res.alternative == alternative
    assert res.method == 'f'
    assert np.ceil(res.sample_size) == 530

def test_confint_1sample():
    x = np.array([ 98.89,  98.75,  97.61, 100.38,  99.56,
                  103.12,  97.73, 102.39, 96.74, 100.06])
    s2 = np.var(x, ddof=1)
    conf = 0.95

    bounded = 'both'
    res = mqr.inference.variance.confint_1sample(x, conf, bounded=bounded)
    assert res.name == 'variance'
    assert res.method == 'chi2'
    assert res.value == s2
    assert res.lower == pytest.approx(1.9871, 1e-4)
    assert res.upper == pytest.approx(13.9980, 1e-4)
    assert res.conf == conf
    assert res.bounded == bounded

    bounded = 'below'
    res = mqr.inference.variance.confint_1sample(x, conf, bounded=bounded)
    assert res.name == 'variance'
    assert res.method == 'chi2'
    assert res.value == s2
    assert res.lower == pytest.approx(2.2342, 1e-4)
    assert res.upper == np.inf
    assert res.conf == conf
    assert res.bounded == bounded

    bounded = 'above'
    res = mqr.inference.variance.confint_1sample(x, conf, bounded=bounded)
    assert res.name == 'variance'
    assert res.method == 'chi2'
    assert res.value == s2
    assert res.lower == 0.0
    assert res.upper == pytest.approx(11.3680, 1e-4)
    assert res.conf == conf
    assert res.bounded == bounded

def test_confint_2sample():
    x = np.array([0, 1, 2])
    y = np.array([0, 2, 4])
    ratio = np.var(x, ddof=1) / np.var(y, ddof=1)
    conf = 0.90

    bounded = 'both'
    res = mqr.inference.variance.confint_2sample(x, y, conf, bounded=bounded)
    assert res.name == 'ratio of variances'
    assert res.method == 'f'
    assert res.value == ratio
    assert res.lower == pytest.approx(0.0132, abs=1e-4)
    assert res.upper == pytest.approx(4.75)
    assert res.conf == conf
    assert res.bounded == bounded

    bounded = 'below'
    res = mqr.inference.variance.confint_2sample(x, y, conf, bounded=bounded)
    assert res.name == 'ratio of variances'
    assert res.method == 'f'
    assert res.value == ratio
    assert res.lower == pytest.approx(0.0278, abs=1e-4)
    assert res.upper == np.inf
    assert res.conf == conf
    assert res.bounded == bounded

    bounded = 'above'
    res = mqr.inference.variance.confint_2sample(x, y, conf, bounded=bounded)
    assert res.name == 'ratio of variances'
    assert res.method == 'f'
    assert res.value == ratio
    assert res.lower == -np.inf
    assert res.upper == pytest.approx(2.25)
    assert res.conf == conf
    assert res.bounded == bounded

def test_test_1sample():
    x = np.array([0, 1, 2])
    H0_var = 1

    alternative = 'two-sided'
    res = mqr.inference.variance.test_1sample(x, H0_var, alternative)
    assert res.description == 'variance'
    assert res.alternative == alternative
    assert res.method == 'chi2'
    assert res.sample_stat == 'var(x)'
    assert res.sample_stat_target == H0_var
    assert res.sample_stat_value == np.var(x, ddof=1)
    assert isinstance(res.stat, numbers.Number)
    assert res.pvalue == pytest.approx(0.7358, abs=1e-4)

    alternative = 'greater'
    res = mqr.inference.variance.test_1sample(x, H0_var, alternative)
    assert res.description == 'variance'
    assert res.alternative == alternative
    assert res.method == 'chi2'
    assert res.sample_stat == 'var(x)'
    assert res.sample_stat_target == H0_var
    assert res.sample_stat_value == np.var(x, ddof=1)
    assert isinstance(res.stat, numbers.Number)
    assert res.pvalue == pytest.approx(0.3679, abs=1e-4)

    alternative = 'less'
    res = mqr.inference.variance.test_1sample(x, H0_var, alternative)
    assert res.description == 'variance'
    assert res.alternative == alternative
    assert res.method == 'chi2'
    assert res.sample_stat == 'var(x)'
    assert res.sample_stat_target == H0_var
    assert res.sample_stat_value == np.var(x, ddof=1)
    assert isinstance(res.stat, numbers.Number)
    assert res.pvalue == pytest.approx(0.6321, abs=1e-4)

def test_test_2sample():
    x = np.array([0, 1, 2])
    y = np.array([1, 2, 4])

    method = 'f'
    alternative = 'two-sided'
    res = mqr.inference.variance.test_2sample(x, y, alternative, method)
    assert res.description == 'ratio of variances'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_stat == 'var(x) / var(y)'
    assert res.sample_stat_target == 1.0
    assert res.sample_stat_value == np.var(x, ddof=1) / np.var(y, ddof=1)
    assert isinstance(res.stat, numbers.Number)
    assert res.pvalue == pytest.approx(0.6)

    alternative = 'less'
    res = mqr.inference.variance.test_2sample(x, y, alternative, method)
    assert res.description == 'ratio of variances'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_stat == 'var(x) / var(y)'
    assert res.sample_stat_target == 1.0
    assert res.sample_stat_value == np.var(x, ddof=1) / np.var(y, ddof=1)
    assert isinstance(res.stat, numbers.Number)
    assert res.pvalue == pytest.approx(0.3)

    alternative = 'greater'
    res = mqr.inference.variance.test_2sample(x, y, alternative, method)
    assert res.description == 'ratio of variances'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_stat == 'var(x) / var(y)'
    assert res.sample_stat_target == 1.0
    assert res.sample_stat_value == np.var(x, ddof=1) / np.var(y, ddof=1)
    assert isinstance(res.stat, numbers.Number)
    assert res.pvalue == pytest.approx(0.7)

    method = 'levene'
    alternative = 'two-sided'
    res = mqr.inference.variance.test_2sample(x, y, alternative, method)
    assert res.description == 'equality of variances'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_stat == 'var(x) - var(y)'
    assert res.sample_stat_target == 0.0
    assert isinstance(res.sample_stat_value, numbers.Number)
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)

    method = 'bartlett'
    res = mqr.inference.variance.test_2sample(x, y, alternative, method)
    assert res.description == 'equality of variances'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_stat == 'var(x) - var(y)'
    assert res.sample_stat_target == 0.0
    assert isinstance(res.sample_stat_value, numbers.Number)
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)
