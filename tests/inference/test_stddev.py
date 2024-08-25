import numbers
import numpy as np
import pytest

import mqr

def test_size_1sample():
    alpha = 0.10
    beta = 0.10

    effect = np.sqrt(1.5)
    alternative = 'two-sided'
    res = mqr.inference.stddev.size_1sample(effect, alpha, beta, alternative)
    assert res.name == 'standard deviation'
    assert res.alpha == alpha
    assert res.beta == beta
    assert res.effect == effect
    assert res.alternative == alternative
    assert res.method == 'chi2'
    assert np.ceil(res.sample_size) == 105

    alternative = 'greater'
    res = mqr.inference.stddev.size_1sample(effect, alpha, beta, alternative)
    assert res.name == 'standard deviation'
    assert res.alpha == alpha
    assert res.beta == beta
    assert res.effect == effect
    assert res.alternative == alternative
    assert res.method == 'chi2'
    assert np.ceil(res.sample_size) == 82

    effect = np.sqrt(0.8)
    alternative = 'less'
    res = mqr.inference.stddev.size_1sample(effect, alpha, beta, alternative)
    assert res.name == 'standard deviation'
    assert res.alpha == alpha
    assert res.beta == beta
    assert res.effect == effect
    assert res.alternative == alternative
    assert res.method == 'chi2'
    assert np.ceil(res.sample_size) == 266

def test_size_2sample():
    alpha = 0.10
    beta = 0.10

    std_ratio = np.sqrt(1.5)
    alternative = 'two-sided'
    res = mqr.inference.stddev.size_2sample(std_ratio, alpha, beta, alternative)
    assert res.name == 'ratio of standard deviations'
    assert res.alpha == alpha
    assert res.beta == beta
    assert res.effect == std_ratio
    assert res.alternative == alternative
    assert res.method == 'f'
    assert np.ceil(res.sample_size) == 211

    alternative = 'greater'
    res = mqr.inference.stddev.size_2sample(std_ratio, alpha, beta, alternative)
    assert res.name == 'ratio of standard deviations'
    assert res.alpha == alpha
    assert res.beta == beta
    assert res.effect == std_ratio
    assert res.alternative == alternative
    assert res.method == 'f'
    assert np.ceil(res.sample_size) == 162

    std_ratio = np.sqrt(0.8)
    alternative = 'less'
    res = mqr.inference.stddev.size_2sample(std_ratio, alpha, beta, alternative)
    assert res.name == 'ratio of standard deviations'
    assert res.alpha == alpha
    assert res.beta == beta
    assert res.effect == std_ratio
    assert res.alternative == alternative
    assert res.method == 'f'
    assert np.ceil(res.sample_size) == 530

def test_confint_1sample():
    x = np.array([0, 1, 2])
    conf = 0.90

    bounded = 'both'
    res = mqr.inference.stddev.confint_1sample(x, conf, bounded=bounded)
    assert res.name == 'standard deviation'
    assert res.method == 'chi2'
    assert res.value == np.std(x, ddof=1)
    assert res.lower == pytest.approx(np.sqrt(0.3338), 1e-4)
    assert res.upper == pytest.approx(np.sqrt(19.4957), 1e-4)
    assert res.conf == conf
    assert res.bounded == bounded

    bounded = 'below'
    res = mqr.inference.stddev.confint_1sample(x, conf, bounded=bounded)
    assert res.name == 'standard deviation'
    assert res.method == 'chi2'
    assert res.value == np.std(x, ddof=1)
    assert res.lower == pytest.approx(np.sqrt(0.4343), 1e-4)
    assert res.upper == np.inf
    assert res.conf == conf
    assert res.bounded == bounded

    bounded = 'above'
    res = mqr.inference.stddev.confint_1sample(x, conf, bounded=bounded)
    assert res.name == 'standard deviation'
    assert res.method == 'chi2'
    assert res.value == np.std(x, ddof=1)
    assert res.lower == 0.0
    assert res.upper == pytest.approx(np.sqrt(9.4912), 1e-4)
    assert res.conf == conf
    assert res.bounded == bounded

def test_confint_2sample():
    x = np.array([0, 1, 2])
    y = np.array([0, 2, 4])
    conf = 0.90

    bounded = 'both'
    res = mqr.inference.stddev.confint_2sample(x, y, conf, bounded=bounded)
    assert res.name == 'ratio of standard deviations'
    assert res.method == 'f'
    assert res.value == np.std(x, ddof=1) / np.std(y, ddof=1)
    assert res.lower == pytest.approx(np.sqrt(0.01316), abs=1e-4)
    assert res.upper == pytest.approx(np.sqrt(4.75))
    assert res.conf == conf
    assert res.bounded == bounded

    bounded = 'below'
    res = mqr.inference.stddev.confint_2sample(x, y, conf, bounded=bounded)
    assert res.name == 'ratio of standard deviations'
    assert res.method == 'f'
    assert res.value == np.std(x, ddof=1) / np.std(y, ddof=1)
    assert res.lower == pytest.approx(np.sqrt(0.02778), abs=1e-4)
    assert res.upper == np.inf
    assert res.conf == conf
    assert res.bounded == bounded

    bounded = 'above'
    res = mqr.inference.stddev.confint_2sample(x, y, conf, bounded=bounded)
    assert res.name == 'ratio of standard deviations'
    assert res.method == 'f'
    assert res.value == np.std(x, ddof=1) / np.std(y, ddof=1)
    assert res.lower == 0.0
    assert res.upper == pytest.approx(np.sqrt(2.25))
    assert res.conf == conf
    assert res.bounded == bounded

def test_test_1sample():
    x = np.array([0, 1, 2])
    H0_std = 1

    alternative = 'two-sided'
    res = mqr.inference.stddev.test_1sample(x, H0_std, alternative)
    assert res.description == 'standard deviation'
    assert res.alternative == alternative
    assert res.method == 'chi2'
    assert res.sample_stat == 'std(x)'
    assert res.sample_stat_target == H0_std
    assert res.sample_stat_value == np.std(x, ddof=1)
    assert isinstance(res.stat, numbers.Number)
    assert res.pvalue == pytest.approx(0.7358, abs=1e-4)

    alternative = 'greater'
    res = mqr.inference.stddev.test_1sample(x, H0_std, alternative)
    assert res.description == 'standard deviation'
    assert res.alternative == alternative
    assert res.method == 'chi2'
    assert res.sample_stat == 'std(x)'
    assert res.sample_stat_target == H0_std
    assert res.sample_stat_value == np.std(x, ddof=1)
    assert isinstance(res.stat, numbers.Number)
    assert res.pvalue == pytest.approx(0.3679, abs=1e-4)

    alternative = 'less'
    res = mqr.inference.stddev.test_1sample(x, H0_std, alternative)
    assert res.description == 'standard deviation'
    assert res.alternative == alternative
    assert res.method == 'chi2'
    assert res.sample_stat == 'std(x)'
    assert res.sample_stat_target == H0_std
    assert res.sample_stat_value == np.std(x, ddof=1)
    assert isinstance(res.stat, numbers.Number)
    assert res.pvalue == pytest.approx(0.6321, abs=1e-4)

def test_test_2sample():
    x = np.array([0, 1, 2])
    y = np.array([1, 2, 4])

    method = 'f'
    alternative = 'two-sided'
    res = mqr.inference.stddev.test_2sample(x, y, alternative)
    assert res.description == 'ratio of standard deviations'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_stat == 'std(x) / std(y)'
    assert res.sample_stat_target == 1.0
    assert res.sample_stat_value == np.std(x, ddof=1) / np.std(y, ddof=1)
    assert isinstance(res.stat, numbers.Number)
    assert res.pvalue == pytest.approx(0.6)

    alternative = 'less'
    res = mqr.inference.stddev.test_2sample(x, y, alternative)
    assert res.description == 'ratio of standard deviations'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_stat == 'std(x) / std(y)'
    assert res.sample_stat_target == 1.0
    assert res.sample_stat_value == np.std(x, ddof=1) / np.std(y, ddof=1)
    assert isinstance(res.stat, numbers.Number)
    assert res.pvalue == pytest.approx(0.3)

    alternative = 'greater'
    res = mqr.inference.stddev.test_2sample(x, y, alternative)
    assert res.description == 'ratio of standard deviations'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_stat == 'std(x) / std(y)'
    assert res.sample_stat_target == 1.0
    assert res.sample_stat_value == np.std(x, ddof=1) / np.std(y, ddof=1)
    assert isinstance(res.stat, numbers.Number)
    assert res.pvalue == pytest.approx(0.7)
