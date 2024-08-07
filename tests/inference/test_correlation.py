'''
Check call-throughs.
'''

import numbers
import numpy as np
import pytest
import scipy

import mqr

def test_confint():
    x = np.array([1, 2, 4, 5])
    y = np.array([1, 3, 4, 4])
    conf = 0.90

    bounded = 'both'
    res = mqr.inference.correlation.confint(x, y, conf, bounded)
    assert res.name == 'correlation'
    assert res.method == 'fisher-z'
    assert res.value == pytest.approx(0.9036961141150641)
    assert res.lower == pytest.approx(-0.151652753)
    assert res.upper == pytest.approx(0.996236807)
    assert res.conf == conf

    bounded = 'above'
    res = mqr.inference.correlation.confint(x, y, conf, bounded)
    assert res.name == 'correlation'
    assert res.method == 'fisher-z'
    assert res.value == pytest.approx(0.9036961141150641)
    assert res.lower == -1.0
    assert res.upper == pytest.approx(0.9922332)
    assert res.conf == conf

    bounded = 'below'
    res = mqr.inference.correlation.confint(x, y, conf, bounded)
    assert res.name == 'correlation'
    assert res.method == 'fisher-z'
    assert res.value == pytest.approx(0.9036961141150641)
    assert res.lower == pytest.approx(0.2074167)
    assert res.upper == 1.0
    assert res.conf == conf

def test_test():
    x = np.array([1, 2, 4, 5])
    y = np.array([1, 3, 4, 4])

    H0_corr = 0.0
    alternative = 'two-sided'
    res = mqr.inference.correlation.test(x, y, H0_corr, alternative)
    assert res.description == 'correlation coefficient'
    assert res.alternative == alternative
    assert res.method == 'pearson'
    assert res.sample_stat == 'corr(x, y)'
    assert res.sample_stat_target == H0_corr
    assert res.sample_stat_value == pytest.approx(0.9037, abs=1e-4)
    assert res.stat == pytest.approx(2.9848, abs=1e-4)
    assert res.pvalue == pytest.approx(0.0963, abs=1e-4)

    H0_corr = 0.0
    alternative = 'less'
    res = mqr.inference.correlation.test(x, y, H0_corr, alternative)
    assert res.description == 'correlation coefficient'
    assert res.alternative == alternative
    assert res.method == 'pearson'
    assert res.sample_stat == 'corr(x, y)'
    assert res.sample_stat_target == H0_corr
    assert res.sample_stat_value == pytest.approx(0.9037, abs=1e-4)
    assert res.stat == pytest.approx(2.9848, abs=1e-4)
    assert res.pvalue == pytest.approx(0.9518, abs=1e-4)

    H0_corr = 0.0
    alternative = 'greater'
    res = mqr.inference.correlation.test(x, y, H0_corr, alternative)
    assert res.description == 'correlation coefficient'
    assert res.alternative == alternative
    assert res.method == 'pearson'
    assert res.sample_stat == 'corr(x, y)'
    assert res.sample_stat_target == H0_corr
    assert res.sample_stat_value == pytest.approx(0.9037, abs=1e-4)
    assert res.stat == pytest.approx(2.9848, abs=1e-4)
    assert res.pvalue == pytest.approx(0.04815, abs=1e-4)

    H0_corr = 0.4
    alternative = 'two-sided'
    res = mqr.inference.correlation.test(x, y, H0_corr, alternative)
    assert res.description == 'correlation coefficient'
    assert res.alternative == alternative
    assert res.method == 'fisher-z'
    assert res.sample_stat == 'corr(x, y)'
    assert res.sample_stat_target == H0_corr
    assert res.sample_stat_value == pytest.approx(0.9037, abs=1e-4)
    assert res.stat == pytest.approx(1.4920, abs=1e-4)
    assert res.pvalue == pytest.approx(0.2853, abs=1e-4)

    H0_corr = 0.4
    alternative = 'less'
    res = mqr.inference.correlation.test(x, y, H0_corr, alternative)
    assert res.description == 'correlation coefficient'
    assert res.alternative == alternative
    assert res.method == 'fisher-z'
    assert res.sample_stat == 'corr(x, y)'
    assert res.sample_stat_target == H0_corr
    assert res.sample_stat_value == pytest.approx(0.9037, abs=1e-4)
    assert res.stat == pytest.approx(1.4920, abs=1e-4)
    assert res.pvalue == pytest.approx(0.8573, abs=1e-4)

    H0_corr = 0.4
    alternative = 'greater'
    res = mqr.inference.correlation.test(x, y, H0_corr, alternative)
    assert res.description == 'correlation coefficient'
    assert res.alternative == alternative
    assert res.method == 'fisher-z'
    assert res.sample_stat == 'corr(x, y)'
    assert res.sample_stat_target == H0_corr
    assert res.sample_stat_value == pytest.approx(0.9037, abs=1e-4)
    assert res.stat == pytest.approx(1.4920, abs=1e-4)
    assert res.pvalue == pytest.approx(0.1427, abs=1e-4)

def test_test_diff():
    x1 = np.array([1, 2, 4, 5])
    y1 = np.array([1, 3, 4, 4])
    x2 = np.array([1, 2, 4, 5])
    y2 = np.array([3, 3, 2, 0])
    H0_corr1 = 0.8
    H0_corr2 = -0.1

    alternative = 'two-sided'
    res = mqr.inference.correlation.test_diff(x1, y1, x2, y2, H0_corr1, H0_corr2, alternative)
    assert res.description == 'difference between correlation coefficients'
    assert res.alternative == alternative
    assert res.method == 'fisher-z'
    assert res.sample_stat == 'corr(x1, y1) - corr(x2, y2)'
    assert res.sample_stat_target == H0_corr1 - H0_corr2
    assert res.sample_stat_value == pytest.approx(1.8074, abs=1e-4)
    assert res.stat == pytest.approx(2.9840, abs=1e-4)
    assert res.pvalue == pytest.approx(0.2068, abs=1e-4)

    alternative = 'less'
    res = mqr.inference.correlation.test_diff(x1, y1, x2, y2, H0_corr1, H0_corr2, alternative)
    assert res.description == 'difference between correlation coefficients'
    assert res.alternative == alternative
    assert res.method == 'fisher-z'
    assert res.sample_stat == 'corr(x1, y1) - corr(x2, y2)'
    assert res.sample_stat_target == H0_corr1 - H0_corr2
    assert res.sample_stat_value == pytest.approx(1.8074, abs=1e-4)
    assert res.stat == pytest.approx(2.9840, abs=1e-4)
    assert res.pvalue == pytest.approx(0.8966, abs=1e-4)

    alternative = 'greater'
    res = mqr.inference.correlation.test_diff(x1, y1, x2, y2, H0_corr1, H0_corr2, alternative)
    assert res.description == 'difference between correlation coefficients'
    assert res.alternative == alternative
    assert res.method == 'fisher-z'
    assert res.sample_stat == 'corr(x1, y1) - corr(x2, y2)'
    assert res.sample_stat_target == H0_corr1 - H0_corr2
    assert res.sample_stat_value == pytest.approx(1.8074, abs=1e-4)
    assert res.stat == pytest.approx(2.9840, abs=1e-4)
    assert res.pvalue == pytest.approx(0.1034, abs=1e-4)
