'''
Check call-throughs.
'''

import numbers
import numpy as np
import pytest

import mqr

def test_power_1sample():
    p0 = 0.4
    alpha = 0.05
    nobs = 200

    pa = 0.5
    alternative = 'two-sided'
    method = 'norm-approx'
    res = mqr.inference.proportion.power_1sample(pa, p0, nobs, alpha, alternative, method)
    assert res.name == 'proportion'
    assert res.alpha == 0.05
    assert res.beta == pytest.approx(0.181922, abs=1e-6)
    assert res.effect == '0.5 - 0.4 = 0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_size == nobs

    alternative = 'greater'
    res = mqr.inference.proportion.power_1sample(pa, p0, nobs, alpha, alternative, method)
    assert res.name == 'proportion'
    assert res.alpha == 0.05
    assert res.beta == pytest.approx(0.111839, abs=1e-6)
    assert res.effect == '0.5 - 0.4 = 0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_size == nobs

    pa = 0.3
    alternative = 'less'
    res = mqr.inference.proportion.power_1sample(pa, p0, nobs, alpha, alternative, method)
    assert res.name == 'proportion'
    assert res.alpha == 0.05
    assert res.beta == pytest.approx(0.0921478, abs=1e-6)
    assert res.effect == '0.3 - 0.4 = -0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_size == nobs

def test_power_2sample():
    p2 = 0.4
    alpha = 0.05
    nobs = 200

    p1 = 0.5
    alternative = 'two-sided'
    method = 'norm-approx'
    res = mqr.inference.proportion.power_2sample(p1, p2, nobs, alpha, alternative, method)
    assert res.name == 'difference between proportions'
    assert res.alpha == 0.05
    assert res.beta == pytest.approx(0.479882, abs=1e-6)
    assert res.effect == '0.5 - 0.4 = 0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_size == nobs

    alternative = 'greater'
    res = mqr.inference.proportion.power_2sample(p1, p2, nobs, alpha, alternative, method)
    assert res.name == 'difference between proportions'
    assert res.alpha == 0.05
    assert res.beta == pytest.approx(0.356779, abs=1e-6)
    assert res.effect == '0.5 - 0.4 = 0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_size == nobs

    p1 = 0.3
    alternative = 'less'
    res = mqr.inference.proportion.power_2sample(p1, p2, nobs, alpha, alternative, method)
    assert res.name == 'difference between proportions'
    assert res.alpha == 0.05
    assert res.beta == pytest.approx(0.324836, abs=1e-6)
    assert res.effect == '0.3 - 0.4 = -0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert res.sample_size == nobs

def test_size_1sample():
    p0 = 0.4
    alpha = 0.05
    beta = 0.05

    pa = 0.5
    alternative = 'two-sided'
    method = 'norm-approx'
    res = mqr.inference.proportion.size_1sample(pa, p0, alpha, beta, alternative, method)
    assert res.name == 'proportion'
    assert res.alpha == 0.05
    assert res.beta == 0.05
    assert res.effect == '0.5 - 0.4 = 0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert np.ceil(res.sample_size) == 318

    alternative = 'greater'
    res = mqr.inference.proportion.size_1sample(pa, p0, alpha, beta, alternative, method)
    assert res.name == 'proportion'
    assert res.alpha == 0.05
    assert res.beta == 0.05
    assert res.effect == '0.5 - 0.4 = 0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert np.ceil(res.sample_size) == 266

    pa = 0.3
    alternative = 'less'
    res = mqr.inference.proportion.size_1sample(pa, p0, alpha, beta, alternative, method)
    assert res.name == 'proportion'
    assert res.alpha == 0.05
    assert res.beta == 0.05
    assert res.effect == '0.3 - 0.4 = -0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert np.ceil(res.sample_size) == 244

    pa = 0.5
    alternative = 'two-sided'
    method = 'arcsin'
    res = mqr.inference.proportion.size_1sample(pa, p0, alpha, beta, alternative, method)
    assert res.name == 'proportion'
    assert res.alpha == 0.05
    assert res.beta == 0.05
    assert res.effect == '0.5 - 0.4 = 0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert np.ceil(res.sample_size) == 321

    alternative = 'greater'
    res = mqr.inference.proportion.size_1sample(pa, p0, alpha, beta, alternative, method)
    assert res.name == 'proportion'
    assert res.alpha == 0.05
    assert res.beta == 0.05
    assert res.effect == '0.5 - 0.4 = 0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert np.ceil(res.sample_size) == 267

    pa = 0.3
    alternative = 'less'
    res = mqr.inference.proportion.size_1sample(pa, p0, alpha, beta, alternative, method)
    assert res.name == 'proportion'
    assert res.alpha == 0.05
    assert res.beta == 0.05
    assert res.effect == '0.3 - 0.4 = -0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert np.ceil(res.sample_size) == 246

def test_size_2sample():
    p2 = 0.4
    alpha = 0.05
    beta = 0.05

    p1 = 0.5
    alternative = 'two-sided'
    method = 'norm-approx'
    res = mqr.inference.proportion.size_2sample(p1, p2, alpha, beta, alternative, method)
    assert res.name == 'proportion'
    assert res.alpha == 0.05
    assert res.beta == 0.05
    assert res.effect == '0.5 - 0.4 = 0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert np.ceil(res.sample_size) == 641

    alternative = 'greater'
    res = mqr.inference.proportion.size_2sample(p1, p2, alpha, beta, alternative, method)
    assert res.name == 'proportion'
    assert res.alpha == 0.05
    assert res.beta == 0.05
    assert res.effect == '0.5 - 0.4 = 0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert np.ceil(res.sample_size) == 533

    p1 = 0.3
    alternative = 'less'
    res = mqr.inference.proportion.size_2sample(p1, p2, alpha, beta, alternative, method)
    assert res.name == 'proportion'
    assert res.alpha == 0.05
    assert res.beta == 0.05
    assert res.effect == '0.3 - 0.4 = -0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert np.ceil(res.sample_size) == 490

    p1 = 0.5
    alternative = 'two-sided'
    method = 'arcsin'
    res = mqr.inference.proportion.size_2sample(p1, p2, alpha, beta, alternative, method)
    assert res.name == 'proportion'
    assert res.alpha == 0.05
    assert res.beta == 0.05
    assert res.effect == '0.5 - 0.4 = 0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert np.ceil(res.sample_size) == 642

    alternative = 'greater'
    res = mqr.inference.proportion.size_2sample(p1, p2, alpha, beta, alternative, method)
    assert res.name == 'proportion'
    assert res.alpha == 0.05
    assert res.beta == 0.05
    assert res.effect == '0.5 - 0.4 = 0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert np.ceil(res.sample_size) == 534

    p1 = 0.3
    alternative = 'less'
    res = mqr.inference.proportion.size_2sample(p1, p2, alpha, beta, alternative, method)
    assert res.name == 'proportion'
    assert res.alpha == 0.05
    assert res.beta == 0.05
    assert res.effect == '0.3 - 0.4 = -0.1'
    assert res.alternative == alternative
    assert res.method == method
    assert np.ceil(res.sample_size) == 491

def test_confint_1sample():
    count = 5
    nobs = 10
    conf = 0.90
    method = 'beta'
    bounded = 'both'

    res = mqr.inference.proportion.confint_1sample(count, nobs, conf, method=method)
    assert res.name == 'proportion'
    assert res.method == 'beta'
    assert res.value == count / nobs
    assert isinstance(res.lower, numbers.Number)
    assert isinstance(res.upper, numbers.Number)
    assert res.conf == conf
    assert res.bounded == bounded

    count = np.array([0, 20, 40])
    nobs = 40

    bounded = 'both'
    lowers, uppers = list(mqr.inference.proportion.confint_1sample(count, nobs, bounded=bounded, method='agresti-coull'))
    assert lowers == pytest.approx(np.array([0.        , 0.35199527, 0.89560378]))
    assert uppers == pytest.approx(np.array([0.10439622, 0.64800473, 1.        ]))
    lowers, uppers = list(mqr.inference.proportion.confint_1sample(count, nobs, bounded=bounded, method='jeffreys'))
    assert lowers == pytest.approx(np.array([0.        , 0.3496035 , 0.93950202]))
    assert uppers == pytest.approx(np.array([0.06049798, 0.6503965 , 1.        ]))
    lowers, uppers = list(mqr.inference.proportion.confint_1sample(count, nobs, bounded=bounded, method='wilson-cc'))
    assert lowers == pytest.approx(np.array([0.        , 0.34063274, 0.89087533]))
    assert uppers == pytest.approx(np.array([0.10912467, 0.65936726, 1.        ]))

    bounded = 'below'
    lower, upper = mqr.inference.proportion.confint_1sample(count, nobs, bounded=bounded, method='agresti-coull')
    assert lower == pytest.approx(np.array([0.        , 0.37414945, 0.92424085]))
    assert upper == pytest.approx(np.array([1., 1., 1.]))
    lower, upper = mqr.inference.proportion.confint_1sample(count, nobs, bounded=bounded, method='jeffreys')
    assert lower == pytest.approx(np.array([0.        , 0.37290554, 0.95340272]))
    assert upper == pytest.approx(np.array([1., 1., 1.]))
    lower, upper = mqr.inference.proportion.confint_1sample(count, nobs, bounded=bounded, method='wilson-cc')
    assert lower == pytest.approx(np.array([0.        , 0.36247821, 0.91495118]))
    assert upper == pytest.approx(np.array([1., 1., 1.]))

    bounded = 'above'
    lower, upper = mqr.inference.proportion.confint_1sample(count, nobs, bounded=bounded, method='agresti-coull')
    assert lower == pytest.approx(np.array([0., 0., 0.]))
    assert upper == pytest.approx(np.array([0.07575915, 0.62585055, 1.        ]))
    lower, upper = mqr.inference.proportion.confint_1sample(count, nobs, bounded=bounded, method='jeffreys')
    assert lower == pytest.approx(np.array([0., 0., 0.]))
    assert upper == pytest.approx(np.array([0.04659728, 0.62709446, 1.        ]))
    lower, upper = mqr.inference.proportion.confint_1sample(count, nobs, bounded=bounded, method='wilson-cc')
    assert lower == pytest.approx(np.array([0., 0., 0.]))
    assert upper == pytest.approx(np.array([0.08504882, 0.63752179, 1.        ]))

def test_confint_1sample_newcomb():
    """
    Checks values against Table I in [1]_.

    References
    ----------
    .. [1] Newcombe, R. G. (1998).
       Two‐sided confidence intervals for the single proportion:
       comparison of seven methods. Statistics in medicine, 17(8), 857-872.
    """
    from mqr.inference.lib.proportion import confint_1sample_wilson as wilson
    from mqr.inference.lib.proportion import confint_1sample_wilson_cc as wilson_cc

    lower, upper = wilson(81, 263, 0.95, 'both')
    assert lower == pytest.approx(0.2553, abs=1e-4)
    assert upper == pytest.approx(0.3662, abs=1e-4)
    lower, upper = wilson_cc(81, 263, 0.95, 'both')
    assert lower == pytest.approx(0.2535, abs=1e-4)
    assert upper == pytest.approx(0.3682, abs=1e-4)

    lower, upper = wilson(15, 148, 0.95, 'both')
    assert lower == pytest.approx(0.0624, abs=1e-4)
    assert upper == pytest.approx(0.1605, abs=1e-4)
    lower, upper = wilson_cc(15, 148, 0.95, 'both')
    assert lower == pytest.approx(0.0598, abs=1e-4)
    assert upper == pytest.approx(0.1644, abs=1e-4)

    lower, upper = wilson(0, 20, 0.95, 'both')
    assert lower == pytest.approx(0.0000, abs=1e-4)
    assert upper == pytest.approx(0.1611, abs=1e-4)
    lower, upper = wilson_cc(0, 20, 0.95, 'both')
    assert lower == pytest.approx(0.0000, abs=1e-4)
    assert upper == pytest.approx(0.2005, abs=1e-4)

    lower, upper = wilson(1, 29, 0.95, 'both')
    assert lower == pytest.approx(0.0061, abs=1e-4)
    assert upper == pytest.approx(0.1718, abs=1e-4)
    lower, upper = wilson_cc(1, 29, 0.95, 'both')
    assert lower == pytest.approx(0.0018, abs=1e-4)
    assert upper == pytest.approx(0.1963, abs=1e-4)

def test_confint_1sample_brown_cai_dasgupta():
    """
    Checks a few values against Table 5 in [1]_.

    References
    ----------
    [1] Brown, L. D. Cai, T. T. and DasGupta, A. (2001).
        "Interval estimation for a binomial proportion".
        Statistical Science, 16(2), 101-133.
    """
    from mqr.inference.lib.proportion import confint_1sample_jeffreys as jeffreys

    lower, upper = jeffreys(0, 7, 0.95, 'both')
    assert lower == 0.0
    assert upper == pytest.approx(0.292, abs=1e-3)

    lower, upper = jeffreys(3, 10, 0.95, 'both')
    assert lower == pytest.approx(0.093, abs=1e-3)
    assert upper == pytest.approx(0.606, abs=1e-3)

    lower, upper = jeffreys(1, 27, 0.95, 'both')
    assert lower == pytest.approx(0.004, abs=1e-3)
    assert upper == pytest.approx(0.160, abs=1e-3)

    lower, upper = jeffreys(13, 27, 0.95, 'both')
    assert lower == pytest.approx(0.303, abs=1e-3)
    assert upper == pytest.approx(0.664, abs=1e-3)

    lower, upper = jeffreys(15, 30, 0.95, 'both')
    assert lower == pytest.approx(0.328, abs=1e-3)
    assert upper == pytest.approx(0.672, abs=1e-3)

def test_confint_2sample():
    count1 = 5
    nobs1 = 10
    count2 = 15
    nobs2 = 30
    conf = 0.90
    bounded = 'both'

    res = mqr.inference.proportion.confint_2sample(count1, nobs1, count2, nobs2, conf, method='newcomb')
    assert res.name == 'difference between proportions'
    assert res.method == 'newcomb'
    assert res.value == count1 / nobs1 - count2 / nobs2
    assert isinstance(res.lower, numbers.Number)
    assert isinstance(res.upper, numbers.Number)
    assert res.conf == conf
    assert res.bounded == bounded

    conf = 0.95
    count1 = np.array([0, 30, 55, 60])
    nobs1 = 60
    count2 = np.array([0, 20, 30, 40])
    nobs2 = 40

    bounded = 'both'
    lower, upper = mqr.inference.proportion.confint_2sample(count1, nobs1, count2, nobs2, bounded=bounded, method='agresti-caffo')
    assert lower == pytest.approx(np.array([-0.06343951, -0.19584581,  0.0131549 , -0.04807853]))
    assert upper == pytest.approx(np.array([0.04807853, 0.19584581, 0.31710624, 0.06343951]))
    lower, upper = mqr.inference.proportion.confint_2sample(count1, nobs1, count2, nobs2, bounded=bounded, method='newcomb-cc')
    assert lower == pytest.approx(np.array([-0.10912467, -0.20595924,  0.00715208, -0.07496504]))
    assert upper == pytest.approx(np.array([0.07496504, 0.20595924, 0.33993579, 0.10912467]))

    bounded = 'below'
    lower, upper = mqr.inference.proportion.confint_2sample(count1, nobs1, count2, nobs2, bounded=bounded, method='agresti-caffo')
    assert lower == pytest.approx(np.array([-0.05447494, -0.16435898,  0.03758857, -0.03911395]))
    assert upper == pytest.approx(np.array([np.inf, np.inf, np.inf, np.inf]))
    lower, upper = mqr.inference.proportion.confint_2sample(count1, nobs1, count2, nobs2, bounded=bounded, method='newcomb-cc')
    assert lower == pytest.approx(np.array([-0.08504882, -0.17724485,  0.02995461, -0.05794766]))
    assert upper == pytest.approx(np.array([np.inf, np.inf, np.inf, np.inf]))

    bounded = 'above'
    lower, upper = mqr.inference.proportion.confint_2sample(count1, nobs1, count2, nobs2, bounded=bounded, method='agresti-caffo')
    assert lower == pytest.approx(np.array([-np.inf, -np.inf, -np.inf, -np.inf]))
    assert upper == pytest.approx(np.array([0.03911395, 0.16435898, 0.29267257, 0.05447494]))
    lower, upper = mqr.inference.proportion.confint_2sample(count1, nobs1, count2, nobs2, bounded=bounded, method='newcomb-cc')
    assert lower == pytest.approx(np.array([-np.inf, -np.inf, -np.inf, -np.inf]))
    assert upper == pytest.approx(np.array([0.05794766, 0.17724485, 0.31391672, 0.08504882]))

def test_test_1sample():
    count = 5
    nobs = 10
    H0_prop = 0.6
    alternative = 'two-sided'

    res = mqr.inference.proportion.test_1sample(count, nobs, H0_prop, alternative, 'binom')
    assert res.description == 'proportion of "true" elements'
    assert res.alternative == alternative
    assert res.method == 'binom'
    assert res.sample_stat == 'count / nobs'
    assert res.sample_stat_target == H0_prop
    assert res.sample_stat_value == count / nobs
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)

    res = mqr.inference.proportion.test_1sample(count, nobs, H0_prop, alternative, 'z')
    assert res.description == 'proportion of "true" elements'
    assert res.alternative == alternative
    assert res.method == 'z'
    assert res.sample_stat == 'count / nobs'
    assert res.sample_stat_target == H0_prop
    assert res.sample_stat_value == count / nobs
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)

    res = mqr.inference.proportion.test_1sample(count, nobs, H0_prop, alternative, 'chi2')
    assert res.description == 'proportion of "true" elements'
    assert res.alternative == alternative
    assert res.method == 'chi2'
    assert res.sample_stat == 'count / nobs'
    assert res.sample_stat_target == H0_prop
    assert res.sample_stat_value == count / nobs
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)

def test_test_2sample():
    count1 = 5
    nobs1 = 10
    count2 = 15
    nobs2 = 60
    H0_diff = 0.5 - 0.25
    alternative = 'two-sided'

    res = mqr.inference.proportion.test_2sample(count1, nobs1, count2, nobs2, H0_diff, alternative)
    assert res.description == 'difference between proportions of "true" elements'
    assert res.alternative == alternative
    assert res.method == 'agresti-caffo'
    assert res.sample_stat == 'count1 / nobs1 - count2 / nobs2'
    assert res.sample_stat_target == H0_diff
    assert res.sample_stat_value == count1 / nobs1 - count2 / nobs2
    assert isinstance(res.stat, numbers.Number)
    assert isinstance(res.pvalue, numbers.Number)

def test_confint_2sample_newcomb():
    """
    Checks values against Table II in [1]_.

    References
    ----------
    .. [1] Newcombe, R. G. (1998).
       Interval estimation for the difference between independent proportions:
       comparison of eleven methods.
       Statistics in medicine, 17(8), 873-890.
    """
    from mqr.inference.lib.proportion import confint_2sample_newcomb as newcomb
    from mqr.inference.lib.proportion import confint_2sample_newcomb_cc as newcomb_cc

    conf = 0.95
    bounded = 'both'

    # (a)
    lower, upper = newcomb(56, 70, 48, 80, conf, bounded)
    assert lower == pytest.approx(0.0524, abs=1e-4)
    assert upper == pytest.approx(0.3339, abs=1e-4)
    lower, upper = newcomb_cc(56, 70, 48, 80, conf, bounded)
    assert lower == pytest.approx(0.0428, abs=1e-4)
    assert upper == pytest.approx(0.3422, abs=1e-4)

    # (b)
    lower, upper = newcomb(9, 10, 3, 10, conf, bounded)
    assert lower == pytest.approx(0.1705, abs=1e-4)
    assert upper == pytest.approx(0.8090, abs=1e-4)
    lower, upper = newcomb_cc(9, 10, 3, 10, conf, bounded)
    assert lower == pytest.approx(0.1013, abs=1e-4)
    assert upper == pytest.approx(0.8387, abs=1e-4)

    # (c)
    lower, upper = newcomb(6, 7, 2, 7, conf, bounded)
    assert lower == pytest.approx(0.0582, abs=1e-4)
    assert upper == pytest.approx(0.8062, abs=1e-4)
    lower, upper = newcomb_cc(6, 7, 2, 7, conf, bounded)
    assert lower == pytest.approx(-0.0290, abs=1e-4)
    assert upper == pytest.approx(0.8423, abs=1e-4)

    # (d)
    lower, upper = newcomb(5, 56, 0, 29, conf, bounded)
    assert lower == pytest.approx(-0.0381, abs=1e-4)
    assert upper == pytest.approx(0.1926, abs=1e-4)
    lower, upper = newcomb_cc(5, 56, 0, 29, conf, bounded)
    assert lower == pytest.approx(-0.0667, abs=1e-4)
    assert upper == pytest.approx(0.2037, abs=1e-4)

    # (e)
    lower, upper = newcomb(0, 10, 0, 20, conf, bounded)
    assert lower == pytest.approx(-0.1611, abs=1e-4)
    assert upper == pytest.approx(0.2775, abs=1e-4)
    lower, upper = newcomb_cc(0, 10, 0, 20, conf, bounded)
    assert lower == pytest.approx(-0.2005, abs=1e-4)
    assert upper == pytest.approx(0.3445, abs=1e-4)

    # (f)
    lower, upper = newcomb(0, 10, 0, 10, conf, bounded)
    assert lower == pytest.approx(-0.2775, abs=1e-4)
    assert upper == pytest.approx(0.2775, abs=1e-4)
    lower, upper = newcomb_cc(0, 10, 0, 10, conf, bounded)
    assert lower == pytest.approx(-0.3445, abs=1e-4)
    assert upper == pytest.approx(0.3445, abs=1e-4)

    # (g)
    lower, upper = newcomb(10, 10, 0, 20, conf, bounded)
    assert lower == pytest.approx(0.6791, abs=1e-4)
    assert upper == pytest.approx(1.0000, abs=1e-4)
    lower, upper = newcomb_cc(10, 10, 0, 20, conf, bounded)
    assert lower == pytest.approx(0.6014, abs=1e-4)
    assert upper == pytest.approx(1.0000, abs=1e-4)

    # (h)
    lower, upper = newcomb(10, 10, 0, 10, conf, bounded)
    assert lower == pytest.approx(0.6075, abs=1e-4)
    assert upper == pytest.approx(1.0000, abs=1e-4)
    lower, upper = newcomb_cc(10, 10, 0, 10, conf, bounded)
    assert lower == pytest.approx(0.5128, abs=1e-4)
    assert upper == pytest.approx(1.0000, abs=1e-4)
