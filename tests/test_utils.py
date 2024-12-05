'''
Check call-throughs.
'''

import mqr.utils as utils

import numpy as np
import pytest

def test_make_ordinal():
    assert utils.make_ordinal(1) == '1st'
    assert utils.make_ordinal(2) == '2nd'
    assert utils.make_ordinal(3) == '3rd'
    assert utils.make_ordinal(4) == '4th'
    assert utils.make_ordinal(5) == '5th'

    assert utils.make_ordinal(10) == '10th'
    assert utils.make_ordinal(11) == '11th'
    assert utils.make_ordinal(12) == '12th'
    assert utils.make_ordinal(13) == '13th'
    assert utils.make_ordinal(14) == '14th'

    assert utils.make_ordinal(21) == '21st'
    assert utils.make_ordinal(22) == '22nd'
    assert utils.make_ordinal(23) == '23rd'
    assert utils.make_ordinal(24) == '24th'
    assert utils.make_ordinal(25) == '25th'

    assert utils.make_ordinal(1.1) == '1.1th'
    assert utils.make_ordinal(5.1) == '5.1th'
    assert utils.make_ordinal(134.1) == '134.1th'

def test_clip_where():
    assert utils.clip_where(0, -1, 1, True) == 0
    assert utils.clip_where(0, 1, 2, True) == 1
    assert utils.clip_where(0, -2, -1, True) == -1

    assert utils.clip_where(0, -1, 1, False) == 0
    assert utils.clip_where(0, 1, 2, False) == 0
    assert utils.clip_where(0, -2, -1, False) == 0

    a = np.array([1, 1, 1, 1, 1, 1])
    res = utils.clip_where(
        a,
        np.array([0, 0, 1, 1, 2, 2]),
        np.inf,
        np.array([1, 0, 1, 0, 1, 0]))
    assert list(res) == [1, 1, 1, 1, 2, 1]
    res = utils.clip_where(
        a,
        -np.inf,
        np.array([0, 0, 1, 1, 2, 2]),
        np.array([1, 0, 1, 0, 1, 0]))
    assert list(res) == [0, 1, 1, 1, 1, 1]
    res = utils.clip_where(
        a,
        np.array([0, 0, 1, 1, 2, 2]),
        np.array([0, 0, 1, 1, 2, 2]),
        np.array([1, 0, 1, 0, 1, 0]))
    assert list(res) == [0, 1, 1, 1, 2, 1]

    a = np.array([1, 1, 1, 1, 1, 1])
    res = utils.clip_where(
        a,
        2,
        3,
        np.array([1, 0, 1, 0, 1, 0]))
    assert list(res) == [2, 1, 2, 1, 2, 1]
    res = utils.clip_where(
        a,
        np.array([0, 0, 1, 1, 2, 2]),
        np.array([0, 0, 1, 1, 2, 2]),
        True)
    assert list(res) == [0, 0, 1, 1, 2, 2]
    res = utils.clip_where(a, 2, 3, True)
    list(res) == [0, 0, 1, 1, 2, 2]
    res = utils.clip_where(a, 2, 3, False)
    list(res) == [1, 1, 1, 1, 1, 1]

def test_fredholm2_1():
    fn_g = lambda t: np.sin(t)
    lmda = 1
    fn_K = lambda t, s: np.sin(t) * np.cos(s)

    c, d = 0, np.pi / 2
    x, w = np.polynomial.legendre.leggauss(10)
    x = c + (x + 1) * (d - c) / 2
    w = w * (d - c) / 2

    t0 = np.linspace(c, d)
    exact_result = 2 * np.sin(t0)

    for i in range(len(t0)):
        fred_result = utils.fredholm2(
            t0=t0[i],
            fn_K=fn_K,
            fn_g=fn_g,
            lmda=lmda,
            x=x,
            w=w)
        assert exact_result[i] == pytest.approx(fred_result, abs=1e-15)

def test_fredholm2_2():
    fn_g = lambda t: 1
    lmda = 1
    fn_K = lambda t, s: t

    alpha = 0.9876
    c, d = 0, alpha
    x, w = np.polynomial.legendre.leggauss(10)
    x = c + (x + 1) * (d - c) / 2
    w = w * (d - c) / 2

    t0 = np.linspace(c, d)
    exact_result = 1 + 2 * alpha / (2 - alpha**2) * t0

    for i in range(len(t0)):
        fred_result = utils.fredholm2(
            t0=t0[i],
            fn_K=fn_K,
            fn_g=fn_g,
            lmda=lmda,
            x=x,
            w=w)
        assert exact_result[i] == pytest.approx(fred_result, abs=1e-15)
