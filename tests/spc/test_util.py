import mqr.spc.util as util

import numpy as np
import pandas as pd
import pytest

def test_c4():
    """
    Checks values against Appendix VI in [1]_.

    References
    ----------
    .. [1]  Montgomery, D. C. (2009).
            Statistical quality control (Vol. 7).
            New York: Wiley.
    """
    n = np.arange(2, 26)
    tab_result = np.array([0.7979, 0.8862, 0.9213, 0.9400, 0.9515, 0.9594,
                           0.9650, 0.9694, 0.9727, 0.9754, 0.9776, 0.9794,
                           0.9810, 0.9823, 0.9835, 0.9845, 0.9854, 0.9862,
                           0.9869, 0.9876, 0.9882, 0.9887, 0.9892, 0.9896,])
    for i in range(len(n)):
        assert util.c4(n[i]) == pytest.approx(tab_result[i], abs=1e-4)

def test_d2():
    """
    Checks values against Appendix VI in [1]_.

    References
    ----------
    .. [1]  Montgomery, D. C. (2009).
            Statistical quality control (Vol. 7).
            New York: Wiley.
    """
    n = np.arange(2, 26)
    tab_result = np.array([1.128, 1.693, 2.059, 2.326, 2.534, 2.704,
                           2.847, 2.970, 3.078, 3.173, 3.258, 3.336,
                           3.407, 3.472, 3.532, 3.588, 3.640, 3.689,
                           3.735, 3.778, 3.819, 3.858, 3.895, 3.931,])
    for i in range(len(n)):
        assert util.d2(n[i]) == pytest.approx(tab_result[i], abs=1e-3)

def test_d3():
    """
    Checks values against Appendix VI in [1]_.

    References
    ----------
    .. [1]  Montgomery, D. C. (2009).
            Statistical quality control (Vol. 7).
            New York: Wiley.
    """
    n = np.arange(2, 26)
    tab_result = np.array([0.853, 0.888, 0.880, 0.864, 0.848, 0.833,
                           0.820, 0.808, 0.797, 0.787, 0.778, 0.770,
                           0.763, 0.756, 0.750, 0.744, 0.739, 0.734,
                           0.729, 0.724, 0.720, 0.716, 0.712, 0.708,])
    for i in range(len(n)):
        assert util.d3(n[i]) == pytest.approx(tab_result[i], abs=1e-3)

def test_c4_fn():
    """
    Checks values against Appendix VI in [1]_.

    References
    ----------
    .. [1]  Montgomery, D. C. (2009).
            Statistical quality control (Vol. 7).
            New York: Wiley.
    """
    n = np.arange(2, 26)
    tab_result = np.array([0.7979, 0.8862, 0.9213, 0.9400, 0.9515, 0.9594,
                           0.9650, 0.9694, 0.9727, 0.9754, 0.9776, 0.9794,
                           0.9810, 0.9823, 0.9835, 0.9845, 0.9854, 0.9862,
                           0.9869, 0.9876, 0.9882, 0.9887, 0.9892, 0.9896,])
    for i in range(len(n)):
        assert util.c4_fn(n[i]) == pytest.approx(tab_result[i], abs=1e-4)

def test_d2_integral():
    """
    Checks values against Appendix VI in [1]_.

    References
    ----------
    .. [1]  Montgomery, D. C. (2009).
            Statistical quality control (Vol. 7).
            New York: Wiley.
    """
    n = np.arange(2, 26)
    tab_result = np.array([1.128, 1.693, 2.059, 2.326, 2.534, 2.704,
                           2.847, 2.970, 3.078, 3.173, 3.258, 3.336,
                           3.407, 3.472, 3.532, 3.588, 3.640, 3.689,
                           3.735, 3.778, 3.819, 3.858, 3.895, 3.931,])
    for i in range(len(n)):
        assert util.d2_integral(n[i]) == pytest.approx(tab_result[i], abs=1e-3)

@pytest.mark.slow
def test_d3_integral():
    """
    Checks values against Appendix VI in [1]_.

    References
    ----------
    .. [1]  Montgomery, D. C. (2009).
            Statistical quality control (Vol. 7).
            New York: Wiley.
    """
    n = np.arange(2, 26)
    tab_result = np.array([0.853, 0.888, 0.880, 0.864, 0.848, 0.833,
                           0.820, 0.808, 0.797, 0.787, 0.778, 0.770,
                           0.763, 0.756, 0.750, 0.744, 0.739, 0.734,
                           0.729, 0.724, 0.720, 0.716, 0.712, 0.708,])
    for i in range(len(n)):
        assert util.d3_integral(n[i]) == pytest.approx(tab_result[i], abs=1e-3)

@pytest.mark.slow
def test_solve_h4():
    """
    Checks mGL20 values from Table 2 in [1]_.

    References
    ----------
    .. [1]  Knoth, S. (2017).
            ARL numerics for MEWMA charts.
            Journal of Quality Technology, 49(1), 78-89.
    """
    arl_0 = np.array([100, 200, 500, 1000, 2000, 5000])
    lmda = np.array([0.25, 0.20, 0.15, 0.10, 0.05])
    results = np.array([
        [10.47, 10.17,  9.71,  8.96,  7.44], # arl_0 = 100
        [12.13, 11.87, 11.46, 10.78,  9.37], # arl_0 = 200
        [14.26, 14.03, 13.68, 13.09, 11.83], # arl_0 = 500, etd
        [15.82, 15.62, 15.31, 14.76, 13.60],
        [17.36, 17.18, 16.89, 16.40, 15.31],
        [19.36, 19.20, 18.95, 18.50, 17.50],
    ])

    for i in range(len(arl_0)):
        for j in range(len(lmda)):
            res = util.solve_h4(arl_0=arl_0[i], p=3, lmda=lmda[j], init_h4=round(results[i, j]))[0]
            assert res == pytest.approx(results[i, j], abs=1e-2)

@pytest.mark.slow
def test_solve_arl():
    """
    Checks mGL20 values from Table 2 in [1]_.

    References
    ----------
    .. [1]  Knoth, S. (2017).
            ARL numerics for MEWMA charts.
            Journal of Quality Technology, 49(1), 78-89.
    """
    arl_0 = np.array([100, 200, 500, 1000, 2000, 5000])
    lmda = np.array([0.25, 0.20, 0.15, 0.10, 0.05])
    results = np.array([
        [10.47, 10.17,  9.71,  8.96,  7.44], # arl_0 = 100
        [12.13, 11.87, 11.46, 10.78,  9.37], # arl_0 = 200
        [14.26, 14.03, 13.68, 13.09, 11.83], # arl_0 = 500, etd
        [15.82, 15.62, 15.31, 14.76, 13.60],
        [17.36, 17.18, 16.89, 16.40, 15.31],
        [19.36, 19.20, 18.95, 18.50, 17.50],
    ])

    for i in range(len(arl_0)):
        for j in range(len(lmda)):
            res = util.solve_arl(h4=results[i, j], p=3, lmda=lmda[j])
            assert arl_0[i] == pytest.approx(res, 0.1)

def test_group_consecutives():
    util.group_consecutives([]) == []
    util.group_consecutives(np.array([1, 2, 3, 4, 5])) == [[1, 2, 3, 4, 5],]
    util.group_consecutives(np.array([1, 3, 5])) == [[1], [3], [5],]
    util.group_consecutives(np.array([1])) == [[1]]

    arg = np.array([1, 2, 4, 5, 6, 8, 10])
    exp = [
        [1, 2],
        [4, 5, 6],
        [8],
        [10],
    ]
    assert util.group_consecutives(arg) == exp

    arg = np.array([1, 2, 3, 5, 7, 8, 10])
    cons_fn = lambda a, b: b - a == 2
    exp = [
        [1],
        [2],
        [3, 5, 7],
        [8, 10]
    ]
    assert util.group_consecutives(arg, cons_fn) == exp

def test_alarm_subsets():
    arg = pd.Series([True, True, False, True, False, True, True, True])
    res = util.alarm_subsets(arg)
    exp = [
        arg[:2].index,
        arg[[3]].index,
        arg[5:].index,
    ]
    for r, e in zip(res, exp):
        assert list(r) == list(e)
