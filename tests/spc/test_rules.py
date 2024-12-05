import mqr

import numpy as np
import pandas as pd

def test_combine():
    mean = 5
    sigma = 1
    zscore = np.array([[0, 3.5, 0, 0, 0, 2.5, 1, 2.5, 2.5, 0]]).T
    data = pd.DataFrame(mean + zscore * sigma)
    params = mqr.spc.XBarParams(centre=mean, sigma=sigma)
    stat = params.statistic(data)

    rule = mqr.spc.rules.combine(
        np.logical_or,
        mqr.spc.rules.limits(),
        mqr.spc.rules.aofb_nsigma(a=3, b=4, n=2))
    alarms = rule(stat, params)

    assert list(alarms.index) == list(stat.stat.index)
    assert list(alarms) == [
        False,
        True, # Larger than UCL => triggers `rules.limits`
        False,
        False,
        False,
        False,
        False,
        False,
        True, # 3 of the 4 points to here (incl) are greater than 2-sigma => `rules.aofb_nsigma`
        False,
    ]

def test_limits():
    mean = 5
    sigma = 1
    zscore = np.array([[0, 3.5, 0, 0, 0, 2.5, 1, 2.5, 2.5, 0]]).T
    data = pd.DataFrame(mean + zscore * sigma)
    params = mqr.spc.XBarParams(centre=mean, sigma=sigma)
    stat = params.statistic(data)

    rule = mqr.spc.rules.limits()
    alarms = rule(stat, params)

    assert list(alarms.index) == list(stat.stat.index)
    assert list(alarms) == [
        False,
        True, # Larger than UCL => triggers `rules.limits`
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]

def test_aofb_nsigma():
    mean = 5
    sigma = 1
    zscore = np.array([[0, 3.5, 0, 0, 0, 2.5, 1, 2.5, 2.5, 0]]).T
    data = pd.DataFrame(mean + zscore * sigma)
    params = mqr.spc.XBarParams(centre=mean, sigma=sigma)
    stat = params.statistic(data)

    rule = mqr.spc.rules.aofb_nsigma(a=3, b=4, n=2)
    alarms = rule(stat, params)

    assert list(alarms.index) == list(stat.stat.index)
    assert list(alarms) == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True, # 3 of the 4 points to here (incl) are greater than 2-sigma => `rules.aofb_nsigma`
        False,
    ]

def test_n1side():
    mean = 5
    sigma = 1
    zscore = np.array([[0, 3.5, 0, 0, 0, -2.5, -1, 1, 2.5, -2.5, 0]]).T
    data = pd.DataFrame(mean + zscore * sigma)
    params = mqr.spc.XBarParams(centre=mean, sigma=sigma)
    stat = params.statistic(data)

    rule = mqr.spc.rules.n_1side(n=2)
    alarms = rule(stat, params)

    assert list(alarms.index) == list(stat.stat.index)
    assert list(alarms) == [
        False,
        False,
        False,
        False,
        False,
        False,
        True, # -2.5 and -1 are two on the same side => rules.n_1side
        False,
        True, # 1 and 2.5 are two on the same side => rules.n_1side
        False,
        False,
    ]

def test_ntrending():
    mean = 5
    sigma = 1
    zscore = np.array([[0, 3.5, 0, 0, 0, -2.5, -1, 1, 2.5, -2.5, 0]]).T
    data = pd.DataFrame(mean + zscore * sigma)
    params = mqr.spc.XBarParams(centre=mean, sigma=sigma)
    stat = params.statistic(data)

    rule = mqr.spc.rules.n_trending(n=4)
    alarms = rule(stat, params)

    assert list(alarms.index) == list(stat.stat.index)
    assert list(alarms) == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True, # -2.5, -1, 1 and 2.5 are trending => rules.n_trending
        False,
        False,
    ]
