import mqr

import numpy as np
import pandas as pd
import pytest


################################################################################
## Fixtures

@pytest.fixture
def sample1():
    data = np.array([
        [1, 2, 3, 1, 2, 3],
        [2, 4, 6, 6, 4, 2],
        [1, 2, 0, 2, 0, 1],
    ])
    return pd.DataFrame(
        data,
        index=range(1, 4),
        columns=[f'x{i}' for i in range(1, 7)])

@pytest.fixture
def sample2():
    data = np.array([
        [1, 2],
        [2, 4],
        [3, 6],
    ])
    return pd.DataFrame(
        data,
        index=range(1, 4),
        columns=[f'x{i}' for i in range(1, 3)])

@pytest.fixture
def sample3():
    data = np.array([
        [0, 2],
        [0, 2],
        [3, 5],
        [3, 5],
        [3, 5],
        [3, 5],
        [3, 5],
        [3, 5],
    ])
    m, n = data.shape
    return pd.DataFrame(
        data,
        index=range(1, m+1),
        columns=[f'' for i in range(1, n+1)])

@pytest.fixture
def sample4():
    data = np.array([
        [1, 5, 5],
        [1, 5, 5],
        [1, 10, 6],
        [1, 10, 5],
        [1, 5, 5],
        [1, 5, 5],
        [4, 10, 6],
        [1, 10, 5],
        [1, 5, 5],
        [1, 5, 5],
    ])
    m, n = data.shape
    return pd.DataFrame(
        data,
        index=range(1, m+1),
        columns=[f'x{i}' for i in range(1, n+1)])

# Same as sample4 but index starts at 5
@pytest.fixture
def sample5():
    data = np.array([
        [1, 5, 5],
        [1, 5, 5],
        [1, 10, 6],
        [1, 10, 5],
        [1, 5, 5],
        [1, 5, 5],
        [4, 10, 6],
        [1, 10, 5],
        [1, 5, 5],
        [1, 5, 5],
    ])
    m, n = data.shape
    return pd.DataFrame(
        data,
        index=range(5, m+5),
        columns=[f'x{i}' for i in range(1, n+1)])

################################################################################
## Tests

def test_ControlStatistic_post_init():
    stat = pd.Series([1, 2, 3, 4, 5])
    nobs = pd.Series([4, 4, 4, 4])
    with pytest.raises(ValueError):
        mqr.spc.ControlStatistic(stat, nobs)


## XBarParams

def test_XBarParams_statistic(sample1):
    params = mqr.spc.XBarParams(0, 1)
    stat = params.statistic(sample1)

    assert list(stat.stat) == [2, 4, 1]
    assert list(stat.nobs) == [6, 6, 6]

def test_XBarParams_se():
    params = mqr.spc.XBarParams(0, 4)
    assert params.se(16) == 1

def test_XBarParams_target():
    params = mqr.spc.XBarParams(3, 1)
    assert params.target() == 3

def test_XBarParams_lcl():
    params = mqr.spc.XBarParams(7, 6, 5)
    assert params.lcl(9) == 7 - (6 / 3) * 5

def test_XBarParams_ucl():
    params = mqr.spc.XBarParams(7, 6, 5)
    assert params.ucl(9) == 7 + (6 / 3) * 5

def test_XBarParams_from_stddev():
    params = mqr.spc.XBarParams.from_stddev(11, 7, 5, 13)
    assert params.centre == 11
    assert params.sigma == 7 / mqr.spc.util.c4(5)
    assert params.nsigma == 13

def test_XBarParams_from_range():
    params = mqr.spc.XBarParams.from_range(11, 7, 5, 13)
    assert params.centre == 11
    assert params.sigma == 7 / mqr.spc.util.d2(5)
    assert params.nsigma == 13

def test_XBarParams_from_data(sample1, sample2):
    params_sbar = mqr.spc.XBarParams.from_data(sample2, 's_bar', 5)
    params_rbar = mqr.spc.XBarParams.from_data(sample2, 'r_bar', 5)

    assert params_sbar.centre == 3
    assert params_rbar.centre == 3

    assert params_sbar.sigma == np.sqrt(2) / mqr.spc.util.c4(2)
    assert params_rbar.sigma == 2 / mqr.spc.util.d2(2)

    assert params_sbar.nsigma == 5
    assert params_rbar.nsigma == 5

    params_sbar = mqr.spc.XBarParams.from_data(sample1, 's_bar', 5)
    params_rbar = mqr.spc.XBarParams.from_data(sample1, 'r_bar', 5)

    assert params_sbar.centre == 7 / 3
    assert params_rbar.centre == 7 / 3

    assert params_sbar.sigma == np.mean(np.std(sample1, ddof=1, axis=1)) / mqr.spc.util.c4(6)
    assert params_rbar.sigma == np.mean(np.ptp(sample1, axis=1)) / mqr.spc.util.d2(6)

    assert params_sbar.nsigma == 5
    assert params_rbar.nsigma == 5


## RParams

def test_RParams_statistic(sample1):
    params = mqr.spc.RParams(3, 5, 7)
    stat = params.statistic(sample1)
    assert list(stat.stat) == [2, 4, 2]
    assert list(stat.nobs) == [6, 6, 6]

def test_RParams_se():
    params = mqr.spc.RParams(3, 5, 7)
    assert params.se(13) == mqr.spc.util.d3(13) * 5

def test_RParams_target():
    params = mqr.spc.RParams(3, 5, 7)
    assert params.target() == 3

def test_RParams_lcl():
    params = mqr.spc.RParams(11, 7, 1)
    assert params.lcl(13) == 11 - mqr.spc.util.d3(13) * 7

    params = mqr.spc.RParams(11, 7, 5)
    assert params.lcl(13) == 0

def test_RParams_ucl():
    params = mqr.spc.RParams(11, 7, 5)
    assert params.ucl(13) == pytest.approx(11 + 5 * mqr.spc.util.d3(13) * 7, 1e-15)

def test_RParams_from_range():
    params = mqr.spc.RParams.from_range(3, 5, 7)
    assert params.centre == 3
    assert params.sigma == 3 / mqr.spc.util.d2(5)
    assert params.nsigma == 7

def test_RParams_from_data(sample1):
    params = mqr.spc.RParams.from_data(sample1, 5)
    assert params.centre == np.mean(np.ptp(sample1, axis=1))
    assert params.sigma == params.centre / mqr.spc.util.d2(6)

## SParams

def test_SParams_statistic(sample1):
    params = mqr.spc.SParams(3, 5, 7)
    stat = params.statistic(sample1)
    assert list(stat.stat) == list(np.std(sample1, ddof=1, axis=1))
    assert list(stat.nobs) == [6, 6, 6]

def test_SParams_se():
    params = mqr.spc.SParams(3, 5, 7)
    c4 = mqr.spc.util.c4(6)
    assert params.se(6) == 3 * np.sqrt(1 - c4**2) / c4

def test_SParams_target():
    params = mqr.spc.SParams(3, 5, 7)
    assert params.target() == 3

def test_SParams_lcl():
    params = mqr.spc.SParams(3, 1)
    c4 = mqr.spc.util.c4(6)
    assert params.lcl(6) == 3 - 3 * np.sqrt(1 - c4**2) / c4

    params = mqr.spc.SParams(3, 5)
    assert params.lcl(6) == 0

def test_SParams_ucl():
    params = mqr.spc.SParams(3, 5)
    c4 = mqr.spc.util.c4(6)
    assert params.ucl(6) == 3 + 3 * 5 * np.sqrt(1 - c4**2) / c4

def test_SParams_from_data(sample1):
    params = mqr.spc.SParams.from_data(sample1, 5)
    assert params.centre == np.mean(np.std(sample1, ddof=1, axis=1))
    assert params.nsigma == 5

## EwmaParams

def test_EwmaParams_statistic(sample3):
    params = mqr.spc.EwmaParams(1, 1, 0.2, 3)
    # The strategy prepends the target as the first value, which is 1 here.
    stat = params.statistic(sample3.iloc[1:])
    # The 1 in the test sample is used as the initial value in the series,
    # then drop the first value (the initialisation value) because only the
    # sample values are returned by the strategy.
    assert list(stat.stat) == list(sample3.mean(axis=1).ewm(alpha=0.2, adjust=False).mean().iloc[1:])
    assert list(stat.nobs) == [2, 2, 2, 2, 2, 2, 2]

def test_EwmaParams_target():
    params = mqr.spc.EwmaParams(5, 1, 0.2, 3)
    assert params.target() == 5

def test_EwmaParams_lcl():
    """
    Check against equations in Chapter 9 of [1]_.

    References
    ----------
    .. [1]  Montgomery, D. C. (2009).
            Statistical quality control (Vol. 7).
            New York: Wiley.
    """
    i = np.array([3, 4, 5, 6])
    nobs = pd.Series([9, 9, 9, 9], index=i)
    frac = 0.2 / (2 - 0.2)
    decay = (1 - (1 - 0.2)**(2 * i))

    params = mqr.spc.EwmaParams(1, 6, 0.2, 5, False)
    assert list(params.lcl(nobs)) == list(1 - 5 * 6/3 * np.sqrt(frac * decay))

    params = mqr.spc.EwmaParams(1, 6, 0.2, 5, True)
    assert all(params.lcl(nobs) == 1 - 5 * 6/3 * np.sqrt(frac))

def test_EwmaParams_ucl():
    """
    Check against equations in Chapter 9 of [1]_.

    References
    ----------
    .. [1]  Montgomery, D. C. (2009).
            Statistical quality control (Vol. 7).
            New York: Wiley.
    """
    i = np.array([3, 4, 5, 6])
    nobs = pd.Series([9, 9, 9, 9], index=i)
    frac = 0.2 / (2 - 0.2)
    decay = (1 - (1 - 0.2)**(2 * i))

    params = mqr.spc.EwmaParams(1, 6, 0.2, 5, False)
    assert list(params.ucl(nobs)) == list(1 + 5 * 6/3 * np.sqrt(frac * decay))

    params = mqr.spc.EwmaParams(1, 6, 0.2, 5, True)
    assert all(params.ucl(nobs) == 1 + 5 * 6/3 * np.sqrt(frac))

def test_EwmaParams_from_stddev():
    params = mqr.spc.EwmaParams.from_stddev(1.2, 3.4, 5, 0.6, 7, False)
    assert params.mu_0 == 1.2
    assert params.sigma == 3.4 / mqr.spc.util.c4(5)
    assert params.lmda == 0.6
    assert params.L == 7
    assert not params.steady_state

    params = mqr.spc.EwmaParams.from_stddev(1.2, 3.4, 5, 0.6, 7, True)
    assert params.mu_0 == 1.2
    assert params.sigma == 3.4 / mqr.spc.util.c4(5)
    assert params.lmda == 0.6
    assert params.L == 7
    assert params.steady_state

def test_EwmaParams_from_data(sample1):
    params = mqr.spc.EwmaParams.from_data(sample1, 0.3, 4, True)
    target = np.mean(sample1)
    sigma = np.mean(np.std(sample1, ddof=1, axis=1)) / mqr.spc.util.c4(6)
    assert params.mu_0 == target
    assert params.sigma == sigma
    assert params.lmda == 0.3
    assert params.L == 4
    assert params.steady_state

    params = mqr.spc.EwmaParams.from_data(sample1, 0.3, 4, False)
    target = np.mean(sample1)
    sigma = np.mean(np.std(sample1, ddof=1, axis=1)) / mqr.spc.util.c4(6)
    assert params.mu_0 == target
    assert params.sigma == sigma
    assert params.lmda == 0.3
    assert params.L == 4
    assert not params.steady_state

## MewmaParams

def test_MewmaParams_statistic(sample5):
    """
    Check against manual construction of EWMA and statistic
    """
    sample = sample5
    idx = sample.index
    lmda = 0.2
    params = mqr.spc.MewmaParams.from_data(sample, np.nan, lmda)
    stat = params.statistic(sample)

    z = pd.DataFrame(index=sample.index, columns=sample.columns)
    z.loc[idx[0]] = lmda * (sample.loc[idx[0]] - params.mu)
    for i in idx[1:]:
        z.loc[i] = lmda * (sample.loc[i] - params.mu) + (1 - lmda) * z.loc[i-1]

    t2 = pd.Series(index=sample.index)
    for i in idx:
        cov_scaled = lmda / (2 - lmda) * params.cov  * (1 - (1 - lmda)**(2*i))
        t2.loc[i] = z.loc[i] @ np.linalg.inv(cov_scaled) @ z.loc[i]

    assert list(stat.stat.index) == list(sample.index)
    assert list(stat.nobs.index) == list(sample.index)
    assert np.all(np.isclose(stat.stat, t2))
    assert all(stat.nobs == 3)

def test_MewmaParams_target(sample4):
    params = mqr.spc.MewmaParams.from_data(sample4, 3.45, 0.5)
    assert params.target() == 0

def test_MewmaParams_lcl(sample4):
    params = mqr.spc.MewmaParams.from_data(sample4, 3.45, 0.2)
    assert params.lcl(sample4) == None

def test_MewmaParams_ucl(sample4):
    params = mqr.spc.MewmaParams.from_data(sample4, 3.45, 0.2)
    ucl = params.ucl(sample4)
    assert list(ucl.index) == list(sample4.index)
    assert all(ucl == 3.45)

def test_MewmaParams_from_data(sample4):
    params = mqr.spc.MewmaParams.from_data(sample4, 3.45, 0.2)
    assert list(params.mu) == list(np.mean(sample4, axis=0))
    assert np.all(np.isclose(params.cov, np.cov(sample4.T, ddof=1)))
    assert params.lmda == 0.2
    assert params.limit == 3.45
