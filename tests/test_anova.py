import mqr

import numpy as np
import pandas as pd
import pytest
import scipy
from statsmodels.formula.api import ols

def test_summary():
    """
    Checking properties calculated for the table, but not the properties of the
    regression provided by statsmodels.
    """
    np.random.seed(0)
    data = pd.DataFrame({
        'x': scipy.stats.norm().rvs(20),
        'y': scipy.stats.norm().rvs(20),
        'z': scipy.stats.norm().rvs(20),
    })
    mod = ols('z ~ x + y', data)
    res = mod.fit()

    summary = mqr.anova.summary(res, formatted=False)

    assert (
        summary.loc['Total', 'df'] ==
        summary.loc['Residual', 'df'] + summary.loc['y', 'df'] + summary.loc['y', 'df']
    )
    assert (
        summary.loc['Total', 'sum_sq'] ==
        summary.loc['Residual', 'sum_sq'] + summary.loc['x', 'sum_sq'] + summary.loc['y', 'sum_sq']
    )
    assert list(summary['sum_sq']) == list(summary['mean_sq'] * summary['df'])

def test_groups():
    # Two blocks of zero-mean random noise for the groups
    np.random.seed(0)
    xs = scipy.stats.norm().rvs(20)
    xs[:10] -= np.mean(xs[:10])
    xs[10:] -= np.mean(xs[10:])

    # Measurements, with the second groups mean higher than the first
    ys = xs.copy()
    ys[:10] += 5

    data = pd.DataFrame({
        'a': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'b': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        'x': ys,
    })

    mod = ols('x ~ C(a) + C(b)', data)
    res = mod.fit()

    groups = mqr.anova.groups(res, value='x', factor='a', conf=0.99, formatted=False)
    assert np.all(np.isclose(
        (groups['upper'] - groups['lower']) / 2,
        scipy.stats.t(18).ppf(0.995) * np.std(xs, ddof=2) / np.sqrt(10)))

def test_interactions():
    data = pd.DataFrame({
        'a': [0, 0, 0, 0, 1, 1, 1, 1],
        'b': [0, 0, 1, 1, 0, 0, 1, 1],
        'c': [0, 1, 0, 1, 0, 1, 0, 1],
        'z': [1.0, 1.2, 1.4, 1.8, 0.75, 0.85, 1.0, 1.6],
    })
    intn = mqr.anova.interactions(data, value='z', between=['a', 'b'], formatted=False)

    assert list(intn.values.flatten()) == [1.1, 1.6, 0.8, 1.3]

def test_adequacy():
    """
    The adequacy table presents data from statsmodels routines.
    Here, checking only that the correct values are passed through.
    """
    data = pd.DataFrame({
        'peas': [1, 1, 0, 0, 0, 0, 0],
        'lupins': [0, 0, 1, 1, 1, 0, 0],
        'mustard': [0, 0, 0, 0, 0, 1, 1],
        'matter': [3.0, 3.6, 3.4, 3.8, 3.3, 4.4, 4.0],
    })
    mod = ols('matter ~ C(peas) + C(lupins) + C(mustard)', data)
    res = mod.fit()
    res = mod.fit()

    adeq = mqr.anova.adequacy(res, formatted=False)
    adeq.loc['', 'R-sq'] == pytest.approx(res.rsquared)
    adeq.loc['', 'R-sq (adj)'] == pytest.approx(res.rsquared_adj)
    adeq.loc['', 'F'] == pytest.approx(res.fvalue)
    adeq.loc['', 'PR(>F)'] == pytest.approx(res.f_pvalue)
    adeq.loc['', 'AIC'] == pytest.approx(res.aic)
    adeq.loc['', 'BIC'] == pytest.approx(res.bic)
