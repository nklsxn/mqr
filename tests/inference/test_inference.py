'''
Check result types and utils.
'''

import numbers
import numpy as np
import pytest

import mqr

def test_alternative():
    assert mqr.interop.inference.alternative('two-sided', 'statsmodels') == 'two-sided'
    assert mqr.interop.inference.alternative('less', 'statsmodels') == 'smaller'
    assert mqr.interop.inference.alternative('greater', 'statsmodels') == 'larger'

def test_bounded():
    assert mqr.interop.inference.bounded('both', 'scipy') == 'two-sided'
    assert mqr.interop.inference.bounded('below', 'scipy') == 'greater'
    assert mqr.interop.inference.bounded('above', 'scipy') == 'less'

    assert mqr.interop.inference.bounded('both', 'statsmodels') == 'two-sided'
    assert mqr.interop.inference.bounded('below', 'statsmodels') == 'larger'
    assert mqr.interop.inference.bounded('above', 'statsmodels') == 'smaller'

def test_confint_result():
    conf = mqr.inference.confint.ConfidenceInterval(
        name='',
        method='',
        value=np.nan,
        lower=-1.234,
        upper=1.234,
        conf=np.nan,
        bounded='both')

    (lower, upper) = conf
    assert lower == -1.234
    assert upper == 1.234

def test_hyptest_result():
    description = 'description-str'
    method = 'method-str'
    sample_stat = 'sample_stat-str'
    sample_stat_target = 'sample_stat_target-str'
    sample_stat_value = 1.234
    stat = 12.34
    pvalue = 0.1234

    alternative = 'two-sided'
    res = mqr.inference.hyptest.HypothesisTest(
        description=description,
        alternative=alternative,
        method=method,
        sample_stat=sample_stat,
        sample_stat_target=sample_stat_target,
        sample_stat_value=sample_stat_value,
        stat=stat,
        pvalue=pvalue)
    assert res.null == 'sample_stat-str == sample_stat_target-str'
    assert res.alt == 'sample_stat-str != sample_stat_target-str'

    alternative = 'less'
    res = mqr.inference.hyptest.HypothesisTest(
        description=description,
        alternative=alternative,
        method=method,
        sample_stat=sample_stat,
        sample_stat_target=sample_stat_target,
        sample_stat_value=sample_stat_value,
        stat=stat,
        pvalue=pvalue)
    assert res.null == 'sample_stat-str == sample_stat_target-str'
    assert res.alt == 'sample_stat-str < sample_stat_target-str'

    alternative = 'greater'
    res = mqr.inference.hyptest.HypothesisTest(
        description=description,
        alternative=alternative,
        method=method,
        sample_stat=sample_stat,
        sample_stat_target=sample_stat_target,
        sample_stat_value=sample_stat_value,
        stat=stat,
        pvalue=pvalue)
    assert res.null == 'sample_stat-str == sample_stat_target-str'
    assert res.alt == 'sample_stat-str > sample_stat_target-str'
