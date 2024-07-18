"""
Confidence intervals and hypothesis tests (non-parametric) for quantiles.
"""

from mqr.inference.confint import ConfidenceInterval
from mqr.inference.hyptest import HypothesisTest

import mqr.interop.inference as interop

import numpy as np
import scipy

def confint_1sample(x, q=0.5, conf=0.95, bounded='both'):
    """
    Confidence interval for the quantile of a sample.
    
    Calls `scipy.stats.quantile_test` (scipy.org).

    Arguments
    ---------
    x (array[float]) -- Calculate the confidence interval for the quantile of this sample.

    Optional
    --------
    q (float) -- Calculate the interval around this quantile. (Default 0.5.)
    conf (float) -- Confidence level that determines the width of the interval.
        (Default 0.95.)
    bounded (str) -- Which sides of the interval to close. (Default "both".)

    Returns
    -------
    mqr.confint.ConfidenceInterval
    """
    value = np.quantile(x, q)
    alt = interop.bounded(bounded, 'scipy', flip=True)
    res = scipy.stats.quantile_test(x, q=np.quantile(x, q), p=q, alternative=alt)
    ci = res.confidence_interval(conf)
    return ConfidenceInterval(
        name='quantile',
        value=value,
        lower=ci.low,
        upper=ci.high,
        conf=conf)

def test_1sample(x, H0_quant=None, q=0.5, alternative='two-sided'):
    """
    Hypothesis test for the quantile of a sample.

    Null-hypothesis: `quantile(x, q) == H0_quant`.

    Calls `scipy.stats.quantile_test` (scipy.org).

    Arguments
    ---------
    x (array[float]) -- Test quantile of this sample.

    Optional
    --------
    H0_quant (float) -- Null-hypothesis value associated with `q`. (Default None.)
    q (float) -- Test the value associated with this quantile. (Default 0.5.)
    alternative (str) -- Sense of alternative hypothesis. One of "two-sided",
        "less" or "greater". (Default "two-sided".)

    Returns
    -------
    mqr.hyptest.HypothesisTest
    """
    if H0_quant is None:
        H0_quant = np.quantile(x, q)

    res = scipy.stats.quantile_test(x, q=H0_quant, p=q, alternative=alternative)

    x_name = x.name if hasattr(x, 'name') else 'x'
    return HypothesisTest(
        description='quantile',
        alternative=alternative,
        method='binom',
        sample_stat=f'quantile({x_name}, {q})',
        sample_stat_target=H0_quant,
        sample_stat_value=np.quantile(x, q),
        stat=res.statistic,
        pvalue=res.pvalue,)
