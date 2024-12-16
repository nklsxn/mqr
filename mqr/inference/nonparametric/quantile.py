from mqr.inference.confint import ConfidenceInterval
from mqr.inference.hyptest import HypothesisTest

from mqr.inference.lib import util
import mqr.interop.inference as interop
import mqr.utils

import numpy as np
import scipy

def confint_1sample(x, q=0.5, conf=0.95, bounded='both'):
    """
    Confidence interval for the quantile of a sample.
    
    Calls :func:`scipy <scipy.stats.quantile_test>`.

    Parameters
    ----------
    x : array_like
        Calculate the confidence interval for the quantile of this sample.
    q : float, optional
        Calculate the interval around this quantile.
    conf : float, optional
        Confidence level that determines the width of the interval.
    bounded : {'both', 'below', 'above'}, optional
        Which sides of the interval to close.

    Returns
    -------
    :class:`mqr.inference.confint.ConfidenceInterval`
    """
    value = np.quantile(x, q)
    alt = interop.bounded(bounded, 'scipy')
    res = scipy.stats.quantile_test(x, q=np.quantile(x, q), p=q, alternative=alt)
    ci = res.confidence_interval(conf)
    percentile = mqr.utils.make_ordinal(100*q)
    return ConfidenceInterval(
        name=f'quantile ({percentile} percentile)',
        method='binom',
        value=value,
        lower=ci.low,
        upper=ci.high,
        conf=conf,
        bounded=bounded)

def test_1sample(x, H0_quant=None, q=0.5, alternative='two-sided'):
    """
    Hypothesis test for the quantile of a sample.

    Null-hypothesis
        quantile(`x`, `q`) == `H0_quant`.

    Calls :func:`scipy..quantile_test <scipy.stats.quantile_test>`.

    Parameters
    ----------
    x : array_like
        Test quantile of this sample.
    H0_quant : float, optional
        Null-hypothesis value associated with ``q``. Default is the ``q`` th
        quantile of ``x``.
    q : float, optional
        Test the value associated with this quantile.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Sense of alternative hypothesis.

    Returns
    -------
    :class:`mqr.inference.hyptest.HypothesisTest`
    """
    if H0_quant is None:
        H0_quant = np.quantile(x, q)

    res = scipy.stats.quantile_test(x, q=H0_quant, p=q, alternative=alternative)

    x_name = util.var_name(x, 'x')
    return HypothesisTest(
        description='quantile',
        alternative=alternative,
        method='binom',
        sample_stat=f'quantile({x_name}, {q})',
        sample_stat_target=H0_quant,
        sample_stat_value=np.quantile(x, q),
        stat=res.statistic,
        pvalue=res.pvalue,)
