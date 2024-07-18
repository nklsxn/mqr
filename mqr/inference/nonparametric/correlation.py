"""
Hypothesis (non-parametric) tests for correlation.
"""

from mqr.inference.hyptest import HypothesisTest

def test(x, y, alternative='two-sided', method='spearman'):
    """
    Hypothesis test for correlation between two samples.

    Null-hypothesis: `corr(x, y) == 0`.

    Arguments
    ---------
    x, y (array[float]) -- Test correlation of these two equal-length samples.

    Optional
    --------
    alternative (str) -- Sense of alternative hypothesis. One of "two-sided",
        "less" or "greater". (Default "two-sided".)
    method (str) -- Type of statistic (default "spearman"):
        "spearman" (`scipy.stats.spearmanr`, scipy.org),
        "kendall" (`scipy.stats.kendalltau`, scipy.org).

    Returns
    -------
    mqr.hyptest.HypothesisTest
    """
    import numpy as np
    import scipy.stats as st
    import mqr

    if method == 'spearman':
        stat, pvalue = st.spearmanr(
            a=x,
            b=y,
            alternative=alternative)
    elif method == 'kendall':
        stat, pvalue = st.kendalltau(
            x=x,
            y=y,
            alternative=alternative)
    else:
        raise ValueError(f'method "{method}" is not available')

    x_name = x.name if hasattr(x, 'name') else 'x'
    y_name = y.name if hasattr(y, 'name') else 'y'
    return HypothesisTest(
        description='correlation coefficient',
        alternative=alternative,
        method=method,
        sample_stat=f'corr({x_name}, {y_name})',
        sample_stat_target=0.0,
        sample_stat_value=stat,
        stat=stat,
        pvalue=pvalue,)
