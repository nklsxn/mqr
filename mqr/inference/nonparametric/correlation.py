from mqr.inference.hyptest import HypothesisTest

import mqr.inference.lib.util as util

def test(x, y, alternative='two-sided', method='spearman'):
    """
    Hypothesis test for correlation between two samples.

    Null hypothesis
        corr(`x`, `y`) == 0

    Parameters
    ----------
    x, y : array_like
        Test correlation of these two equal-length samples.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Sense of alternative hypothesis.
    method : {'spearman', 'kendall'}, optional
        | 'spearman'
        |   Spearman correlation coefficient.
            Calls :func:`scipy..spearmanr <scipy.stats.spearmanr>`.
        | 'kendall'
        |   Kendall's tau measure.
            Calls :func:`scipy..kendalltau <scipy.stats.kendalltau>`.

    Returns
    -------
    :class:`mqr.inference.hyptest.HypothesisTest`
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
        raise ValueError(util.method_error_msg(method, ['spearman', 'kendall']))

    x_name = util.var_name(x, 'x')
    y_name = util.var_name(y, 'y')
    return HypothesisTest(
        description='correlation coefficient',
        alternative=alternative,
        method=method,
        sample_stat=f'corr({x_name}, {y_name})',
        sample_stat_target=0.0,
        sample_stat_value=stat,
        stat=stat,
        pvalue=pvalue,)
