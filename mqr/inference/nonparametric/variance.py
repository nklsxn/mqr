"""
Hypothesis tests (non-parametric) for the variance.
"""

from mqr.inference.hyptest import HypothesisTest

import scipy

def test_nsample(*x, alternative='two-sided', method='levene'):
    """
    Hypothesis test for homogeneity of variances of multiple samples.

    Null hypothesis: `var(x_i) == var(x_j)` for all `i`, `j`.

    Arguments
    ---------
    x (list[array[float]]) -- Test equality of variances of these samples.

    Optional
    --------
    alternative (str) -- Sense of alternative hypothesis. One of "two-sided",
        "less" or "greater". (Default "two-sided".)
    method (str) -- Type of test (default "levene"):
        "levene" (`scipy.stats.levene`, scipy.org),
        "fligner-killeen" (`scipy.stats.fligner`, scipy.org).

    Returns
    -------
    mqr.hyptest.HypothesisTest
    """
    if method == 'levene':
        if alternative != 'two-sided':
            raise ValueError('one-sided alternative not available in Levene test')
        (stat, pvalue) = scipy.stats.levene(*x)
    elif method == 'fligner-killeen':
        if alternative != 'two-sided':
            raise ValueError('one-sided alternative not available in Fligner-Killeen test')
        (stat, pvalue) = scipy.stats.fligner(*x)
    else:
        raise ValueError(f'method "{method}" not available')

    return HypothesisTest(
        description='equality of variances',
        alternative=alternative,
        method=method,
        sample_stat='var(x_i)',
        sample_stat_target='var(x_j)',
        sample_stat_value=stat,
        stat=stat,
        pvalue=pvalue,)
