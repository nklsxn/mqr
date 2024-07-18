"""
Hypothesis tests (non-parametric) for the median.
"""

from mqr.inference.hyptest import HypothesisTest

def test_1sample(x, H0_median=0.0, alternative='two-sided', method='sign'):
    """
    Hypothesis test for the median of a sample.

    Null-hypothesis: `median(x) == H0_median`.

    Arguments
    ---------
    x (array[float]) -- Test the median of this sample.

    Optional
    --------
    H0_median (float) -- Null-hypothesis median. (Default 0.0.)
    alternative (str) -- Sense of alternative hypothesis. One of "two-sided",
        "less" or "greater". (Default "two-sided".)
    method (str) -- Type of test (default "sign"):
        "sign" (`statsmodels.stats.descriptivestats.sign_test`, scipy.org) or
        "wilcoxon" (`statsmodels.stats.descriptivestats.sign_test`, statsmodels.org).

    Returns
    -------
    mqr.hyptest.HypothesisTest
    """
    import mqr, statsmodels
    import numpy as np, scipy.stats as st

    if method == 'sign':
        if alternative != 'two-sided':
            raise ValueError(f'invalid alternative "{alternative}"')
        stat, pvalue = statsmodels.stats.descriptivestats.sign_test(x, mu0=H0_median)
    elif method == 'wilcoxon':
        stat, pvalue = st.wilcoxon(x-H0_median, alternative=alternative)
    else:
        raise ValueError(f'method "{method}" not available')

    x_name = x.name if hasattr(x, 'name') else 'x'
    return HypothesisTest(
        description='median',
        alternative=alternative,
        method=method,
        sample_stat=f'median({x_name})',
        sample_stat_target=H0_median,
        sample_stat_value=np.median(x),
        stat=stat,
        pvalue=pvalue,
    )

def test_nsample(*x, alternative='two-sided', method='kruskal-wallace'):
    """
    Hypothesis test for equality of medians amongst samples.

    Null-hypothesis: `median(x_i) == median(x_j)` for all `i`, `j`.

    Arguments
    ---------
    x (list[array[float]]) -- Test the medians of these samples.

    Optional
    --------
    alternative (str) -- Sense of alternative hypothesis. One of "two-sided",
        "less" or "greater". (Default "two-sided".)
    method (str) -- Type of test (default "kruskal-wallace"):
        "kruskal-wallace" (`scipy.stats.kruskal`, scipy.org),
        "mann-whitney" (`scipy.stats.mannwhitneyu`, scipy.org).

    Returns
    -------
    mqr.hyptest.HypothesisTest
    """
    import mqr, numpy as np, scipy.stats as st

    if method == 'kruskal-wallace':
        if alternative != 'two-sided':
            raise ValueError(f'invalid alternative "{alternative}"')
        sample_stat = 'median(x_i)'
        sample_stat_value = 'median(x_j)'
        stat, pvalue = st.kruskal(*x)
    elif method == 'mann-whitney':
        n = len(x)
        if n != 2:
            raise ValueError(f'Mann-Whitney test cannot be applied to {n} samples (compare exactly 2)')
        sample_stat = 'median(x) - median(y)'
        sample_stat_value = 0.0
        stat, pvalue = st.mannwhitneyu(x[0], x[1], alternative=alternative)
    else:
        raise NotImplementedError(f'method {method} not available')

    return HypothesisTest(
        description='equality of medians',
        alternative=alternative,
        method=method,
        sample_stat=sample_stat,
        sample_stat_target=sample_stat_value,
        sample_stat_value=np.nan,
        stat=stat,
        pvalue=pvalue,)
