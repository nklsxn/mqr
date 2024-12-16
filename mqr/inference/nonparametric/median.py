from mqr.inference.hyptest import HypothesisTest
import mqr.inference.lib.util as util

def test_1sample(x, H0_median=0.0, alternative='two-sided', method='sign'):
    """
    Hypothesis test for the median of a sample.

    Null-hypothesis
        median(`x`) == H0_median

    Parameters
    ----------
    x : array_like
        Test the median of this sample.
    H0_median : float, optional
        Null-hypothesis median.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Sense of alternative hypothesis.
    method : {'sign', 'wilcoxon'}, optional
        | 'sign'
        |   Sign test.
            Calls :func:`sm..sign_test <statsmodels.stats.descriptivestats.sign_test>`.
        | 'wilcoxon'
        |   Wilcoxon signed-rank test.
            Calls :func:`scipy..wilcoxon <scipy.stats.wilcoxon>`.

    Returns
    -------
    :class:`mqr.inference.hyptest.HypothesisTest`
    """
    import mqr, statsmodels
    import numpy as np, scipy.stats as st

    if method == 'sign':
        if alternative != 'two-sided':
            raise ValueError(util.alternative_error_msg(alternative))
        stat, pvalue = statsmodels.stats.descriptivestats.sign_test(x, mu0=H0_median)
    elif method == 'wilcoxon':
        stat, pvalue = st.wilcoxon(x-H0_median, alternative=alternative)
    else:
        raise ValueError(util.method_error_msg(method, ['sign', 'wilcoxon']))

    x_name = util.var_name(x, 'x')
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

def test_nsample(*x, alternative='two-sided', method='kruskal-wallis'):
    """
    Hypothesis test for equality of medians amongst samples.

    Null-hypothesis
        median(`x[i]`) == median(`x[j]`), for all `i`, `j`

    Parameters
    ----------
    x : array_like
        Test the medians of these samples.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Sense of alternative hypothesis.
    method : {'kruskal-wallis', 'mann-whitney'}, optional
        | 'kruskal-wallis'
        |   Kruskal-Wallis test. A one-way ANOVA based on ranks.
            Calls :func:`scipy..kruskal <scipy.stats.kruskal>`.
        | 'mann-whitney'
        |   Mann-Whitney U test for the differences between two samples.
            Calls :func:`scipy..mannwhitneyu <scipy.stats.mannwhitneyu>`.

    Returns
    -------
    :class:`mqr.inference.hyptest.HypothesisTest`
    """
    import mqr, numpy as np, scipy.stats as st

    if method == 'kruskal-wallis':
        if alternative != 'two-sided':
            raise ValueError(util.alternative_error_msg(alternative))
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
        raise ValueError(util.method_error_msg(method, ['kruskal-wallis', 'mann-whitney']))

    return HypothesisTest(
        description='equality of medians',
        alternative=alternative,
        method=method,
        sample_stat=sample_stat,
        sample_stat_target=sample_stat_value,
        sample_stat_value=np.nan,
        stat=stat,
        pvalue=pvalue,)
