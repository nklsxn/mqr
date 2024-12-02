from mqr.inference.hyptest import HypothesisTest

import mqr.inference.lib.util as util

import scipy

def test_nsample(*x, alternative='two-sided', method='levene'):
    """
    Hypothesis test for homogeneity of variances of multiple samples.

    Null hypothesis
        var(`x[i]`) == var(`x[j]`), for all `i`, `j`

    Parameters
    ----------
    x : array_like
        Test equality of variances of these samples.
    alternative : {'two-sided'}, optional
        Sense of alternative hypothesis. Only 'two-sided' is currently supported.
    method : {'levene', 'fligner-killeen'}, optional
        | 'levene'
        |   Levene's test for homogeneity of variance.
            Calls :func:`scipy..levene <scipy.stats.levene>`.
        | 'fligner-killeen'
        |   Fligner-Killeen test for homogeneity of variance.
            Calls :func:`scipy..fligner <scipy.stats.fligner>`.

    Returns
    -------
    :class:`mqr.inference.hyptest.HypothesisTest`
    """
    if method == 'levene':
        if alternative != 'two-sided':
            raise ValueError('One-sided alternative not available in Levene test.')
        (stat, pvalue) = scipy.stats.levene(*x)
    elif method == 'fligner-killeen':
        if alternative != 'two-sided':
            raise ValueError('One-sided alternative not available in Fligner-Killeen test.')
        (stat, pvalue) = scipy.stats.fligner(*x)
    else:
        raise ValueError(util.method_error_msg(method, ['levene', 'fligner-killeen']))

    return HypothesisTest(
        description='equality of variances',
        alternative=alternative,
        method=method,
        sample_stat='var(x_i)',
        sample_stat_target='var(x_j)',
        sample_stat_value=stat,
        stat=stat,
        pvalue=pvalue,)
