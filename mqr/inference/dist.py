from mqr.inference.confint import ConfidenceInterval
from mqr.inference.hyptest import HypothesisTest

import mqr.inference.lib.util as util
import mqr.interop.inference as interop

def test_1sample(x, test='ad-norm'):
    """
    Hypothesis test on sampling distribution.

    Null-hypothesis
        | 'ad-norm', 'ks-norm'
        |   `x` was sampled from the a normal distribution

    Parameters
    ----------
    x : array_like
        Test the distribution of these values.
    test : {'ad-norm', 'ks-norm'}, optional
        | 'ad-norm'
        |   Anderson-Darling normality test.
            Calls :func:`sm..normal_ad <statsmodels.stats.diagnostic.normal_ad>`.
        | 'ks-norm'
        |   Kolmogoroc-Smirnov test against the normal distribution.
            Calls :func:`sm..kstest_normal <statsmodels.stats.diagnostic.kstest_normal>`.

    Returns
    -------
    :class:`mqr.inference.hyptest.HypothesisTest`
    """
    if test == 'ad-norm':
        from statsmodels.stats.diagnostic import normal_ad
        description = 'non-normality'
        alternative = 'two-sided'
        method = 'anderson-darling'
        target = 'normal'
        statistic, pvalue = normal_ad(x)
    elif test == 'ks-norm':
        from statsmodels.stats.diagnostic import kstest_normal
        description = 'non-normality'
        alternative = 'two-sided'
        method = 'kolmogorov-smirnov'
        target = 'normal'
        statistic, pvalue = kstest_normal(x, dist='norm')
    else:
        raise ValueError(util.method_error_msg(method, ['ad-norm', 'ks-norm']))

    return HypothesisTest(
        description=description,
        alternative=alternative,
        method=method,
        sample_stat=f'distribution',
        sample_stat_target=target,
        sample_stat_value=None,
        stat=statistic,
        pvalue=pvalue,
    )
