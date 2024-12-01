"""
Hypothesis tests (parametric) for the distribution.
"""

from mqr.inference.confint import ConfidenceInterval
from mqr.inference.hyptest import HypothesisTest

import mqr.inference.lib.util as util
import mqr.interop.inference as interop

def test_1sample(x, test='ad-norm'):
    """
    Hypothesis test on distribution.

    Null-hypothesis: `x` was sampled from `dist`.

    Arguments
    ---------
    x (array[float]) -- Test the hypothesis that `x` was sampled from `dist`.

    Optional
    --------
    test (str) -- Statistical test to run. Only the default 'ad-norm' for the
        Anderson-Darling normality test is available
        (`statsmodels.stats.diagnostic.normal_ad`, statsmodels.org).

    Returns
    -------
    mqr.confit.HypothesisTest
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
