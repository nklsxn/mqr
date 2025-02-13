from mqr.inference.confint import ConfidenceInterval
from mqr.inference.hyptest import HypothesisTest

import mqr.inference.lib.util as util
import mqr.interop.inference as interop

import warnings

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
        | 'lf-norm'
        |   Lilliefors test against the normal distribution.
            Lilliefors test is a Kolmogorov-Smirnov test with estimated parameters.
            Calls :func:`sm..lilliefors <statsmodels.stats.diagnostic.lilliefors>`.
        | 'sw-norm'
        |   Shapiro-Wilk normality test.
            Calls :func:`scipy..shaprio <scipy.stats.shapiro>`.
        | 'dp-norm'
        |   Dagastino-Pearson normality test.
            Calls :func:`scipy..normaltest <scipy.stats.normaltest>`.
        | 'jb-norm'
        |   Jarque-Bera normality test. Can lead to high Type-I error under some conditions,
            see the discussion on `Wikipedia <https://en.wikipedia.org/wiki/Jarque–Bera_test>`_.
            Calls :func:`scipy..jarque_bera <scipy.stats.jarque_bera>`.
        | 'cvm-norm'
        |   Cramer-von Mises test for goodness of fit, applied to the standard Normal CDF.
            Calls :func:`scipy..cramervonmises <scipy.stats.cramervonmises>`.

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
    elif test == 'lf-norm':
        from statsmodels.stats.diagnostic import lilliefors
        description = 'non-normality'
        alternative = 'two-sided'
        method = 'lilliefors'
        target = 'normal'
        statistic, pvalue = lilliefors(x, dist='norm')
    elif test == 'sw-norm':
        from scipy.stats import shapiro
        description = 'non-normality'
        alternative = 'two-sided'
        method = 'shapiro-wilk'
        target = 'normal'
        statistic, pvalue = shapiro(x)
    elif test == 'dp-norm':
        from scipy.stats import normaltest
        description = 'non-normality'
        alternative = 'two-sided'
        method = 'dagostino-pearson'
        target = 'normal'
        statistic, pvalue = normaltest(x)
    elif test == 'jb-norm':
        if len(x) < 100:
            msg = (
                'This method is recommended for large samples. '
                'Consider using another method. '
                'See the discussion in https://en.wikipedia.org/wiki/Jarque–Bera_test.')
            warnings.warn(msg)
        from scipy.stats import jarque_bera
        description = 'non-normality'
        alternative = 'two-sided'
        method = 'jarque-bera'
        target = 'normal'
        statistic, pvalue = jarque_bera(x)
        if pvalue < 0.05:
            msg = (
                'This method might be inaccurate when the p-value is small. '
                'Consider using another method. '
                'See the discussion in https://en.wikipedia.org/wiki/Jarque–Bera_test.')
            warnings.warn(msg)
    elif test == 'cvm-norm':
        from scipy.stats import cramervonmises, norm
        description = 'non-normality'
        alternative = 'two-sided'
        method = 'cramer-vonmises'
        target = 'normal'
        res = cramervonmises(x, norm().cdf)
        statistic, pvalue = res.statistic, res.pvalue
    else:
        valid_methods = [
            'ad-norm',
            'ks-norm',
            'lf-norm',
            'sw-norm',
            'dp-norm',
            'jb-norm',
            'cvm-norm',
        ]
        raise ValueError(util.method_error_msg(test, valid_methods))

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
