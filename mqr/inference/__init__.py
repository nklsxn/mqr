"""
==================================================
Basic statistical inference (:mod:`mqr.inference`)
==================================================

.. :currentmodule:`mqr.inference`

User guide
    :doc:`/user_guide/inference`

Detailed examples
    https://github.com/nklsxn/mqrpy-guide

This package contains wrappers and functions for basic inference on statistics
calculated from samples of data. Many routines call either :mod:`scipy` or
:mod:`statsmodels` (see function docs).

One of the main goals of this module is to organise statistical tests by
purpose. The name of a test or method used in a routine is given in the
``method`` argument, instead of the function name.

Routines in the top level (:mod:`mqr.inference`) are parametric, while
routines in the :mod:`mqr.inference.nonparametric` module are non-parametric.

The modules implement hypothesis tests for all of the listed statistics,
and in addition, the parametric modules also calculate confidence intervals and
sample-size.

.. rubric:: Parametric statistics
.. autosummary::
    :template: autosummary/submodule.rst
    :toctree: generated/

    dist
    correlation
    mean
    proportion
    rate
    stddev
    variance

.. rubric:: Non-parametric statistics
.. autosummary::
    :template: autosummary/submodule.rst

    nonparametric.correlation
    nonparametric.dist
    nonparametric.median
    nonparametric.quantile
    nonparametric.variance

.. rubric:: Result types
.. autosummary::
    :toctree: generated/

    ~confint.ConfidenceInterval
    ~hyptest.HypothesisTest
    ~power.TestPower

Notes
-----
NIST [1]_ has further information on how to compare processes and products using
hypothesis tests. 

Examples
--------
All functions have a similar interface and return one of the result types above.

This example calculates a confidence interval for the difference between Pearson
correlation coefficients.

>>> x = np.array([1, 2, 1, 1, 2])
>>> y = np.array([3, 1, 2, 2, 1])
>>> mqr.inference.correlation.confint(x, y)
ConfidenceInterval(
    name='correlation',
    method='fisher-z',
    value=-0.8728715609439694,
    lower=-0.9915444093872248,
    upper=0.040865942695342376,
    conf=0.95,
    bounded='both')

This example calculates the sample size required to achieve an hypothesis test
on the mean with standardised effect size (Cohen's F) of 0.5, 95% confidence and
90% power.

>>> import mqr
>>> mqr.inference.mean.size_1sample(0.5, 0.05, 0.1, 'greater')
TestPower(
    name='mean',
    alpha=0.05,
    beta=0.1,
    effect=0.5,
    alternative='greater',
    method='t',
    sample_size=35.65267806948644)

Here is an example of an hypothesis test on the ratio of rates of events. Two
samples were taken, one before a process improvement activity and another after.
The sample before showed 5 defects in a sample of fabric 0.5m*0.5m, which is 20
defects per squqre metre. After the improvement, the same sized sample showed
only 1 defect --- that's 4 defects per square metre on average. This tests the
hypothesis that the defect rate has been decreased by at least 10 defects per
square metre.

>>> import mqr
>>> mqr.inference.rate.test_2sample(
>>>     5, 0.25, 1, 0.25, H0_value=10,
>>>     alternative='less', compare='diff')
HypothesisTest(
    description='difference between rates of events',
    alternative='less',
    method='score',
    sample_stat='rate1 - rate2',
    sample_stat_target=10,
    sample_stat_value=16.0,
    stat=0.65209454185206,
    pvalue=0.7428299075363056,
    null='rate1 - rate2 == 10',
    alt='rate1 - rate2 < 10')

From these samples, there is not enough evidence to conclude that the defect
rate has been improved by 10 defects per square metre.

This is an example of a test for normality. First, create a random sample of
F-distributed data.

>>> import numpy as np
>>> import scipy.stats as st
>>> np.random.seed(0)
>>> data = st.f(1, 2).rvs(40)

Check the hypothesis that they are normally distributed using the Kolmogorov-
Smirnov test.

>>> import mqr
>>> mqr.inference.dist.test_1sample(data, test='ks-norm')
HypothesisTest(
    description='non-normality',
    alternative='two-sided',
    method='kolmogorov-smirnov',
    sample_stat='distribution',
    sample_stat_target='normal',
    sample_stat_value=None,
    stat=0.2881073311024548,
    pvalue=0.0009999999999998899,
    null='distribution == normal',
    alt='distribution != normal')

References
----------
.. [1] National Institute of Standards and Technology (US Dep. of Commerce).
       "Product and Process Comparisons."
       https://www.itl.nist.gov/div898/handbook/prc/prc.htm
"""

import mqr.inference.nonparametric

import mqr.inference.correlation
import mqr.inference.dist
import mqr.inference.mean
import mqr.inference.power
import mqr.inference.proportion
import mqr.inference.rate
import mqr.inference.stddev
import mqr.inference.variance
