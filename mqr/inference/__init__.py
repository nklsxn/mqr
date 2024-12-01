"""
Wrappers and functions for basic inference on samples of data.

The statistics covered are:
* quantile,
* mean,
* variance,
* proportion,
* rate,
* correlation, and
* distribution.

Routines in the top level (eg. `mqr.inference.mean`) are parametric, while
routines in the `nonparametric` module are non-parametric.

The modules have hypothesis tests for all of the listed distributions,
and also confidence intervals and sample-size calculators for the parametric
modules.

This is a list of reference material for further reading and advice.
- National Institute of Standards and Technology (US Dep. of Commerce).
    "Product and Process Comparisons."
    https://www.itl.nist.gov/div898/handbook/prc/prc.htm
- Brown, L. D. Cai, T. T. and DasGupta, A. (2001).
    "Interval estimation for a binomial proportion".
    Statistical Science, 16(2), 101-133.
- Patil, V. V., & Kulkarni, H. V. (2012).
    "Comparison of confidence intervals for the Poisson mean: some new aspects."
    REVSTAT-Statistical Journal, 10(2), 211-22.
- Park, H., & Leemis, L. M. (2019).
    Ensemble confidence intervals for binomial proportions.
    Statistics in Medicine, 38(18), 3460-3475.
- Newcombe, R. G. (1998).
    Two‐sided confidence intervals for the single proportion: comparison of seven methods.
    Statistics in medicine, 17(8), 857-872.
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
