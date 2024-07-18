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
"""

import mqr.inference.nonparametric

import mqr.inference.correlation
import mqr.inference.dist
import mqr.inference.mean
import mqr.inference.proportion
import mqr.inference.rate
import mqr.inference.stddev
import mqr.inference.variance
