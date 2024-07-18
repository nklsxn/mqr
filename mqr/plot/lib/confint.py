"""
Plot confidence intervals.
"""

import matplotlib.pyplot as plt
import numpy as np

from mqr.inference.confint import ConfidenceInterval

def confint(ci: ConfidenceInterval, ax, hyp_value=None):
    """
    Draw a confidence interval with an hypothesised value; a graphical
    representation of an hypothesis test.

    When the the hypothesised value (`hyp_value`) is outside the confidence
    interval, the null-hypothesis is refejected at the confidence level used to
    construct the interval (`ci.conf`).

    Representing this relationship graphically makes sense when the distribution
    of the test statistic is the distribution of the sample. For example, in an
    hypothesis test for the mean of a sample, using a z-score as the test statistic.

    Arguments
    ---------
    ci (mqr.confint.ConfidenceInterval) -- The confidence interval to draw.
    ax (matplotlib.axes.Axes) -- Axes for the plot.

    Optional
    --------
    hyp_value (float) -- The null-hypothesis value.
    """
    y_loc = 0

    if not np.isfinite(ci.lower):
        ax.plot(2 * ci.value - ci.upper, y_loc, marker=None, linewidth=0)

    if not np.isfinite(ci.upper):
        ax.plot(2 * ci.value - ci.lower, y_loc, marker=None, linewidth=0)

    l, r = ax.get_xlim()
    l_mkr = '<'
    r_mkr = '>'
    if np.isfinite(ci.lower):
        l = ci.lower
        l_mkr = '|'

    if np.isfinite(ci.upper):
        r = ci.upper
        r_mkr = '|'

    ax.plot(l, y_loc, marker=l_mkr, color='k')
    ax.plot(r, y_loc, marker=r_mkr, color='k')
    ax.plot(np.linspace(l, r), np.linspace(y_loc, y_loc), color='k', linewidth=0.8)
    ax.plot(ci.value, y_loc, marker='o', color='C0')
    if hyp_value is not None:
        ax.plot(hyp_value, y_loc, marker='o', color='C3')

    ax.set_yticks([y_loc])
    ax.set_yticklabels([ci.name])
    ax.tick_params(axis='y', left=False)
    ax.set_ylim(y_loc-1, y_loc+1)
