"""
Visual representation of descriptive statistics.
"""

import matplotlib.pyplot as plt
import seaborn as sns

import mqr
from mqr.summary import Sample

def summary(sample: Sample, ax):
    """
    Histogram, boxplot and confidence interval for a sample.

    Best plotted on axes arranged vertically.

    Arguments
    ---------
    sample (mqr.summary.Sample) -- Sample to use for all three plots.
    ax (matplotlib.axes.Axes) -- Axes for plot.
    """
    ax = ax.flatten()
    assert ax.shape == (3,) , f'Axes shape must be (3,) but is {ax.shape}.'

    sns.histplot(
        sample.data,
        color='lightgray',
        ax=ax[0])
    sns.boxplot(
        sample.data,
        orient='h',
        color='lightgray',
        linecolor='k',
        ax=ax[1])
    mqr.plot.confint(sample.conf_mean, ax=ax[2])

    ax[1].sharex(ax[0])
    ax[2].sharex(ax[0])
    plt.setp(ax[0].get_xticklabels(), visible=False)
    plt.setp(ax[1].get_xticklabels(), visible=False)
    ax[1].tick_params(axis='y', left=False)
    ax[2].set_title('')
    ax[0].set_xlabel('')
    ax[1].set_xlabel('')
    ax[2].set_xlabel(sample.name)
