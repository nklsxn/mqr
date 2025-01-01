"""
================================================
Process and capability (:mod:`mqr.plot.process`)
================================================

.. currentmodule:: mqr.plot.process

.. rubric:: Functions
.. autosummary::
    :toctree: generated/

    summary
    pdf
    tolerance
    capability
"""

import mqr
from mqr.process import Sample, Specification, Capability, Summary
from mqr.plot.defaults import Defaults
from mqr.plot.lib.util import set_kws

from matplotlib import pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import scipy.stats as st
import seaborn as sns

def pdf(sample: Sample, ax,
        nsigma=None, cp=None, bins='auto', show_long_term=False,
        short_kws=None, long_kws=None):
    """
    Plots a Gaussian PDF for the given sample.

    The PDF is truncated at six sigmas on each side.

    Parameters
    ----------
    sample : mqr.process.Sample
        Sample to plot.
    ax : matplotlib.axes.Axes
        Axes for the plot.
    nsigma : float, optional
        How many stddevs of the Gaussian PDF to plot (total). Default 6.
    cp : float, optional
        Width of the plot in triples of the standard deviation. Ie. when `cp` is
        2, the width will be 6 standard deviations. Default 2.
    bins : str, optional
        Passed to :func:`numpy.histogram_bin_edges`. Can be used to ensure the
        density corresponds to a histogram on the same plot.
    show_long_term : bool, optional
        Plots two more densities shifted left and right by 1.5 standard deviations.
    short_kws : dict, optional
        Keyword arguments passed to `matplotlib.pyplot.plot` for short-term
        densities.
    long_kws : dict, optional
        Keyword arguments passed to `matplotlib.pyplot.fill_between` for
        long-term densities.

    Examples
    --------
    See example in :mod:`mqr.plot.process`.
    """
    if (nsigma is not None) and (cp is not None):
        raise ValueError(f'Only one of `nsigma` or `cp` can be specified.')
    elif (nsigma is None) and (cp is None):
        nsigma = 6.0

    short_kws = set_kws(
        short_kws,
        color='C1',
        marker=2,
        markersize=8,
        mew=2,
        zorder=1,
    )
    long_kws = set_kws(
        long_kws,
        color='C2',
        alpha=0.1,
        zorder=1,
    )

    if cp is not None:
        nsigma = cp * 3

    dist = st.norm(sample.mean, sample.std)
    xmin_st = sample.mean - nsigma * sample.std
    xmax_st = sample.mean + nsigma * sample.std
    xs = np.linspace(xmin_st, xmax_st, 250)
    ys = mqr.plot.lib.util.scaled_density(xs, sample.data, bins=bins, dist=dist)

    line_kws = {k:v for k, v in short_kws.items() if k != 'marker'}
    ax.plot(xs, ys, **line_kws, label='Fitted density')
    ax.plot(xs[0], ys[0], **short_kws, label=f'$\\pm {nsigma:.2f} \\sigma$; $c_p={nsigma/3:.2f}$')
    ax.plot(xs[-1], ys[-1], **short_kws)

    if show_long_term:
        fill_kws = {k:v for k, v in long_kws.items() if k not in []}
        shift = 1.5 * sample.std
        dist_l = st.norm(sample.mean - shift, sample.std)
        dist_r = st.norm(sample.mean + shift, sample.std)
        ys_l = mqr.plot.lib.util.scaled_density(xs, sample.data, bins=bins, dist=dist_l)
        ys_r = mqr.plot.lib.util.scaled_density(xs, sample.data, bins=bins, dist=dist_r)
        ax.fill_between(xs, ys_l, **fill_kws, label='Long-term densities')
        ax.fill_between(xs, ys_r, **fill_kws)

def tolerance(spec: Specification, ax,
              prec=3, line_kws=None, tol_kws=None):
    """
    Plots tolerance region.

    Parameters
    ----------
    spec : mqr.process.Specification
        Spec containing tolerance bounds.
    ax : matplotlib.axes.Axes
        Axes for the plot.
    prec : int, optional
        Significant figures for the limit bounds.
    line_kws : dict, optional
        Keyword arguments for vertical lines at the centreline and limits. Passed
        to `matplotlib.pyplot.axvline`.
    tol_kws : dict, optional
        Keyword arguments for the shading over the tolerance region. Passed
        to `matplotlib.pyplot.axvspan`.

    Examples
    --------
    See example in :mod:`mqr.plot.process`.
    """
    line_kws = set_kws(
        line_kws,
        color='gray',
        zorder=-1,
    )
    tol_kws = set_kws(
        tol_kws,
        color='lightgray',
        zorder=-2,
    )

    ax.axvline(spec.target, **line_kws)
    ax.axvline(spec.lsl, **line_kws)
    ax.axvline(spec.usl, **line_kws)
    ax.axvspan(spec.lsl, spec.usl, **tol_kws, label='Tolerance')

    sec = ax.secondary_xaxis(location='top')
    sec.set_xticks([spec.lsl, spec.target, spec.usl])
    sec.tick_params(axis='x', which='major', direction='out',
                    top=True, labeltop=True, bottom=False, labelbottom=False)

def capability(summary: Summary, name: str, ax, nsigma=None, cp=None, show_long_term=False):
    """
    Plots all three of the process histogram, fitted normal distribution, and
    tolerance region for the sample called `name` in `summary`.

    Parameters
    ----------
    summary : mqr.process.Summary
        Summary containing sample with `name`.
    name : str
        Name of process whose capability will be shown.
    ax : matplotlib.axes.Axes
        Axes for the plot.
    nsigma : float, optional
        How many stddevs of the Gaussian PDF to plot (total). Default 6.
    cp : float, optional
        Width of the plot in triples of the standard deviation. Ie. when `cp` is
        2, the width will be 6 standard deviations. Default 2.
    show_long_term : bool, optional
        Plots two more densities shifted left and right by 1.5 standard deviations.

    Examples
    --------
    See example in :mod:`mqr.plot.process`.
    """
    if name not in summary.capabilities:
        raise ValueError(f'Summary has no specification/capability defined for {name}.')

    sample = summary[name]
    capability = summary.capabilities[name]
    specification = capability.spec

    sns.histplot(
        sample.data,
        color='C0',
        alpha=0.8,
        zorder=0,
        ax=ax,
        label='Sample histogram')
    tolerance(specification, ax=ax)
    pdf(sample=sample,
        show_long_term=show_long_term,
        nsigma=nsigma,
        cp=cp,
        ax=ax)
    ax.set
    ax.set_xlabel(
        f'{sample.name} ('
        f'target={specification.target:#.3g}, '
        f'cp={capability.cp:#.3g}, '
        f'cpk={capability.cpk:#.3g}'
        ')')
    ax.set_ylabel('count')

def summary(sample: Sample, ax, hyp_mean=None,
            hist_kws=None, box_kws=None, conf_kws=None):
    '''
    Histogram, boxplot and confidence interval for a sample.

    Best plotted on axes arranged vertically.

    Parameters
    ----------
    sample : mqr.process.Sample
        Sample to use for all three plots.
    ax : matplotlib.axes.Axes
        Axes for plot.
    hyp_mean: float, optional
        Hypothesised mean to plot on a confidence interval.

    Examples
    --------
    See example in :mod:`mqr.plot.process`.
    '''
    hist_kws = set_kws(
        hist_kws,
        color='C0',
    )
    box_kws = set_kws(
        box_kws,
        orient='h',
        color='C0',
    )
    conf_kws = set_kws(
        conf_kws,
    )

    ax = ax.flatten()
    if ax.shape != (3,):
        raise ValueError(f'Axes shape must be (3,) but is {ax.shape}.')

    sns.histplot(sample.data, ax=ax[0], **hist_kws)
    sns.boxplot(sample.data, ax=ax[1], **box_kws)
    mqr.plot.confint(sample.conf_mean, ax=ax[2], hyp_value=hyp_mean, **conf_kws)

    ax[1].sharex(ax[0])
    ax[2].sharex(ax[0])
    plt.setp(ax[0].get_xticklabels(), visible=False)
    plt.setp(ax[1].get_xticklabels(), visible=False)
    ax[1].tick_params(axis='y', left=False)
    ax[2].set_title('')
    ax[0].set_xlabel('')
    ax[1].set_xlabel('')
    ax[2].set_xlabel(sample.name)
