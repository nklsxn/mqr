"""
================================================
Process and capability (:mod:`mqr.plot.process`)
================================================

.. currentmodule:: mqr.plot.process

.. rubric:: Functions
.. autosummary::
    :toctree: generated/

    capability
    pdf
    tolerance

Examples
--------
Since `pdf` and `tolerance` are elements used to show `capability`, this example
just shows the final capability. Capability is constructed from a
:class:`mqr.process.Process` object.

.. plot::

    fig, ax = plt.subplots(figsize=(7, 3))

    # Raw data
    data = pd.read_csv(mqr.sample_data('study-random-5x5.csv'))

    # Prepare data
    study = mqr.summary.Study(data, measurements=['KPO1'])
    specs = {
        'KPO1': mqr.process.Specification(160, 150, 170),
    }
    process = mqr.process.Process(study, specs)

    # Plot capability
    mqr.plot.process.capability(process, 'KPO1', ax)

"""

import mqr
from mqr.process import Specification, Capability, Process
from mqr.summary import Sample
from mqr.plot.defaults import Defaults
from mqr.plot.lib.util import set_kws

import matplotlib.transforms as transforms
import numpy as np
import scipy.stats as st
import seaborn as sns

def pdf(sample: Sample, spec: Specification, capability: Capability, ax,
        nsigma=None, cp=None, bins='auto', show_long_term=False,
        short_kws=None, long_kws=None):
    """
    Plots a Gaussian PDF of the given data.

    Parameters
    ----------
    sample : mqr.summary.Sample
        Sample to plot.
    spec : mqr.process.Specification
        Specification for the process sample.
    capability : mqr.process.Capability
        Capability of the process.
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
    ax.plot(xs[0], ys[0], **short_kws, label=f'$\\pm {nsigma/2} \\sigma$')
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

    ax.set_xlabel(f'{sample.name} (cp={capability.cp:#.3g}, cpk={capability.cpk:#.3g})')

def tolerance(spec: Specification, ax, prec=3,
              line_kws=None, tol_kws=None):
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

def capability(process: Process, name: str, ax, nsigma=None, cp=None, show_long_term=False):
    """
    Plots all three of the process histogram, fitted normal distribution, and
    tolerance region for the sample called `name` in `process`.

    Parameters
    ----------
    process : mqr.process.Process
        Process to plot.
    name : str
        Name of the process.
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
    sample = process.study[name]
    specification = process.specifications[name]
    capability = process.capabilities[name]

    sns.histplot(
        sample.data,
        color='C0',
        alpha=0.8,
        zorder=0,
        ax=ax,
        label='Sample histogram')
    tolerance(specification, ax=ax)
    pdf(sample=sample,
        spec=specification,
        capability=capability,
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
