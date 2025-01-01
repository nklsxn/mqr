from mergedeep import merge
import matplotlib.transforms as transforms
import numpy as np
import scipy

from mqr.plot.defaults import Defaults

def grouped_df(data, ax,
               line_kws=None, text_kws=None):
    """
    Plots from a pandas.DataFrame with columns grouped along the x-axis.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe whose columns will be plot next to each other.
    ax : matplotlib.axes.Axes
        Axes for plot.
    line_kws : dict
        Keyword args for lines, passed to :func:`matplotlib..plot <matplotlib.pyplot.plot>`.
    text_kws : dict
        Keyword args for text, passed to :func:`matplotlib..text <matplotlib.pyplot.text>`.
    """
    line_kws = set_kws(
        line_kws,
        color='C0',
        marker=Defaults.marker,
        markersize=5,
    )
    text_kws = set_kws(
        text_kws,
        rotation=0,
        rotation_mode='anchor',
        va='top',
        ha='left',
    )

    idx_labels = data.index
    idx = np.arange(len(idx_labels))
    M = len(idx)

    groups = data.columns
    N = len(groups)

    tr = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for i, g in enumerate(groups):
        xs = idx + i * M
        ax.plot(xs, data.loc[:, g], label=g, **line_kws)
        sep_x = i * M - 0.5
        ax.axvline(sep_x, linewidth=0.8, color='gray', linestyle=(0, (5, 5)))
        ax.text(sep_x+0.25, 0.99, g, transform=tr, **text_kws)

    ax.set_xticks(np.arange(len(idx)*N))
    ax.set_xticklabels(np.tile(idx_labels, N), rotation=90)

def scaled_density(xs, data, dist=None, bins='auto'):
    """
    Calculate the PDF of the points `xs`, scaled to match a histogram.

    Parameters
    ----------
    xs : array_like
        X values.
    data : array_like
        Sampled random data.
    dist : :class:`scipy.stats.rv_continuous`
        Distribution of the density.
    bins : str, optional
        Passed to :func:`numpy.histogram_bin_edges` to calculate bin widths.

    Returns
    -------
    array_like
        Y values corresponding to `xs` for the scaled density.
    """
    if dist is None:
        dist = scipy.stats.norm.fit(data)
    elif isinstance(dist, scipy.stats.distributions.rv_continuous):
        dist = dist(*dist.fit(data))
    N = len(data)
    edges = np.histogram_bin_edges(data, bins=bins)
    binwidth = edges[1] - edges[0]
    ys = dist.pdf(xs)
    return ys * N * binwidth

def set_kws(input_kws, **default_kws):
    """
    Merges dicts of keyword arguments.

    Parameters
    ----------
    input_kws : dict
        Overriding keyword arguments that, when specified, replace the values
        in `default_kws`.
    default_kws : dict
        Default values that can be overriden by `input_kws`.

    Returns
    -------
    dict
        A set of keyword arguments containing all of `input_kws` and any of
        `default_kws` whose key did not appear in `default_kws`.
    """
    if input_kws is None:
        input_kws = {}
    return merge(default_kws, input_kws)
