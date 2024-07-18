"""
Tools for probability plots.
"""

import numpy as np
import scipy.stats as st

def pp_grp(x, grp_ax, grp=None, dist=None, grp_kwargs=None):
    """
    Plots data points against a theoretical distribution by using a P-P plot.
    
    When grouping is used, calculates the fit from all points, then produces a plot for each group.
    Based on M&R, chapter 6.
    
    Arguments
    ---------
    x (numpy.ndarray) -- Empirical data.
    grp_ax (np.ndarray[matplotlib.axes.Axes]) -- Axes for the plots. Must have
        same number of elements as `len(grp)`.

    Optional
    --------
    grp (list[numpy.ndarray]) -- A list of indexers that select which points to plot.
        Default incremental from `0` to `len(x)-1`.
    dist -- Distribution object from scipy.stats. For example `scipy.stats.norm(0, 1)`.
        Defaults to Standard Normal.
    grp_kwargs (list[dict]) -- Dictionaries containing kwargs passed to each
        axis (per group). Must be the same length as grp.
        the same number of elements as `grp`. Will be `flatten`ed before use.
    """
    from collections.abc import Iterable
    import scipy

    n = len(x)
    if grp is None:
        grp = [np.arange(n)]

    N = len(grp)

    if (grp_kwargs is not None) and (len(grp_kwargs) != N):
        raise ValueError('`grp_kwargs` must be None or the same length as x')
    if grp_kwargs is None:
        grp_kwargs = [{}] * N

    if not isinstance(grp_ax, Iterable):
        grp_ax = [grp_ax]
    grp_ax = grp_ax.flatten()

    if dist is None:
        dist = st.norm(np.mean(x), np.std(x, ddof=1))

    p = np.sort(x)
    pp = (np.arange(n) + 0.5) / n

    for g, kwargs, ax in zip(grp, grp_kwargs, grp_ax):
        def_kwargs = {'linewidth': 0, 'marker': '.', 'color': 'k'}
        def_kwargs.update(kwargs)
        ax.plot(p[g], pp[g], **def_kwargs)

        ax.set_yscale('function', functions=(dist.ppf, dist.cdf))
        yticks = np.array([0.001, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99, 0.999])
        ax.set_yticks(yticks)
        ax.set_yticklabels([f'{100*t}' for t in yticks])

        ax.set_title('Probability plot')
        ax.set_ylabel('Theoretical probability (%)')

def pp_grp_cdfline(x, grp_ax, dist=None, grp_kwargs=None):
    """
    Adds the cdf line (for use on a P-P plot).
    
    Arguments
    ---------
    x (numpy.ndarray) -- Empirical data.
    grp_ax (np.ndarray[matplotlib.axes.Axes]) -- Axes for the plots. Must have
        the same number of elements as `grp`. Will be `flatten`ed before use.

    Optional
    --------
    dist -- Distribution object from scipy.stats; for example `scipy.stats.norm(0, 1)`.
        Defaults to Standard Normal.
    grp_kwargs (list[dict]) -- Dictionaries containing kwargs passed to each
        axis (per group). Must be the same length as grp_ax.
    """
    from collections.abc import Iterable
    import scipy

    N = len(grp_ax)

    if (grp_kwargs is not None) and (len(grp_kwargs) != N):
        raise ValueError('`grp_kwargs` must be None or the same length as x')

    if grp_kwargs is None:
        grp_kwargs = [{}] * N

    if not isinstance(grp_ax, Iterable):
        grp_ax = [grp_ax]

    if dist is None:
        dist = st.norm(np.mean(x), np.std(x, ddof=1))

    xs = np.linspace(np.min(x), np.max(x))
    ys = dist.cdf(xs)

    for ax, kwargs in zip(grp_ax, grp_kwargs):
        def_kwargs = {'linewidth': 1.0, 'color': 'grey'}
        def_kwargs.update(kwargs)
        ax.plot(xs, ys, **def_kwargs)

def pp_grp_qline(x, grp_ax, q_lower=0.25, q_upper=0.75, dist=None, grp_kwargs=None):
    """
    Adds a line through two quantile points (for use on a P-P plot).

    Suggested by M&R in Example 6-7.
    
    Arguments
    ---------
    x (numpy.ndarray) -- Empirical data.
    grp_ax (np.ndarray[matplotlib.axes.Axes]) -- Axes for the plots. Must have
        the same number of elements as `grp`. Will be `flatten`ed before use.

    Optional
    --------
    q_lower (float) -- Lower quantile point. Default 0.25.
    q_upper (float) -- Upper quantile point. Default 0.75.
    dist -- Distribution object from scipy.stats; for example `scipy.stats.norm(0, 1)`.
        Defaults to Standard Normal.
    grp_kwargs (list[dict]) -- Dictionaries containing kwargs passed to each
        axis (per group). Must be the same length as grp_ax.
    """
    from collections.abc import Iterable
    import scipy

    n = len(x)
    N = len(grp_ax)

    if (grp_kwargs is not None) and (len(grp_kwargs) != N):
        raise ValueError('`grp_kwargs` must be None or the same length as x')

    if grp_kwargs is None:
        grp_kwargs = [{}] * N

    if not isinstance(grp_ax, Iterable):
        grp_ax = [grp_ax]

    if dist is None:
        dist = st.norm(np.mean(x), np.std(x, ddof=1))

    ql = np.quantile(x, q_lower)
    qu = np.quantile(x, q_upper)
    m = (dist.ppf(q_upper) - dist.ppf(q_lower)) / (qu - ql)
    ys = np.linspace(0.001, 0.999) # matches the pp plot, but poor form here
    xs = (dist.ppf(ys) - dist.ppf(q_lower)) / m + ql

    for ax, kwargs in zip(grp_ax, grp_kwargs):
        def_kwargs = {'linewidth': 1.0, 'color': 'grey'}
        def_kwargs.update(kwargs)
        ax.plot(xs, ys, **def_kwargs)
        ax.plot(ql, q_lower, marker='+', markersize=20, color='C0', fillstyle='none')
        ax.plot(qu, q_upper, marker='+', markersize=20, color='C0', fillstyle='none')
