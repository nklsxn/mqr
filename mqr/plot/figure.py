"""
Tool for producing plots in jupyter notebooks.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from mqr.plot.defaults import Defaults

class Figure(object):
    """
    Context manager that creates, shows and closes matplotlib subplots.

    Convenient in jupyter notebooks for reducing plotting boilerplate.

    Examples
    --------
    >>> with Figure(5, 3, m=1, n=2) as (fig, ax):
    >>>     ax[0].plot(...)
    >>>     seaborn.histplot(..., ax=ax[1])

    """

    def __init__(
            self,
            width=6, height=3,
            m=1, n=1,
            backend=None,
            filename=None,
            rc_params=None,
            gridspec_kw=None,
            subplot_kw=None,
            **fig_kw):
        """
        Construct the Figure context manager.

        Parameters
        ----------
        width, height : float, optional
            Width and height of the figure.
        m, n : int, optional
            Size of the grid of subplots. (Default 1 and 1.)
        backend : optional
            A matplotlib backend to switch to for the duration of the context
            manager. The previous backend is restored when the context manager
            exits. The default `None` makes no change to the backend.
        filename : str, optional
            A filename to save the figure to before closing the figure. The
            default `None` results in no change.
        gridspec_kw : dict, optional
            Passed as ``gridspec_kw=gridspec_kw` to :func:`matplotlib.pyplot.subplots`.
        subplot_kw : dict, optional
            Passed as ``subplot_kw=subplot_kw`` to :func:`matplotlib.pyplot.subplots`.
        fig_kw : kwargs, optional
            Keyword args passed as ``**fig_wk` to :func:`matplotlib.pyplot.subplots`.
        """
        fig_kw['figsize'] = (width, height)
        if 'layout' not in fig_kw:
            fig_kw['layout'] = 'constrained'
        self.m = m
        self.n = n
        self.prev_backend = None
        self.backend = backend
        self.filename = filename
        self.gridspec_kw = gridspec_kw
        self.subplot_kw = subplot_kw
        self.fig_kw = fig_kw

        if rc_params is None:
            rc_params = {}
        rc_params = Defaults.rc_params | rc_params
        self.rc_context = matplotlib.rc_context(rc_params)

    def __enter__(self):
        self.rc_context.__enter__()
        if self.backend is not None:
            self.prev_backend = matplotlib.get_backend()
            matplotlib.use(self.backend)
        fig, ax = plt.subplots(
            self.m, self.n,
            gridspec_kw=self.gridspec_kw,
            subplot_kw=self.subplot_kw,
            **self.fig_kw)
        self.fig = fig
        return fig, ax

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.filename is not None:
            plt.savefig(self.filename)
        plt.show()
        plt.close(self.fig)
        if self.prev_backend is not None:
            matplotlib.use(self.prev_backend)
        self.rc_context.__exit__(exc_type, exc_val, exc_tb)

def points_to_inches(width_pts, *, height_pts=None, hw_ratio=None):
    _PT_PER_INCH = 72.27
    width_inches = width_pts / _PT_PER_INCH
    if height_pts is not None:
        height_inches = height_pts / _PT_PER_INCH
    else:
        if hw_ratio is None:
            hw_ratio = 2 / (1 + np.sqrt(5))
        height_inches = width_inches * hw_ratio
    return width_inches, height_inches

def mm_to_inches(width_mm, *, height_mm=None, hw_ratio=None):
    _PT_PER_MM = 25.4
    width_inches = width_mm / _PT_PER_MM
    if height_mm is not None:
        height_inches = height_mm / _PT_PER_MM
    else:
        if hw_ratio is None:
            hw_ratio = 2 / (1 + np.sqrt(5))
        height_inches = width_inches * hw_ratio
    return width_inches, height_inches
