"""
=================================================
Statistical process control (:mod:`mqr.plot.spc`)
=================================================

.. currentmodule:: mqr.plot.spc

.. rubric:: Functions
.. autosummary::
    :toctree: generated

    chart
    alarms
    oc
"""

import mqr
from mqr.plot.defaults import Defaults
from mqr.plot.lib.util import set_kws

import numpy as np
import scipy.stats as st

def chart(control_statistic, control_params, ax, *,
          line_kws=None, in_kws=None, out_kws=None, target_kws=None, control_kws=None):
    """
    Plots an control chart defined by ControlParams.

    Parameters
    ----------
    control_statistic : mqr.spc.ControlStatistic
        The control statistic from samples of a monitored process.
    control_params : mqr.spc.ControlParams
        The control parameters, usually calculated from reference/historical data.
    ax : matplotlib.axes.Axes
        Axes for the plot.
    line_kws : dict, optional
        Keyword args for the statistic line (matplotlib.pyplot.plot).
    target_kws : dict, optional
        Keyword args for the target line (matplotlib.pyplot.axhline).
    control_kws : dict, optional
        Keyword args for the control lines (matplotlib.pyplot.plot).

    Examples
    --------
    This example charts the X-bar chart (sample mean) of a process with mean 1
    and standard deviation 5, using sample sizes of 6 observations. A total of
    20 samples are shown.

    .. plot::

        fig, ax = plt.subplots(figsize=(7, 3))

        # Raw data
        np.random.seed(0)
        x = pd.DataFrame(scipy.stats.norm(1, 5).rvs([20, 6]))

        # Parameters
        params = mqr.spc.XBarParams(centre=1, sigma=5)
        stat = params.statistic(x)

        # Charts
        mqr.plot.spc.chart(stat, params, ax=ax)

    """

    line_kws = set_kws(
        line_kws,
        marker=Defaults.marker,
        color='C0',
        zorder=1,
    )
    target_kws = set_kws(
        target_kws,
        linewidth=0.5,
        color='k',
        zorder=0,
    )
    control_kws = set_kws(
        control_kws,
        linewidth=0.5,
        color='gray',
        drawstyle='steps-mid',
        zorder=0,
    )

    stat = control_statistic.stat
    nobs = control_statistic.nobs
    index = stat.index
    target = control_params.target()
    lcl = control_params.lcl(nobs)
    ucl = control_params.ucl(nobs)

    line_values = [
        target,
        lcl.iloc[-1] if lcl is not None else None,
        ucl.iloc[-1] if ucl is not None else None,]
    yticks = [tick for tick in line_values if tick is not None]
    yticklabels = [
        f'{label}={value:g}'
        for label, value in zip(['target', 'LCL', 'UCL'], line_values)
        if value is not None]

    ax.plot(stat, **line_kws)
    ax.axhline(target, **target_kws)
    if lcl is not None:
        ax.plot(lcl, **control_kws)
    if ucl is not None:
        ax.plot(ucl, **control_kws)
    sec = ax.secondary_yaxis('right')
    sec.set_yticks(yticks)
    sec.set_yticklabels(yticklabels)
    ax.set_ymargin(0.15)
    ax.set_title(f'{control_params.name} chart')
    ax.set_xticks(index)

def alarms(control_statistic, control_params, control_rule, ax, *,
           point_kws=None, span_kws=None):
    """
    Plots alarms over a control chart.

    The control statistic and control parameters should be the same as the values
    passed to the `chart(...)` that drew the plot over which these alarms will be drawn.

    Alarm points are marked two ways:

    - a marker (a red dot by default) drawn over the statistic point, and
    - a region of colour (red by default) showing the span of alarmed points.

    Either can be switched off using point_kws/span_kws.

    Parameters
    ----------
    control_statistic : mqr.spc.ControlStatistic
        The control statistic from samples of a monitored process.
    control_params : mqr.spc.ControlParams
        The control parameters, usually calculated from reference/historical data.
    control_rule : Callable[[ControlStatistic, ControlParams], pandas.Series[bool]]
        A map from the control statistic and control parameters to a series of
        bools with a corresponding index, where True values are an alarm at that 
        index.
    ax : matplotlib.axes.Axes
    point_kws : dict, optional
        Keyword args for the alarm markers (matplotlib.pyplot.plot).
    span_kws : dict, optional
        Keyword args for the alarm regions (matplotlib.pyplot.axvspan).

    Examples
    --------
    This example shows an X-bar chart (sample mean) of a process with mean 1
    and standard deviation 5, using sample sizes of 6 observations. A total of
    20 samples are shown. The limits are 2 standard deviations of the standard
    error of the mean.

    .. plot::

        fig, ax = plt.subplots(figsize=(7, 3))

        # Raw data
        np.random.seed(0)
        x = pd.DataFrame(
            scipy.stats.norm(1, 5).rvs([20, 6]),
            columns=range(6))

        # Parameters
        params = mqr.spc.XBarParams(centre=1, sigma=5, nsigma=2)
        stat = params.statistic(x)
        rule = mqr.spc.rules.limits()

        # Charts
        mqr.plot.spc.chart(stat, params, ax=ax)
        mqr.plot.spc.alarms(stat, params, rule, ax=ax)

    """

    point_kws = set_kws(
        point_kws,
        linewidth=0,
        color='C3',
        marker='.',
    )
    span_kws = set_kws(
        span_kws,
        color='C3',
        alpha=0.2,
        zorder=-1,
    )

    stat = control_statistic.stat
    alarms = control_rule(control_statistic, control_params)
    for alarm in mqr.spc.util.alarm_subsets(alarms):
        ax.plot(stat.loc[alarm], **point_kws)
        a = alarm[0] - 0.5
        b = alarm[-1] + 0.5
        ax.axvspan(a, b, **span_kws)

def oc(n, c, ax, defect_range=None, line_kws=None):
    """
    Plot an OC curve.

    Parameters
    ----------
    n : int
        Sample size.
    c : int
        Acceptance number.
    ax : matplotlib.axes.Axes
        Axes for the plot.
    defect_range : tuple[float, float], optional
        Range of defect rates to show (on the x-axis).
    line_kws : dict, optional
        Keyword args for the line. Passed to :func:`matplotlib.pyplot.plot`.

    Examples
    --------
    This example shows an OC curve with sample size 40 and acceptance number 6.
    The plot shows only defect rates between 0 to 0.3 (along the x axis).

    .. plot::

        fig, ax = plt.subplots(figsize=(7, 3))

        mqr.plot.spc.oc(n=40, c=6, defect_range=(0, 0.3), ax=ax)

    """
    line_kws = set_kws(
        line_kws,
        color='C0',
    )

    if defect_range is None:
        defect_range = (0, 1)

    ps = np.linspace(*defect_range)

    pa = np.empty(ps.shape)
    for i in range(len(ps)):
        pa[i] = st.binom(n, ps[i]).cdf(c)

    ax.plot(ps, pa, **line_kws)
    ax.grid()
    ax.set_xlabel('Defect rate')
    ax.set_ylabel('Prob of Acceptance')
