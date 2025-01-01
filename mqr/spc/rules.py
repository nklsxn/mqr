"""
======================================
SPC alarm rules (:mod:`mqr.spc.rules`)
======================================

.. currentmodule:: mqr.spc.rules

User guide
    :doc:`/user_guide/statistical-process-control`

Detailed examples
    https://github.com/nklsxn/mqrpy-guide

.. rubric:: Functions

.. autosummary::
    :toctree: generated/

    limits
    aofb_nsigma
    n_1side
    n_trending
    combine
"""

from dataclasses import dataclass, field
import functools
import numpy as np
import pandas as pd

def combine(combn_fn, *rules):
    """
    Create a new rule from others and a logical combination.

    **A word on types**. This function can be confusing because it transforms
    functions into another function. The list of arguments `rules` and the
    result of this function all have the same signature. Take a look at
    :func:`limits` for an example of that signature. On the other hand,
    `combn_fn` has a different signature: it takes two series of bools and
    returns another series of bools whose elements correspond to the first two.
    That is, it takes results from two rules and makes a new result.
    See the example below.

    The new rule is the same as evaluating each rule in `rules`, then combining
    the outputs with `combn_fn(left_result, right_result)` from left to right.

    Parameters
    ----------
    combn_fn : Callable[[[Series[bool], pandas.Series[bool]], pandas.Series[bool]]
        Function that combines the result series from two rules into a single
        Series.
    *rules : Callable[[ControlStatistic, ControlParams], Series[bool]]
        One or more rules whose results will be combined.

    Examples
    --------

    **Creating a combination**

    This example creates a combination function that alarms when both of its
    inputs are alarmed, but not when only one is an alarm.

    >>> def both(series1, series2):
    >>>     result = pd.Series(index=series1.index, dtype=bool)
    >>>     for i in range(len(result)):
    >>>         result[i] = series1[i] and series2[i]
    >>>     return result

    That code demonstrates the interface with a simple example, but in practise,
    use :func:`numpy.logical_and` for the functionality of "both" above, since
    it is much more efficient, and already written:

    >>> mqr.spc.rules.combine(np.logical_and, rule1, rule2)

    **Using a combination**

    Create a rule that alarms when either (1) a process goes outside its control
    limits using :func:`mqr.spc.rules.limits()`, or (2) the process has 3 of 4
    observations outside 2 sigmas using :func:`mqr.spc.rules.aofb_nsigma`. To
    make the numbers simple, the process here has standard deviation of 1 and
    samples of size 1 are taken -- a single observation of 1 is then a sample
    with mean at 1 standard error.

    1. The data first violates the UCL of 3, with an average observation 3.5.
    2. The data then violates the 3/4 outside 2sigma limit, with the four sample
       averages 2.5, 1, 2.5, 2.5 (three of which are bigger than 2).
       The last of these four observations is labelled as violating the limit.

    .. plot::

        fig, ax = plt.subplots(figsize=(7, 3))

        # Make the new rule
        combine_rule = mqr.spc.rules.combine(
            np.logical_or,
            mqr.spc.rules.limits(),
            mqr.spc.rules.aofb_nsigma(a=3, b=4, n=2))

        # Create violating data for demonstration
        mean = 5
        sigma = 1
        zscore = np.array([[0, 3.5, 0, 0, 0, 2.5, 1, 2.5, 2.5, 0]]).T
        #                      ^^^(>UCL)     ^^^^^^^^^^^^^^^^(3/4>2)
        data = pd.DataFrame(mean + zscore * sigma)

        # Create parameters and calculate statistic for the example chart
        params = mqr.spc.XBarParams(centre=mean, sigma=sigma)
        stat = params.statistic(data)

        # Show the chart and alarms overlay for the combined rule
        mqr.plot.spc.chart(stat, params, ax=ax)
        mqr.plot.spc.alarms(stat, params, combine_rule, ax=ax)

    """
    def _combine(control_statistic, control_params):
        alarms = [rule(control_statistic, control_params) for rule in rules]
        return functools.reduce(combn_fn, alarms)
    return _combine

def limits():
    """
    Rule monitoring when a process violates control limits.

    This function is an alarm rule that triggers an alarm any time the statistic
    is outside the upper or lower control limits.

    This is a rule and can be used directly in :func:`mqr.plot.spc.alarms` or
    :func:`combine`.

    Returns
    -------
    Callable[(ControlStatistic, ControlParams), pandas.Series[bool]]
        A function taking a control statistic and the params used to create them,
        and returning a series with `True` marking alarms.

    Examples
    --------
    This function can be passed directly to the plotting routines. This example
    overlays alarms on an X-bar chart.

    .. plot::

        data = pd.DataFrame({
            'Xbar': np.array([9, 10, 9, 13, 12, 10])
        #                               ^^ (>UCL)
        })

        params = mqr.spc.XBarParams(centre=10, sigma=1)
        stat = params.statistic(data)
        rule = mqr.spc.rules.limits()

        fig, ax = plt.subplots(figsize=(7, 3))
        mqr.plot.spc.chart(stat, params, ax=ax)
        mqr.plot.spc.alarms(stat, params, rule, ax=ax)

    """
    def _rule(control_statistic, control_params):
        stat = control_statistic.stat
        nobs = control_statistic.nobs
        lcl = control_params.lcl(nobs)
        ucl = control_params.ucl(nobs)
        return np.logical_or(stat >= ucl, stat <= lcl)
    return _rule

def aofb_nsigma(a, b, n):
    """
    Rule monitoring the proportion of samples outside a multiple of sigma.

    This function returns an alarm rule that alarms when `a` out of `b` statistics
    in a row are beyond `n` multiples of the statistic standard error. Only the
    last sample of the `b` is marked as an alarm.

    This function generates a rule. The result of this function is another
    function that can be passed to :func:`mqr.plot.spc.alarms` or
    :func:`combine`.

    Parameters
    ----------
    a : int
        Intensity -- the number of statistics in `b` periods outside `n`
        standard deviations required to trigger an alarm.
    b : int
        Period -- the number of samples to check for `a` statistics outside
        `n` standard deviations required to trigger an alarm.
    n : float
        Threshold in multiples of the standard error.

    Returns
    -------
    Callable[(ControlStatistic, ControlParams), pandas.Series[bool]]
        A function taking a control statistic and the params used to create them,
        and returning a series with `True` marking alarms.

    Examples
    --------
    The result of this function can be passed to the plotting routines. This
    example overlays a "4/5 > 2sigma" rule on an X-bar chart.

    .. plot::

        data = pd.DataFrame({
            'Xbar': np.array([11, 13, 10, 12, 12, 12, 9, 11, 12])
        #                         ^^^^^^^^^^^^^^^^^^ (4/5>=2sigma)
        })

        params = mqr.spc.XBarParams(centre=10, sigma=1)
        stat = params.statistic(data)
        rule = mqr.spc.rules.aofb_nsigma(a=4, b=5, n=2)

        fig, ax = plt.subplots(figsize=(7, 3))
        mqr.plot.spc.chart(stat, params, ax=ax)
        mqr.plot.spc.alarms(stat, params, rule, ax=ax)

    """
    if a > b:
        raise ValueError(f'Cannot detect more than b of b signals (was passed "{a} of {b}").')
    def _rule(control_statistic, control_params):
        stat = control_statistic.stat
        nobs = control_statistic.nobs
        target = control_params.target()
        se = control_params.se(nobs)

        alarms = pd.Series(False, index=stat.index)
        for seq in (stat >= target + n * se).rolling(b):
            if np.sum(seq) >= a:
                alarms[seq.index[-1]] = True
        for seq in (stat <= target - n * se).rolling(b):
            if np.sum(seq) >= a:
                alarms[seq.index[-1]] = True
        return alarms
    return _rule

def n_1side(n):
    """
    Rule monitoring the count on one side the centreline.

    This function returns an alarm rule that alarms when `n` statistics in a row
    fall on one side of the centreline.

    This function generates a rule. The result of this function is another
    function that can be passed to :func:`mqr.plot.spc.alarms` or
    :func:`combine`.

    Parameters
    ----------
    n : int
        Number of consecutive statistics falling on one side of the centreline
        required to trigger an alarm.

    Returns
    -------
    Callable[(ControlStatistic, ControlParams), pandas.Series[bool]]
        A function taking a control statistic and the params used to create them,
        and returning a series with `True` marking alarms.

    Examples
    --------

    .. plot::

        data = pd.DataFrame({
            'Xbar': np.array([9, 10, 11, 13, 12, 8])
        #                            ^^^^^^^^^^ (3 > target)
        })

        params = mqr.spc.XBarParams(centre=10, sigma=1)
        stat = params.statistic(data)
        rule = mqr.spc.rules.n_1side(3)

        fig, ax = plt.subplots(figsize=(7, 3))
        mqr.plot.spc.chart(stat, params, ax=ax)
        mqr.plot.spc.alarms(stat, params, rule, ax=ax)

    """
    def _rule(control_statistic, control_params):
        stat = control_statistic.stat
        target = control_params.target()

        alarms = pd.Series(False, index=stat.index)
        for seq in np.sign(stat - target).rolling(n):
            values = set(seq)
            if (len(seq) == n) and (len(values) == 1) and (values != {0}):
                alarms[seq.index[-1]] = True

        return alarms
    return _rule

def n_trending(n):
    """
    Rule monitoring the count of trending points.

    This function returns an alarm rule that alarms when `n` statistics in a row
    trend (up or down).

    This function generates a rule. The result of this function is another
    function that can be passed to :func:`mqr.plot.spc.alarms` or
    :func:`combine`.

    Parameters
    ----------
    n : int
        Number of sample in a trend required to cause an alarm.

    Returns
    -------
    Callable[(ControlStatistic, ControlParams), pandas.Series[bool]]
        A function taking a control statistic and the params used to create them,
        and returning a series with `True` marking alarms.

    Examples
    --------

    .. plot::

        data = pd.DataFrame({
            'Xbar': np.array([11, 9, 12, 10, 9, 8, 12])
        #                            ^^^^^^^^^^^^ (4 trending)
        })

        params = mqr.spc.XBarParams(centre=10, sigma=1)
        stat = params.statistic(data)
        rule = mqr.spc.rules.n_trending(4)

        fig, ax = plt.subplots(figsize=(7, 3))
        mqr.plot.spc.chart(stat, params, ax=ax)
        mqr.plot.spc.alarms(stat, params, rule, ax=ax)

    """
    def _rule(control_statistic, control_params):
        stat = control_statistic.stat
        target = control_params.target()

        alarms = pd.Series(False, index=stat.index)
        for seq in np.sign(stat.diff()).rolling(n-1):
            values = set(seq)
            if (len(seq) == n-1) and (len(values) == 1) and (values != {0}):
                alarms[seq.index[-1]] = True

        return alarms
    return _rule
