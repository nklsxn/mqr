"""
============================================
Analysis of variance (:mod:`mqr.plot.anova`)
============================================

.. currentmodule:: mqr.plot.anova

.. rubric:: Functions
.. autosummary::
    :toctree: generated/

    main_effects
    interactions
    model_means
"""

from collections.abc import Iterable
import numpy as np

import mqr
from mqr.plot.defaults import Defaults
from mqr.plot.lib.util import set_kws

def groups(groups_df, ax, ci_kws=None):
    """
    Draw a bar graph with error bars for the groups in an ANOVA.

    Parameters
    ----------
    ci : :class:`mqr.confint.ConfidenceInterval`
        The confidence interval to draw.
    ax : matplotlib.axes.Axes
        Axes for the plot.
    """
    ci_kws = set_kws(
        ci_kws,
        fmt=Defaults.marker,
        capsize=4.0,
    )

    y_err = (groups_df.iloc[:, -1] - groups_df.iloc[:, -2]) / 2
    ax.errorbar(
        x=groups_df.index,
        y=groups_df['mean'],
        yerr=y_err,
        **ci_kws)
    ax.set_xticks(groups_df.index)
    ax.set_xmargin(0.2)
    ax.grid(alpha=0.5)

def main_effects(data, response, factors, *, axs, line_kws=None, mean_kws=None):
    """
    Plot the main effects from experimental data.

    Parameters
    ----------
    data : pandas.DataFrame
    response : str
    factors : list[str]
    axs : array_like[matplotlib.axes.Axes]
        Axes for the plot. Must have the same number of elements as `factors`.
        Will be flattened before use.
    line_kws : dict, optional
        Keyword args for the effect lines.
        Passed to :func:`matplotlib.pyplot.plot`.
    mean_kws : dict, optional
        Keyword args for the overall average line.
        Passed to :func:`matplotlib.pyplot.axhline`.

    Examples
    --------
    This example loads sample data and shows a main effects plot for the two factors.

    .. plot::

        data = pd.read_csv(mqr.sample_data('anova-glue.csv'), index_col='Run')

        fig, axs = plt.subplots(1, 2, figsize=(5, 2), layout='constrained')
        mqr.plot.tools.sharey(fig, axs)
        mqr.plot.anova.main_effects(
            data,
            response='adhesion_force',
            factors=['primer', 'glue'],
            axs=axs)

    """
    axs = axs.flatten()
    if len(factors) != len(axs):
        raise ValueError('Number of axes must equal the length of `factors`.')

    line_kws = set_kws(
        line_kws,
        marker=Defaults.marker,
    )
    mean_kws = set_kws(
        mean_kws,
        linestyle='--',
        color='k',
        alpha=0.6,
    )

    mean = data[response].mean()
    axs[0].set_ylabel(response)
    for ax, factor in zip(axs, factors):
        grp = data.groupby(factor)[response].mean()
        ax.plot(grp, **line_kws)
        ax.axhline(mean, **mean_kws)
        ax.set_xticks(data[factor].unique())
        ax.set_xmargin(0.2)
        ax.set_xlabel(factor)

def interactions(data, response, group, factors, *, axs, line_kws=None, mean_kws=None):
    """
    Interaction plot for observation `obs` between the two categories in `between`.
    
    Parameters
    ----------
    data : pd.DataFrame
        Table of categorical data.
    response : str
        Column name that contains response (real numbers).
    group : str
        Categorical factor whose values are shown as separate lines on each plot.
    factors : list[str]
        Categorical factors whose interactions with `group` are plot on each axis.
    axs : matplotlib.axes.Axes, list[matplotlib.axes.Axes]
        Axes for the plot. Must match the length of `factors` (if the length of
        `factors` is 1, then `axs` can be an axes object).
    line_kws : dict, optional
        Keyword arguments for the lines. Passed to :func:`matplotlib.pyplot.plot`.
    mean_kws : dict, optional
        Keyword arguments for the mean line.
        Passed to :func:`matplotlib.pyplot.axhline`.

    Examples
    --------
    This example loads sample data and shows an interaction plot for the two factors.
    Since one factor is used for grouping, there is one factor left for plots.

    .. plot::

        data = pd.read_csv(mqr.sample_data('anova-glue.csv'), index_col='Run')

        fig, axs = plt.subplots(figsize=(3, 2), layout='constrained')
        mqr.plot.anova.interactions(
            data,
            response='adhesion_force',
            group='primer',
            factors=['glue'],
            axs=axs)

    """

    if not isinstance(axs, Iterable):
        axs = np.array([axs])

    axs = axs.flatten()
    if len(factors) != len(axs):
        raise ValueError('Number of axes must equal the length of `factors`.')

    line_kws = set_kws(
        line_kws,
        marker=Defaults.marker,
    )
    mean_kws = set_kws(
        mean_kws,
        linestyle='--',
        color='k',
        alpha=0.6,
    )

    mean = data[response].mean()
    axs[0].set_ylabel(response)
    levels = data[group].unique()
    for ax, factor in zip(axs, factors):
        for level in levels:
            slice = data[group] == level
            values = data[slice].groupby(factor)[response].mean()
            ax.plot(values, **line_kws)
        ax.axhline(mean, **mean_kws)
        ax.set_xticks(data[factor].unique())
        ax.set_xmargin(0.2)
        ax.set_xlabel(factor)
        ax.legend(levels, title=group)

def model_means(result, response, factors, axs, ci_kws=None):
    """
    Plot the means of each combination of factor levels in an ANOVA result.

    Parameters
    ----------
    result : statsmodels.regression.linear_model.RegressionResults
        Result of calling `fit` on a statsmodel linear regression model.
    response : str
        Name of response variable.
    factors : list[str]
        List of names of categorical factors.
    axs : array_like[matplotlib.axes.Axes]
        Axes for the plot. Must have the same number of elements as `factors`.
        Will be flattened before use.
    ci_kws: dict, optional
        Keyword args for confidence intervals.
        Passed to :func:`matplotlib..errorbar <matplotlib.pyplot.errorbar>`.

    Examples
    --------
    This example performs an ANOVA on sample data, then shows the means of the
    factors.

    .. plot::
        :include-source:

        from statsmodels.formula.api import ols

        data = pd.read_csv(mqr.sample_data('anova-glue.csv'), index_col='Run')
        model = ols('adhesion_force ~ C(primer) + C(glue)', data=data)
        result = model.fit()

        fig, axs = plt.subplots(1, 2, figsize=(4, 2), layout='constrained')
        mqr.plot.tools.sharey(fig, axs)
        mqr.plot.anova.model_means(
            result,
            response='adhesion_force',
            factors=['primer', 'glue'],
            axs=axs)

    """
    axs = axs.flatten()
    if len(axs) != len(factors):
        raise ValueError('Length of `axs` must equal length of `factors`.')

    for ax, factor in zip(axs, factors):
        df_grp = mqr.anova.groups(result, value=response, factor=factor, formatted=False)
        mqr.plot.anova.groups(df_grp, ax, ci_kws)
        ax.set_xlabel(factor)
