"""
================================================
Regression analysis (:mod:`mqr.plot.regression`)
================================================

.. currentmodule:: mqr.plot.regression

.. autosummary::
    :toctree: generated/

    residuals
    influence
    res_probplot
    res_histogram
    res_v_obs
    res_v_fit
    res_v_factor

Examples
--------
This example uses :func:`residuals` which displays :func:`res_probplot`,
:func:`res_histogram`, :func:`res_v_obs` and :func:`res_v_fit` into four supplied
axes. The example also plots th residuals against factors using :func:`res_v_factor`.
Finally, the example overlay on the `res_v_obs` plot an influence statistic using
:func:`influence`.

First set up the data and fit a model whose residuals will be shown.

.. plot::
    :context: close-figs
    :nofigs:

    import statsmodels

    # Raw data
    data = pd.read_csv(mqr.sample_data('anova-glue.csv'), index_col='Run')

    # Fit a linear model
    model = statsmodels.formula.api.ols('adhesion_force ~ C(primer) * C(glue)', data)
    result = model.fit()

Now create the residuals plots. There are six axes altogether: four for
:func:`residuals` and another two for :func:`res_v_factor` applied to each factor.

.. plot::
    :context: close-figs

    fig, axs = plt.subplots(3, 2, figsize=(8, 5), layout='constrained')

    # show the four residuals plots
    mqr.plot.regression.residuals(result.resid, result.fittedvalues, axs=axs[:2, :])

    # plot residuals against each factor
    mqr.plot.regression.res_v_factor(result.resid, data['primer'], axs[2, 0])
    mqr.plot.regression.res_v_factor(result.resid, data['glue'], axs[2, 1])

    # show Cook's Distance measure of influence.
    mqr.plot.regression.influence(result, 'cooks_dist', axs[1, 0])

"""

import numpy as np
import probscale
import statsmodels.api as sm
import scipy.stats as st
import seaborn as sns

import mqr
from mqr.plot.defaults import Defaults
from mqr.plot.lib.util import set_kws

def res_probplot(resid, ax, probplot_kws=None):
    """
    Probability plot of residuals from result of calling `fit()` on statsmodels
    model.

    Parameters
    ----------
    resid : array_like
        Residuals from regression.
    ax : matplotlib.axes.Axes
        Axes for the plot.
    probplot_kws : dict, optional
        Keyword arguments passed to `probscale.probplot`.
    """
    probplot_kws = set_kws(
        probplot_kws,
        probax='y',
        bestfit=True,
        scatter_kws=dict(
            color='C0',
            marker=Defaults.marker,
            zorder=0,
        ),
        line_kws=dict(
            color='C1',
            alpha=0.8,
            path_effects=Defaults.line_overlay_path_effects,
            zorder=1,
        ),
    )

    probscale.probplot(resid, ax=ax, **probplot_kws)
    ax.set_yticks([1, 5, 20, 50, 80, 95, 99])
    ax.set_ylabel('probability')

def res_histogram(resid, ax, show_density=True, hist_kws=None, density_kws=None):
    """
    Plot histogram of residuals from result of calling `fit()` on statsmodels
    model.

    Parameters
    ----------
    resid : array_like
        Residuals from regression.
    ax : matplotlib.axes.Axes
        Axes for the plot.
    show_density : bool, optional
        Draw a fitted normal distribution density over the histogram.
    hist_kws : dict, optional
        Keyword arguments passed to `seaborn.histplot`.
    density_kws : dict, optional
        Keyword arguments passed to `matplotlib.pyplot.plot`.
    """
    hist_kws = set_kws(
        hist_kws,
        color='C0',
        bins='doane',
        stat='count',
        zorder=0,
    )
    density_kws = set_kws(
        density_kws,
        color='C1',
        path_effects=Defaults.line_overlay_path_effects,
        zorder=1,
    )

    sns.histplot(resid, ax=ax, **hist_kws)
    if show_density:
        mean = np.mean(resid)
        std = np.std(resid, ddof=1)
        xs = np.linspace(mean-3*std, mean+3*std, 200)
        ys = mqr.plot.lib.util.scaled_density(xs, resid, st.norm, hist_kws['bins'])
        ax.plot(xs, ys, **density_kws)
    ax.set_xlabel('residual')
    ax.set_ylabel('frequency')

def res_v_obs(resid, ax, plot_kws=None, bar_kws=None):
    """
    Plot residuals versus observations.

    Parameters
    ----------
    resid : array_like
        Residuals from regression.
    ax : matplotlib.axes.Axes
        Axes for plot.
    plot_kws : dict, optional
        Keyword arguments passed to `matplotlib.pyplot.plot`.
    bar_kws : dict, optional
        Keyword arguments passed to `matplotlib.pyplot.bar`.
    """
    plot_kws = set_kws(
        plot_kws,
        color='C0',
        marker=Defaults.marker,
    )
    bar_kws = set_kws(
        bar_kws,
        alpha=0.5,
        color='C0',
        zorder=0,
    )

    if hasattr(resid, 'index'):
        index = resid.index
    else:
        index = np.arange(len(resid))

    ax.plot(index, resid, **plot_kws)
    ax.grid(axis='y')
    try:
        ax.set_xlabel(index.name)
    except:
        ax.set_xlabel('run')
    ax.set_ylabel('residual')

def res_v_fit(resid, fitted, ax, plot_kws=None):
    """
    Plot residual versus fit.

    Parameters
    ----------
    resid : array_like
        Residuals from regression.
    fitted : array_like
        Fitted values from regression.
    ax : matplotlib.axes.Axes
        Axes for plot.
    plot_kws : dict, optional
        Keyword arguments passed to `matplotlib.pyplot.plot`.
    """
    plot_kws = set_kws(
        plot_kws,
        color='C0',
        linewidth=0,
        marker=Defaults.marker,
    )

    ax.plot(fitted, resid, **plot_kws)
    ax.grid(axis='y')
    ax.set_xlabel('fitted value')
    ax.set_ylabel('residual')

def res_v_factor(resid, factor, ax, factor_ticks=True, factor_name=None, plot_kws=None):
    """
    Plot a factor versus fit.

    Parameters
    ----------
    resid : array_like
        Residuals from regression.
    factor
        Values of a factor (levels) from data.
    ax : matplotlib.axes.Axes
        Axes for plot.
    factor_ticks : bool, optional
        When `True`, uses unique values in `factor` as x-ticks.
    factor_name : str, optional
        Name of the factor to be printed as the x-axis label.
    plot_kws : dict, optional
        Keyword arguments passed to `matplotlib.pyplot.plot`.
    """
    plot_kws = set_kws(
        plot_kws,
        color='C0',
        linewidth=0,
        marker=Defaults.marker,
    )

    if factor_name is None:
        if hasattr(factor, 'name'):
            factor_name = factor.name
        else:
            factor_name = 'factor'
    ax.plot(factor, resid, **plot_kws)
    if factor_ticks:
        ax.set_xticks(factor.unique())
        ax.set_xmargin(0.2)
    ax.set_xlabel(factor_name)
    ax.set_ylabel('residual')
    ax.grid(axis='y')

def residuals(resid, fitted, axs):
    """
    Plot a probability plot of residuals, histogram of residuals, residuals
    versus observation and residuals versus fitted values for the residuals in
    a fitted statsmodels model.

    Parameters
    ----------
    resid
        Residuals from regression.
    fitted
        Fitted values from regression.
    axs : np.ndarray[matplotlib.axes.Axes]
        Array of axes for plot. Must have four elements. Will be flattened
        before use.

    Examples
    --------
    See :mod:`mqr.plot.regression`.
    """
    axs = axs.flatten()
    assert len(axs) == 4 , f'subplots must have 4 axes.'

    res_probplot(resid, ax=axs[0])
    res_histogram(resid, ax=axs[1])
    res_v_obs(resid, ax=axs[2])
    res_v_fit(resid, fitted, ax=axs[3])

def influence(result, influence_stat, ax, bar_kws=None):
    """
    Plot a bar graph of an influence statistic onto a twin axis of `ax`.

    Parameters
    ----------
    result : statsmodels.regression.linear_model.RegressionResults
        Result of calling `fit` on a statsmodel linear regression model.
    influence_stat: {'cooks_dist', 'bonferroni'}
        Plot this influence statistic for each residual as a bar.
    ax : matplotlib.axes.Axes
        Axes from which the twin axes will be created for the plot.
    bar_kws : dict, optional
        Keyword arguments passed to `matplotlib.pyplot.bar`.

    Examples
    --------
    See :mod:`mqr.plot.regression`.
    """
    bar_kws = set_kws(
        bar_kws,
        alpha=0.5,
        color='C0',
        zorder=0,
    )

    if hasattr(result.resid, 'index'):
        index = result.resid.index
    else:
        index = np.arange(len(resid))

    ax.set_ylabel('residual')

    if influence_stat == 'cooks_dist':
        p = 1 - result.get_influence().cooks_distance[1]
        label = "Cook's distance (1-p)"
    elif influence_stat == 'bonferroni':
        p = 1 - result.outlier_test().loc[:, 'bonf(p)']
        label = 'Bonferroni (1-p)'
    else:
        raise RuntimeError(f'statistic not recognised: {influence_stat}')

    axt = ax.twinx()
    axt.bar(index, p, **bar_kws)
    axt.set_ylim(0.0, 1.0)
    axt.set_ylabel(label)
