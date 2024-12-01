import numpy as np
import probscale
import statsmodels.api as sm
import scipy.stats as st
import seaborn as sns

import mqr
from mqr.plot.defaults import Defaults
from mqr.plot.lib.util import set_kws

def res_probplot(resid, ax, probplot_kws=None):
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
    axs = axs.flatten()
    assert len(axs) == 4 , f'subplots must have 4 axes.'

    res_probplot(resid, ax=axs[0])
    res_histogram(resid, ax=axs[1])
    res_v_obs(resid, ax=axs[2])
    res_v_fit(resid, fitted, ax=axs[3])

def influence(result, influence_stat, ax, bar_kws=None):
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
