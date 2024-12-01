from collections.abc import Iterable
import numpy as np

import mqr
from mqr.plot.defaults import Defaults
from mqr.plot.lib.util import set_kws

def groups(groups_df, ax, ci_kws=None):
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
    axs = axs.flatten()
    if len(axs) != len(factors):
        raise ValueError('Length of `axs` must equal length of `factors`.')

    for ax, factor in zip(axs, factors):
        df_grp = mqr.anova.groups(result, value=response, factor=factor)
        mqr.plot.anova.groups(df_grp, ax, ci_kws)
        ax.set_xlabel(factor)
