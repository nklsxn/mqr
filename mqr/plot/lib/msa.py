"""
Plots of the results of measurement system analysis (MSA).
"""

import numpy as np
import seaborn as sns

import mqr

def bar_var_pct(grr, ax, sources=None):
    """
    Bar graph of percent contributions from `sources` in a GRR study.

    Arguments
    ---------
    grr (mqr.msa.GRR) -- Results from a GRR study.
    ax (matplotlib.axes.Axes) -- Axes for the plot.

    Optional
    --------
    sources (list[str]) -- Sources to include. Defaults to `["% Contribution",
        "% StudyVar", "% Tolerance"]`.
    """
    if sources is None:
        indices = ['Gauge RR', 'Repeatability', 'Reproducibility', 'Part-to-Part']
    else:
        indices = sources
    columns = ['% Contribution', '% StudyVar', '% Tolerance']

    pct_data = grr.grr_table.loc[indices, columns]
    pct_data.plot(kind='bar', rot=0, ax=ax)
    ax.legend(
        [c for c in columns],
        prop={'size': 8},
        fancybox=True,
        bbox_to_anchor=(1.02, 0.5, 0.0, 0.0),
        loc='center left',
        borderaxespad=0.0)

    ax.set_ylabel('Percentage (%)')
    ax.set_title('Components of Variation')
    ax.grid()

def box_measurement_by_part(grr, ax):
    """
    Box plot showing spread in a GRR by part.

    Arguments
    ---------
    grr (mqr.msa.GRR) -- Results from a GRR study.
    ax (matplotlib.axes.Axes) -- Axes for the plot.
    """
    name_p = grr.names.part
    name_m = grr.names.measurement

    sns.boxplot(grr.data, x=name_p, y=name_m, width=0.4, ax=ax)
    
    means = grr.data.groupby(name_p)[name_m].mean()
    ax.plot(grr.data[name_p].unique().astype('str'), means, linewidth=0.8)

    ax.set_xlabel(name_p)
    ax.set_ylabel(name_m)
    ax.set_title(f'{name_m} by {name_p}')

    ax.grid()

def box_measurement_by_operator(grr, ax):
    """
    Box plot showing spread in a GRR by operator.

    Arguments
    ---------
    grr (mqr.msa.GRR) -- Results from a GRR study.
    ax (matplotlib.axes.Axes) -- Axes for the plot.
    """
    name_o = grr.names.operator
    name_m = grr.names.measurement

    sns.boxplot(grr.data, x=name_o, y=name_m, width=0.4, ax=ax)

    means = grr.data.groupby(name_o)[name_m].mean()
    ax.plot(grr.data[name_o].unique().astype('str'), means, linewidth=0.8)

    ax.set_xlabel(name_o)
    ax.set_ylabel(name_m)
    ax.set_title(f'{name_m} by {name_o}')

    ax.grid()

def interaction(grr, ax):
    """
    Interaction plot showing the part-operator interaction in a GRR study.

    Arguments
    ---------
    grr (mqr.msa.GRR) -- Results from a GRR study.
    ax (matplotlib.axes.Axes) -- Axes for the plot.
    """
    name_p = grr.names.part
    name_o = grr.names.operator
    name_m = grr.names.measurement
    name_r = grr.names.replicate
    
    mqr.plot.regression.interaction(
        grr.data,
        obs=name_m, between=(name_p, name_o),
        ax=ax)

    ax.set_ylabel(f'{name_m}\n(mean over {name_r})')
    ax.set_title(f'{name_p} * {name_o} Interaction')

def xbar_operator(grr, ax):
    """
    XBar-chart for the operator mean in a GRR study.

    Arguments
    ---------
    grr (mqr.msa.GRR) -- Results from a GRR study.
    ax (matplotlib.axes.Axes) -- Axes for the plot.
    """
    # Plot sample mean per operator
    name_p = grr.names.part
    name_o = grr.names.operator
    name_m = grr.names.measurement
    name_r = grr.names.replicate
    grp = grr.data.groupby([name_p, name_o])[name_m]
    mqr.plot.grouped_df(
        grp.mean().unstack(),
        ax=ax)

    # Add control bars
    ybar = np.mean(grr.data[name_m])
    Rbar = (grp.max() - grp.min()).mean()
    N = grr.data[name_r].nunique()
    ax.axhline(ybar, color='green', linewidth=0.8, alpha=0.8, zorder=0)
    ax.axhline(
        mqr.control.xbar_ucl(ybar, Rbar, N),
        color='red', linewidth=0.8, alpha=0.8, zorder=0)
    ax.axhline(
        mqr.control.xbar_lcl(ybar, Rbar, N),
        color='red', linewidth=0.8, alpha=0.8, zorder=0)

    ax.set_xlabel(name_p)
    ax.set_ylabel(f'{name_m}\n(mean over {name_r})')
    ax.set_title(f'Xbar chart by {name_o}')

def r_operator(grr, ax):
    """
    R chart for the operator range in a GRR study.

    Arguments
    ---------
    grr (mqr.msa.GRR) -- Results from a GRR study.
    ax (matplotlib.axes.Axes) -- Axes for the plot.
    """
    # Plot sample range per operator
    name_p = grr.names.part
    name_o = grr.names.operator
    name_m = grr.names.measurement
    name_r = grr.names.replicate
    grp = grr.data.groupby([name_p, name_o])[name_m]
    range_r = (grp.max() - grp.min()).unstack()
    mqr.plot.grouped_df(
        range_r,
        ax=ax)

    # Add control bars
    rbar = np.mean(range_r)
    N = grr.data[name_r].nunique()
    ax.axhline(rbar, color='green', linewidth=0.8, alpha=0.8, zorder=0)
    ax.axhline(mqr.control.rbar_ucl(rbar, N), color='red', linewidth=0.8, alpha=0.8, zorder=0)
    ax.axhline(mqr.control.rbar_lcl(rbar, N), color='red', linewidth=0.8, alpha=0.8, zorder=0)

    ax.set_xlabel(name_p)
    ax.set_ylabel(f'{name_m}\n(mean over {name_r})')
    ax.set_title(f'Range by {name_o}')

def grr(grr, axs, sources=None):
    """
    GRR summary plots.

    A 3 by 2 grid of:
    - bar graph of components of variation,
    - measurement by part,
    - R-chart by operator,
    - measurement by operator,
    - Xbar-chart by operator, and
    - part * operator interaction.

    Arguments
    ---------
    grr (mqr.msa.GRR) -- GRR study.
    axs (numpy.ndarray) -- A 3*2 array of matplotlib axes.

    Optional
    --------
    sources (list[str]) -- A list of components of variation to include in the bar graph (optional).
    """
    axs = axs.flatten()
    assert len(axs) == 6, 'GRR Tableau requires 6 subplot axes.'

    bar_var_pct(grr, sources=sources, ax=axs[0])
    box_measurement_by_part(grr, ax=axs[1])
    xbar_operator(grr, ax=axs[2])
    box_measurement_by_operator(grr, ax=axs[3])
    r_operator(grr, ax=axs[4])
    interaction(grr, ax=axs[5])
