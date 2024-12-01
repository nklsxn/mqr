import numpy as np
import seaborn as sns

import mqr
from mqr.plot.defaults import Defaults
from mqr.plot.lib.util import set_kws

def bar_var_pct(grr_table, ax, sources=None, bar_kws=None):
    bar_kws = set_kws(
        bar_kws,
        zorder=1,
    )

    if sources is None:
        indices = ['Gauge RR', 'Repeatability', 'Reproducibility', 'Part-to-Part']
    else:
        indices = sources
    columns = ['% Contribution', '% StudyVar', '% Tolerance']

    x = np.arange(len(indices))  # the label locations
    width = 0.8 / len(columns)  # the width of the bars
    pct_data = grr_table.table.loc[indices, columns]
    for i, row in enumerate(pct_data.items()):
        offset = width * (i - 1)
        rects = ax.bar(x + offset, row[1], width, label=row[0], **bar_kws)

    ax.legend(
        [c for c in columns],
        prop={'size': 8},
        fancybox=True,
        bbox_to_anchor=(1.02, 0.5, 0.0, 0.0),
        loc='center left',
        borderaxespad=0.0)
    ax.set_xticks(x)
    ax.set_xticklabels(indices)
    for label in ax.get_xticklabels():
        label.set_rotation(15)
        label.set_ha('right')
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Components of Variation')
    ax.grid()

def box_measurement_by_part(grr, ax, box_kws=None, line_kws=None):
    box_kws = set_kws(
        box_kws,
        color='C0',
        fill=False,
        width=0.4,
    )
    line_kws = set_kws(
        line_kws,
        color='C0',
        marker=Defaults.marker,
    )

    name_p = grr.names.part
    name_m = grr.names.measurement

    sns.boxplot(grr.data, x=name_p, y=name_m, ax=ax, **box_kws)

    names = grr.data[name_p].unique().astype('str')
    means = grr.data.groupby(name_p)[name_m].mean()
    ax.plot(names, means, **line_kws)

    ax.set_xlabel(name_p)
    ax.set_ylabel(name_m)
    ax.set_title(f'{name_m} by {name_p}')

    ax.grid()

def box_measurement_by_operator(grr, ax, box_kws=None, line_kws=None):
    box_kws = set_kws(
        box_kws,
        color='C0',
        fill=False,
        width=0.4,
    )
    line_kws = set_kws(
        line_kws,
        color='C0',
        marker=Defaults.marker,
    )

    name_o = grr.names.operator
    name_m = grr.names.measurement

    sns.boxplot(grr.data, x=name_o, y=name_m, ax=ax, **box_kws)

    names = grr.data[name_o].unique().astype('str')
    means = grr.data.groupby(name_o)[name_m].mean()
    ax.plot(names, means, **line_kws)

    ax.set_xlabel(name_o)
    ax.set_ylabel(name_m)
    ax.set_title(f'{name_m} by {name_o}')

    ax.grid()

def line_part_operator_intn(grr, ax, line_kws=None):
    line_kws = set_kws(
        line_kws,
        marker='_',
    )

    name_p = grr.names.part
    name_o = grr.names.operator
    name_m = grr.names.measurement
    name_r = grr.names.replicate

    intn = mqr.anova.interactions(grr.data, value=name_m, between=[name_p, name_o])

    ax.plot(intn, **line_kws)
    ax.set_xticks(intn.index)
    ax.set_xlabel(name_p)

    ax.grid()
    ax.legend(
        intn.columns,
        title=name_o,
        prop={'size': 8},
        fancybox=True,
        bbox_to_anchor=(1.02, 0.5, 0.0, 0.0),
        loc='center left',
        borderaxespad=0.0)

    ax.set_ylabel(f'{name_m}\n(mean over {name_r})')
    ax.set_title(f'{name_p} * {name_o} Interaction')

def xbar_operator(grr, ax,
                  line_kws=None, text_kws=None,
                  target_kws=None, control_kws=None):
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
        zorder=0,
    )

    # Plot sample mean per operator
    name_p = grr.names.part
    name_o = grr.names.operator
    name_m = grr.names.measurement
    name_r = grr.names.replicate
    grp = grr.data.groupby([name_p, name_o])[name_m]
    mqr.plot.grouped_df(
        grp.mean().unstack(),
        ax=ax,
        line_kws=line_kws,
        text_kws=text_kws)

    # Add control bars
    if not np.all(grp.count() == grp.count().iloc[0]):
        raise ValueError('Only balanced experiments are supported.')

    N = grp.count().iloc[0]
    params = mqr.spc.XBarParams.from_range(
        np.mean(grr.data[name_m]),
        grp.apply(np.ptp).mean(),
        N,)

    ax.axhline(params.target(), **target_kws)
    ax.axhline(params.ucl(N), **control_kws)
    ax.axhline(params.lcl(N), **control_kws)

    ax.set_xlabel(name_p)
    ax.set_ylabel(f'{name_m}\n(mean over {name_r})')
    ax.set_title(f'Xbar chart by {name_o}')

def r_operator(grr, ax,
               line_kws=None, text_kws=None,
               target_kws=None, control_kws=None):
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
        zorder=0,
    )

    # Plot sample range per operator
    name_p = grr.names.part
    name_o = grr.names.operator
    name_m = grr.names.measurement
    name_r = grr.names.replicate
    grp = grr.data.groupby([name_p, name_o])[name_m]
    range_r = grp.apply(np.ptp).unstack()
    mqr.plot.grouped_df(
        range_r,
        ax=ax,
        line_kws=line_kws,
        text_kws=text_kws)

    # Add control bars
    if not np.all(grp.count() == grp.count().iloc[0]):
        raise ValueError('Only balanced experiments are supported.')

    N = grp.count().iloc[0]
    rbar = np.mean(range_r)
    params = mqr.spc.RParams.from_range(rbar, N)

    ax.axhline(params.target(), **target_kws)
    ax.axhline(params.ucl(N), **control_kws)
    ax.axhline(params.lcl(N), **control_kws)
    ax.set_xlabel(name_p)
    ax.set_ylabel(f'{name_m}\n(mean over {name_r})')
    ax.set_title(f'Range by {name_o}')

def grr(grr, axs, sources=None):
    axs = axs.flatten()
    assert len(axs) == 6, 'GRR Tableau requires 6 subplot axes.'
    grr_table = mqr.msa.VarianceTable(grr)

    bar_var_pct(grr_table, sources=sources, ax=axs[0])
    box_measurement_by_part(grr, ax=axs[1])
    xbar_operator(grr, ax=axs[2])
    box_measurement_by_operator(grr, ax=axs[3])
    r_operator(grr, ax=axs[4])
    line_part_operator_intn(grr, ax=axs[5])