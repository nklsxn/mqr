import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from mqr import inference
from mqr.plot.defaults import Defaults
from mqr.plot.lib.util import set_kws

def matrix(data, ax, conf=0.95, fig=None, show_conf=False, cmap='RdBu',
           hist_kws=None, reg_kws=None, scatter_kws=None, line_kws=None, text_kws=None):
    M, N = data.shape

    assert ax.shape == (N, N) , f'Tableau shape is ({N},{N}) but ax.shape is {ax.shape}.'
    assert data.ndim == 2, '`data` must be a 2-dimensional array.'

    colormap = mpl.colormaps[cmap]

    hist_kws = set_kws(
        hist_kws,
        color='C0',
        stat='count',
    )
    reg_kws = set_kws(
        reg_kws,
        marker='.',
    )
    scatter_kws = set_kws(
        scatter_kws,
        color='C0',
    )
    line_kws = set_kws(
        line_kws,
        color='C0',
    )
    text_kws = set_kws(
        text_kws,
        color='k',
        fontsize=8,
        ha='center',
        va='center',
    )

    if fig is None:
        fig = ax[0, 0].get_figure()

    fig.set_layout_engine(None)
    fig.subplots_adjust(wspace=0, hspace=0)

    alpha = 1 - conf
    axis_names = data.columns #[name for name in study.samples]

    # Plot histograms
    for n in range(N):
        ax_hist = ax[n, n].twinx()
        sns.histplot(data.iloc[:, n], ax=ax_hist, **hist_kws)
        ax_hist.set_ylabel(None)
        ax_hist.set_yticks([])


    for i in range(N):
        for j in range(N):
            ax[i, j].set_xlabel(None)
            ax[i, j].set_ylabel(None)
            if i >= j: continue

            if show_conf:
                ci = inference.correlation.confint(
                    x=data.iloc[:, i],
                    y=data.iloc[:, j],
                    conf=conf)
            test = inference.correlation.test(
                x=data.iloc[:, i],
                y=data.iloc[:, j])
            rho = test.sample_stat_value
            p = test.pvalue

            text = f'r={rho:.2f}\n(p={p:.2f})'
            if show_conf:
                text += f'\n\n{ci.conf*100:.0f}% CI:\n[{ci.lower:.2f}, {ci.upper:.2f}]'
                ci = 100 * conf
                plot_alpha = 0.6 if p < alpha else 0.2
                fontweight = 'bold' if p < alpha else 'normal'
            else:
                ci = None
                plot_alpha = 0.6
                fontweight = 'normal'

            scatter_kws |= dict(alpha=plot_alpha, linewidths=0)
            line_kws |= dict(alpha=plot_alpha)
            text_kws |= dict(fontweight=fontweight)

            if show_conf:
                (r, g, b, a) = colormap((1 + rho) / 2)
                ax[i, j].set_facecolor((r, g, b, a))
            ax[i, j].text(
                0.5, 0.5,
                text,
                transform=ax[i, j].transAxes,
                **text_kws)

            sns.regplot(
                x=data.iloc[:, i],
                y=data.iloc[:, j],
                x_ci='ci', ci=ci,
                ax=ax[j, i],
                scatter_kws=scatter_kws,
                line_kws=line_kws,
                **reg_kws)

    # Share axes along rows and down columns ...
    for i in range(1, N):
        for j in range(1, N):
            ax[i, j].sharey(ax[i, 0]) # share y along rows

    for j in range(0, N-1):
        for i in range(1, N):
            ax[N-i-1, j].sharex(ax[N-1, j]) # share x axis up columns

    # ... turn off ticks and labels to start.
    for i in range(N-1):
        for j in range(N):
            plt.setp(ax[i, j].get_xticklabels(), visible=False)
            ax[i, j].tick_params(axis='x', bottom=False)
    for i in range(N):
        for j in range(1, N):
            plt.setp(ax[i, j].get_yticklabels(), visible=False)
            ax[i, j].tick_params(axis='y', left=False)
    plt.setp(ax[0, 0].get_yticklabels(), visible=False)
    ax[0, 0].tick_params(axis='y', left=False)

    # Rotate the tick labels along the bottom.
    for j in range(N):
        # ... along the bottom ...
        ax[N-1, j].tick_params(axis='x', rotation=90)

    # Set labels.
    for i in range(N):
        ax[i, N-1].set_ylabel(axis_names[i])
        ax[i, N-1].yaxis.set_label_position('right')

    for j in range(N):
        ax[0, j].set_xlabel(axis_names[j])
        ax[0, j].xaxis.set_label_position('top')
