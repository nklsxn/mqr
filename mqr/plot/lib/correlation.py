"""
Correlation plot.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from mqr import inference

def matrix(data, ax, conf=0.95, show_conf=False, cmap='coolwarm'):
    """
    Plot a correlation matrix and associated statistics.

    Plots an n by n grid (where n is the number of columns in data). Each column
    and row corresponds to a column in the dataframe `data`. The lower diagonal
    shows scatter plots between the corresponding variables. The upper diagonal
    shows the statistics (1) Pearson correlation coefficient, (2) p-value for
    the correlation coefficient, and when `show_conf` is true, (3) confidence
    interval for the correlation coefficient.

    Also:
    * Whenever the p-value is less than the significance level (`1-conf`), the
    values are printed in bold.
    * When `show_conf` is true, the upper triangle is also coloured according to
    the correlation coefficient. Positive correlations are shown in blue while
    negative correlations are shown in red.

    Arguments
    ---------
    data (pd.DataFrame) -- Data with samples in columns. Calculate the
        correlations between columns.
    ax (matplotlib.axes.Axes) -- Axes for the plot.

    Optional
    --------
    conf (float) -- Confidence level of the interval.
    show_conf (bool) -- Shows confidence intervals and colours when true.
    cmap (matplotlib.colors.Colormap) -- Correlation coefficient is mapped to a
        colour in this colourmap and shown in the upper triangle.
    """
    M, N = data.shape

    assert ax.shape == (N, N) , f'Tableau shape is ({N},{N}) but ax.shape is {ax.shape}.'
    assert data.ndim == 2, '`data` must be a 2-dimensional array.'

    fig = ax[0, 0].get_figure()
    fig.set_layout_engine(None)
    fig.subplots_adjust(wspace=0, hspace=0)

    alpha = 1 - conf
    axis_names = data.columns #[name for name in study.samples]

    # Plot histograms
    for n in range(N):
        ax_hist = ax[n, n].twinx()
        sns.histplot(data.iloc[:, n], stat='count', color='lightgray', ax=ax_hist)
        ax_hist.set_ylabel(None)
        ax_hist.set_yticks([])

    colormap = mpl.colormaps[cmap]

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
                color = 'k' if p < alpha else 'gray'
                fontweight = 'bold' if p < alpha else 'normal'
            else:
                ci = None
                color = 'k'
                fontweight = 'normal'
            fontsize = 8

            sns.regplot(x=data.iloc[:, i], y=data.iloc[:, j], x_ci='ci',
                        marker='.',
                        scatter_kws={'color':color, 'alpha':0.6, 'marker':'.', 'linewidths':0},
                        line_kws={'color':color, 'alpha':0.6, 'linewidth':0.8},
                        color=color,
                        ci=ci,
                        ax=ax[j, i])

            if show_conf:
                (r, g, b, a) = colormap((1 - rho) / 2)
                ax[i, j].set_facecolor((r, g, b, a))
            ax[i, j].text(0.5, 0.5,
                          text,
                          ha='center', va='center',
                          transform=ax[i, j].transAxes,
                          fontweight=fontweight,
                          fontsize=fontsize,
                          color='k',)

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
