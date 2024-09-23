import numpy as np

def groups(groups_df, ax):
    """
    Draw a bar graph with error bars for the groups in an ANOVA.

    Arguments
    ---------
    ci (mqr.confint.ConfidenceInterval) -- The confidence interval to draw.
    ax (matplotlib.axes.Axes) -- Axes for the plot.
    """
    y_err = (groups_df.iloc[:, -1] - groups_df.iloc[:, -2]) / 2
    ax.errorbar(
        x=groups_df.index,
        y=groups_df['mean'],
        yerr=y_err,
        fmt='o', linewidth=1.0, capsize=4.0)
    ax.set_xticks(groups_df.index)
    ticks = ax.get_xticks()
    delta = np.mean(np.diff(ticks))
    bounds = ticks[[0, -1]]
    ax.set_xlim(bounds + np.array([-delta/2, delta/2]))
    ax.grid(alpha=0.5)
