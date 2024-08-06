def groups(groups_df, ax):
    """
    Draw a bar graph with error bars for the groups in an ANOVA.

    Arguments
    ---------
    ci (mqr.confint.ConfidenceInterval) -- The confidence interval to draw.
    ax (matplotlib.axes.Axes) -- Axes for the plot.
    """
    y_err = groups_df.iloc[:, -1] - groups_df.iloc[:, -2]
    ax.bar(x=groups_df.index, height=groups_df['mean'], width=0.5)
    ax.errorbar(x=groups_df.index, y=groups_df['mean'], yerr=y_err, color='k', fmt='none')

