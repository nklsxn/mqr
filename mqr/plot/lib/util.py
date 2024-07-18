"""
Plot dataframe columns adjacent to one another.

Commonly used to present measurement system analysis results.
"""

import matplotlib.transforms as transforms
import numpy as np

def grouped_df(data, ax):
    """
    Plots from a pandas.DataFrame with columns grouped along the x-axis.

    Arguments
    ---------
    data (pd.DataFrame) -- Dataframe whose columns will be plot next to each other.
    ax (matplotlib.axes.Axes) -- Axes for plot.
    """

    idx_labels = data.index
    idx = np.arange(len(idx_labels))
    M = len(idx)

    groups = data.columns
    N = len(groups)

    tr = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for i, g in enumerate(groups):
        xs = idx + i * M
        ax.plot(xs, data.loc[:, g],
            color='C0', linewidth=0.8, marker='o', markersize=5,
            label=g)
        sep_x = i * M - 0.5
        ax.axvline(sep_x, linewidth=0.8, color='gray', linestyle=(0, (5, 5)))
        ax.text(
            sep_x+0.25, 0.99,
            g,
            transform=tr,
            rotation=0, rotation_mode='anchor', va='top', ha='left')

    ax.set_xticks(np.arange(len(idx)*N))
    ax.set_xticklabels(np.tile(idx_labels, N), rotation=90)
