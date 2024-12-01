from dataclasses import dataclass
import numpy as np

import matplotlib as mpl
import matplotlib.patheffects as pe
from cycler import cycler

@dataclass
class Defaults:
    """
    Default values for plotting libraries.

    These defaults are loaded automatically each time the `mqr.plot.Figure`
    context is entered.

    Attributes
    ----------
    rc_params : dict
        Paramters passed to `matplotlib` as defaults for all plots.
    marker : str
        The marker to use when a line plot is drawn with markers.
    line_overlay_path_effects : list[:class:`matplotlib.patheffects.AbstractPathEffect`]
        Path effects that are used to foreground a line on busy or dense plots.
    ishikawa : dict
        Default settings for Ishikawa diagrams. See :func:`mqr.plot.ishikawa`.

    """
    rc_params = {
        'axes.formatter.useoffset': False,
        'axes.prop_cycle': cycler(color=[
            '#4e79a7', # blue
            '#f28e2b', # orange
            '#59a14f', # green
            '#e15759', # red
            '#b070ac', # purple
            '#9c755f', # brown
            '#ff9da7', # pink/salmon
            '#bab0ac', # grey
            '#edc948', # yellow
            '#76b7b2', # turquoise
        ]),
        'lines.linewidth': 1.2,
    }

    # General
    marker = 'o'
    line_overlay_path_effects = [
            pe.Stroke(linewidth=3, alpha=0.5, foreground='w'),
            pe.Normal()]

    # Ishikawa diagrams
    ishikawa = dict(
        head_space=2.0,
        bone_space=8.0, # NB: niklsxn: should be auto based on widest text
        bone_angle=np.pi/4,
        cause_space=1.0,
        cause_length=2.0,
        padding=(1.0, 1.0, 1.0, 1.0),
        line_kwargs=dict(linewidth=0.8, color='k'),
        defect_font_dict=dict(fontsize=12, weight='bold'),
        cause_font_dict=dict(fontsize=9, weight='bold'),
        primary_font_dict=dict(fontsize=9),
    )
