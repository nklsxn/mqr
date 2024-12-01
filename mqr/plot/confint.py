import matplotlib.pyplot as plt
import numpy as np

from mqr.inference.confint import ConfidenceInterval
from mqr.plot.defaults import Defaults
from mqr.plot.lib.util import set_kws

def confint(ci: ConfidenceInterval, ax, hyp_value=None, interval_kws=None, mean_kws=None, hyp_kws=None):
    interval_kws = set_kws(
        interval_kws,
        color='C0',
    )
    mean_kws = set_kws(
        mean_kws,
        marker=Defaults.marker,
        color='C0',
    )
    hyp_kws = set_kws(
        hyp_kws,
        marker=Defaults.marker,
        color='C1',
    )

    y_loc = 0

    l, r = ax.get_xlim()
    l_mkr = '<'
    r_mkr = '>'
    if np.isfinite(ci.lower):
        l = ci.lower
        l_mkr = '|'
    if np.isfinite(ci.upper):
        r = ci.upper
        r_mkr = '|'

    ax.plot(l, y_loc, marker=l_mkr, **interval_kws) # left marker
    ax.plot(r, y_loc, marker=r_mkr, **interval_kws) # right marker
    ax.plot([l, r], [y_loc, y_loc], **interval_kws) # interval line
    ax.plot(ci.value, y_loc, **mean_kws)
    if hyp_value is not None:
        ax.plot(hyp_value, y_loc, **hyp_kws)

    ax.set_yticks([y_loc])
    ax.set_yticklabels([ci.name])
    ax.tick_params(axis='y', left=False)
    ax.set_ylim(y_loc-1, y_loc+1)
