import mqr
from mqr.plot.defaults import Defaults
from mqr.plot.lib.util import set_kws

import numpy as np
import scipy.stats as st

def chart(control_statistic, control_params, ax, *,
          line_kws=None, in_kws=None, out_kws=None, target_kws=None, control_kws=None):

    line_kws = set_kws(
        line_kws,
        marker=Defaults.marker,
        color='C0',
        zorder=1,
    )
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
        drawstyle='steps-mid',
        zorder=0,
    )

    stat = control_statistic.stat
    nobs = control_statistic.nobs
    index = stat.index
    target = control_params.target()
    lcl = control_params.lcl(nobs)
    ucl = control_params.ucl(nobs)

    line_values = [
        target,
        lcl.iloc[-1] if lcl is not None else None,
        ucl.iloc[-1] if ucl is not None else None,]
    yticks = [tick for tick in line_values if tick is not None]
    yticklabels = [
        f'{label}={value:g}'
        for label, value in zip(['target', 'LCL', 'UCL'], line_values)
        if value is not None]

    ax.plot(stat, **line_kws)
    ax.axhline(target, **target_kws)
    if lcl is not None:
        ax.plot(lcl, **control_kws)
    if ucl is not None:
        ax.plot(ucl, **control_kws)
    sec = ax.secondary_yaxis('right')
    sec.set_yticks(yticks)
    sec.set_yticklabels(yticklabels)
    ax.set_ymargin(0.15)
    ax.set_title(f'{control_params.name} chart')
    ax.set_xticks(index)

def alarms(control_statistic, control_params, control_rule, ax, *,
           point_kws=None, span_kws=None):

    point_kws = set_kws(
        point_kws,
        linewidth=0,
        color='C3',
        marker='.',
    )
    span_kws = set_kws(
        span_kws,
        color='C3',
        alpha=0.2,
        zorder=-1,
    )

    stat = control_statistic.stat
    alarms = control_rule(control_statistic, control_params)
    for alarm in mqr.spc.util.alarm_subsets(alarms):
        ax.plot(stat.loc[alarm], **point_kws)
        a = alarm[0] - 0.5
        b = alarm[-1] + 0.5
        ax.axvspan(a, b, **span_kws)

def oc(n, c, ax, defect_range=None, line_kws=None):
    line_kws = set_kws(
        line_kws,
        color='C0',
    )

    if defect_range is None:
        defect_range = (0, 1)

    ps = np.linspace(*defect_range)

    pa = np.empty(ps.shape)
    for i in range(len(ps)):
        pa[i] = st.binom(n, ps[i]).cdf(c)

    ax.plot(ps, pa, **line_kws)
    ax.grid()
    ax.set_xlabel('Defect rate')
    ax.set_ylabel('Prob of Acceptance')
