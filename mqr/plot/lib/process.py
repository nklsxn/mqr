"""
Plots for describing processes and capability.
"""

from mqr.process import Specification, Capability, Process
from mqr.summary import Sample

import matplotlib.transforms as transforms
import numpy as np
import scipy.stats as st
import seaborn as sns

def pdf(sample: Sample, spec: Specification, capability: Capability, ax,
            nsigma=6, show_long_term=False):
    """
    Plots a Gaussian PDF of the given data.
    
    Arguments
    ---------
    sample (mqr.summary.Sample) -- Sample to plot.
    spec (mqr.process.Specification) -- Specification for the process sample.
    capability (mqr.process.Capability) -- Capability of the process.
    ax (matplotlib.axes.Axes) -- Axes for the plot.

    Optional
    --------
    nsigma (float) -- How many stddevs of the Gaussian PDF to plot (total).
        Default 6.
    show_long_term (bool) -- Plots a second Gaussian with 1.5 times more stddev.
        Default False.
    """
    dist = st.norm(sample.mean, sample.std)
    xmin_st = sample.mean - nsigma * sample.std / 2
    xmax_st = sample.mean + nsigma * sample.std / 2
    xs_st = np.linspace(xmin_st, xmax_st, 250)
    ys_st = dist.pdf(xs_st)
    
    ax.plot(xs_st, ys_st, color='k', linewidth=1.5)
    ax.plot(xs_st[0], ys_st[0], color='k', marker='.')
    ax.plot(xs_st[-1], ys_st[-1], color='k', marker='.')
    ax.axvline(sample.mean, color='k', linewidth=1.5, linestyle=(0, (5, 5)))
    
    if show_long_term:
        dist_lt = st.norm(sample.mean, 1.5*sample.std)
        xmin_lt = sample.mean - 1.5 * nsigma * sample.std / 2
        xmax_lt = sample.mean + 1.5 * nsigma * sample.std / 2
        xs_lt = np.linspace(xmin_lt, xmax_lt, 250)
        ys_lt = dist_lt.pdf(xs_lt)

        ax.plot(xs_lt, ys_lt, color='k', linewidth=1, alpha=0.5)
        ax.plot(xs_lt[0], ys_lt[0], color='k', marker='.', alpha=0.5)
        ax.plot(xs_lt[-1], ys_lt[-1], color='k', marker='.', alpha=0.5)
    
    ax.set_xlabel(f'{sample.name} (cp={capability.cp:#.3g}, cpk={capability.cpk:#.3g})')
    ax.set_ylabel('density')

def tolerance(spec: Specification, ax, prec=3):
    """
    Plots tolerance region.

    Arguments
    ---------
    spec (mqr.process.Specification) -- Spec containing tolerance bounds.
    ax (matplotlib.axes.Axes) -- Axes for the plot.

    Optional
    --------
    prec (int) --- Significant figures for the limit bounds. Default 3.
    """
    linestyle = (0, (5, 5))
    ax.axvline(spec.target, color='C0', linestyle=linestyle, linewidth=1.5)
    ax.axvline(spec.lsl, color='C0', linewidth=0.5)
    ax.axvline(spec.usl, color='C0', linewidth=0.5)
    ax.axvspan(spec.lsl, spec.usl, color='C0', alpha=0.2, zorder=0)

    tr = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(
        spec.lsl, 0.5,
        f'LSL = {spec.lsl:#.{prec}g}',
        transform=tr,
        rotation=90, rotation_mode='anchor', va='baseline', ha='center')
    ax.text(
        spec.usl, 0.5,
        f'USL = {spec.usl:#.{prec}g}',
        transform=tr,
        rotation=90, rotation_mode='anchor', va='baseline', ha='center')

def capability(process: Process, name: str, ax,
        show_long_term=False, pdf_kwargs={}):
    """
    Plots all three of the processes histogram, fitted normal distribution, and
    tolerance region for the sample called `name` in `process`.

    Arguments
    ---------
    process (mqr.process.Process) -- Process to plot.
    name (str) -- Name of the process.
    ax (matplotlib.axes.Axes) -- Axes for the plot.

    Optional
    --------
    show_long_term (bool) -- Plots a second Gaussian with 1.5 times more stddev.
        Default False.
    pdf_kwargs (dict) -- Dict passed to `mqr.plot.process` as keyword args.
    """
    sample = process.study[name]
    specification = process.specifications[name]
    capability = process.capabilities[name]

    ax1 = ax
    ax2 = ax.twinx()
    tolerance(specification, ax=ax1)
    sns.histplot(sample.data, color='lightgray', ax=ax1)
    pdf(
        sample=sample,
        spec=specification,
        capability=capability,
        show_long_term=show_long_term,
        ax=ax2,
        **pdf_kwargs)
    ax1.set_xlabel(f'{sample.name} (cp={capability.cp:#.3g}, cpk={capability.cpk:#.3g})')
