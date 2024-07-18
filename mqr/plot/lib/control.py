"""
Plots for statistical process control.
"""

import numpy as np
import scipy.stats as st

def xbar_chart(samples, ax, x_limits=None):
    """
    Plots an Xbar chart.

    Where limits are not provided, they will be calculated from the data using
    `control.x_control_limits` function.

    Arguments
    ---------
    samples (numpy.ndarray) -- Samples with period/time on the first dimension
        and replicate/groups on the second.
    ax (matplotlib.axes.Axes) -- Axes for the plot.

    Optional
    --------
    x_limits (tuple[float, float, float]) -- the lower limit, target and upper limit.
    """

    if x_limits is None:
        (x_lcl, x_cl, x_ucl) = x_control_limits(samples)
    else:
        (x_lcl, x_cl, x_ucl) = x_limits
    x_bar = np.mean(samples, axis=1)
    
    ax.plot(x_bar, marker='o', color='k', label='X')
    ax.axhline(x_ucl, linewidth=0.8, color='C3', label=f'UCL={x_ucl:.4f}')
    ax.axhline(x_cl, linewidth=0.8, color='C0', label=f'CL={x_cl:.4f}')
    ax.axhline(x_lcl, linewidth=0.8, color='C3', label=f'LCL={x_lcl:.4f}')
    ax.set_title('Xbar Chart')
    ax.set_xticks(np.arange(len(samples)))
    ax.legend()
    
def r_chart(samples, ax, r_limits=None):
    """
    Plots an R chart.

    Where limits are not provided, they will be calculated from the data using
    `control.r_control_limits` function.

    Arguments
    ---------
    samples (numpy.ndarray) -- Samples with period/time on the first dimension
        and replicate/groups on the second.
    ax (matplotlib.axes.Axes) -- Axes for the plot.

    Optional
    --------
    r_limits (tuple[float, float, float]) -- the lower limit, target and upper limit.
    """
    
    if r_limits is None:
        (r_lcl, r_cl, r_ucl) = r_control_limits(samples)
    else:
        (r_lcl, r_cl, r_ucl) = r_limits
    r_bar = np.max(samples, axis=1) - np.min(samples, axis=1)

    ax.plot(r_bar, marker='o',color='k',label='R')
    ax.axhline(r_ucl, linewidth=0.8, color='C3', label=f'UCL={r_ucl:.4f}')
    ax.axhline(r_cl, linewidth=0.8, color='C0', label=f'CL={r_cl:.4f}')
    ax.axhline(r_lcl, linewidth=0.8, color='C3', label=f'LCL={r_lcl:.4f}')
    ax.set_title('R Chart')
    ax.set_xticks(np.arange(len(samples)))
    ax.legend()

def oc(n, c, ax, defect_range=None):
    """
    Plot an OC curve.

    Arguments
    ---------
    n (int) -- Sample size.
    c (int) -- Acceptance number.
    ax (matplotlib.axes.Axes) -- Axes for the plot.

    Optional
    --------
    defect_range (tuple[float, float]) -- Range of defect rates to show (on the
        x-axis).
    """

    if defect_range is None:
        defect_range = (0, 1)

    ps = np.linspace(*defect_range)
    
    pa = np.empty(ps.shape)
    for i in range(len(ps)):
        pa[i] = st.binom(n, ps[i]).cdf(c)
    
    ax.plot(ps, pa, linewidth=0.8, color='k')
    ax.grid()
    ax.set_xlabel('Defect rate')
    ax.set_ylabel('Prob of Acceptance')
