"""
Plots for showing the results of regression analysis.
"""

import numpy as np
import statsmodels.api as sm
import scipy.stats as st
import seaborn as sns

import mqr

def _tr_residuals(result, tr=None):
    if tr is None:
        return result.resid

    transformed = result.resid.copy()
    if tr == 'studentised':
        transformed.loc[:] = result.get_influence().resid_studentized
    elif tr == 'studentised_internal':
        transformed.loc[:] = result.get_influence().resid_studentized_internal
    elif tr == 'studentised_external':
        transformed.loc[:] = result.get_influence().resid_studentized_external
    elif tr == 'PRESS':
        transformed.loc[:] = result.get_influence().resid_press
    else:
        raise RuntimeError(f'transform not recognised: {tr}')
    return transformed

def residual_histogram(result, ax, tr=None):
    """
    Plot histogram of residuals from result of calling `fit()` on statsmodels
    model.

    Arguments
    ---------
    result -- Result of fitting a `statsmodels` model.
    ax (matplotlib.axes.Axes) -- Axes for the plot.

    Optional
    --------
    tr (str) -- Transformation to apply to the residuals. One of "studentised",
        "studentised_internal", "studentised_external", "PRESS". Default `None`.
    """
    ax0l = ax
    ax0r = ax.twinx()
    resid = _tr_residuals(result, tr)
    xs = np.linspace(-3*np.std(resid), 3*np.std(resid), 200)
    ys = st.norm(np.mean(resid), np.std(resid, ddof=1)).pdf(xs)
    ax0r.plot(xs, ys, linewidth=1.0, color='k')
    sns.histplot(resid, stat='density', ax=ax0l, color='C0')
    ax.set_xlabel('residual')
    ax.set_ylabel('density')

def res_v_obs(result, ax, tr=None, influence_stat=None):
    """
    Plot residuals versus observations.

    Arguments
    ---------
    result -- Result of fitting a `statsmodels` model.
    ax (matplotlib.axes.Axes) -- Axes for plot.

    Optional
    --------
    tr (str) -- Transformation to apply to the residuals. One of "studentised",
        "studentised_internal", "studentised_external", "PRESS". Default `None`.
    influence_stat () -- If not `None` (default), plot this influence statistic
        for each residual as a bar. One of "cooks_dist" or "bonferroni".
    """
    resid = _tr_residuals(result, tr)
    ax.plot(resid.index, resid, linewidth=0.5, marker='.')
    ax.grid(axis='y')
    if not resid.index.name:
        ax.set_xlabel('run')
    else:
        ax.set_xlabel(resid.index.name)
    ax.set_ylabel('residual')

    if influence_stat is not None:
        if influence_stat == 'cooks_dist':
            p = 1 - result.get_influence().cooks_distance[1]
            label = "Cook's distance (1-p)"
        elif influence_stat == 'bonferroni':
            p = 1 - result.outlier_test().loc[:, 'bonf(p)']
            label = 'Bonferroni (1-p)'
        else:
            raise RuntimeError(f'statistic not recognised: {influence_stat}')

        axt = ax.twinx()
        axt.bar(np.arange(result.nobs), p, alpha=0.5, color='C1')
        axt.set_ylim(0.0, 1.0)
        axt.set_ylabel(label)

def res_v_fit(result, ax, tr=None):
    """
    Plot residual versus fit.

    Arguments
    ---------
    result -- Result of fitting a `statsmodels` model.
    ax (matplotlib.axes.Axes) -- Axes for plot.

    Optional
    --------
    tr (str) -- Transformation to apply to the residuals. One of "studentised",
        "studentised_internal", "studentised_external", "PRESS". Default `None`.
    """
    resid = _tr_residuals(result, tr)
    fitted = result.fittedvalues
    ax.plot(fitted, resid, linewidth=0, marker='.')
    ax.grid(axis='y')
    ax.set_xlabel('fitted value')
    ax.set_ylabel('residual')

def residuals(result, axs, tr=None, influence_stat=None):
    """
    Plot a probability plot of residuals, histogram of residuals, residuals
    versus observation and residuals versus fitted values for the residuals in
    a fitted statsmodels model.

    Arguments
    ---------
    result -- Result of fitting a `statsmodels` model.
    axs (np.ndarray[matplotlib.axes.Axes]) -- Array of axes for plot. Must have
        four elements. Will be flattened before use.

    Optional
    --------
    tr (str) -- Transformation to apply to the residuals. One of "studentised",
        "studentised_internal", "studentised_external", "PRESS". Default `None`.
    influence_stat () -- If not `None` (default), plot this influence statistic
        for each residual as a bar. If specified, one of "cooks_dist" or
        "bonferroni". Default `None`.
    """
    axs = axs.flatten()
    assert len(axs) == 4 , f'subplots must have 4 axes.'

    resid = _tr_residuals(result, tr)
    gen = sm.ProbPlot(
        resid,
        dist=st.norm,
        fit=True)
    gen.qqplot(line='s', ax=axs[0])
    residual_histogram(result, tr=tr, ax=axs[1])
    res_v_obs(result, tr=tr, influence_stat=influence_stat, ax=axs[2])
    res_v_fit(result, tr=tr, ax=axs[3])

def interaction(data, obs:str, ax, between=tuple[str, str]):
    """
    Interaction plot for observation `obs` between the two categories in `between`.
    
    Arguments
    ---------
    data (pd.DataFrame) -- Table of categorical data.
    obs (str) -- Column name that contains observations (real numbers).
    between (tuple[str, str]) -- Pair of column names to show interaction
        table (categorical).
    ax (matplotlib.axes.Axes) -- Axes for the plot.
    """
    intn = mqr.anova.interactions(data, value=obs, between=between)
    ax.plot(intn, marker='_')
    ax.set_xticks(intn.index)
    ax.set_xlabel(between[0])
    ax.set_ylabel(f'{obs}')

    ax.grid()
    ax.legend(
        intn.columns,
        title=between[1],
        prop={'size': 8},
        fancybox=True,
        bbox_to_anchor=(1.02, 0.5, 0.0, 0.0),
        loc='center left',
        borderaxespad=0.0)

    ax.set_title(f'{between[0]} * {between[1]} Interaction')
