"""
=======================================
Analysis of Variance (:mod:`mqr.anova`)
=======================================

.. :currentmodule:: mqr.anova

User guide
    :doc:`/user_guide/regression-anova`

Detailed examples
    https://github.com/nklsxn/mqrpy-guide

Tools for interpreting ANOVA results.

.. rubric:: Functions

.. autosummary::
    :toctree: generated/

    adequacy
    summary
    coeffs
    groups
    interactions
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats as st
import seaborn as sns
import statsmodels
import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings

################################################################################
## Tools
################################################################################

def summary(result, typ=2, formatted=True):
    '''
    The ANOVA table for a regression.

    Parameters
    ----------
    result : statsmodels.regression.linear_model.RegressionResults
        Result of calling `fit` on a statsmodels linear regression model.
    typ : {1, 2, 3, 'I', 'II', 'III'}, optional
        The ANOVA analysis type. Passed to :func:`anova_lm(..., typ=typ) 
        <statsmodels.stats.anova.anova_lm>`.
    formatted : bool, optional
        When True, returns a formatted DataFrame. Set this to False to access
        the calculated values.

    Returns
    -------
    pandas.DataFrame or pandas.io.formats.style.Styler
        The ANOVA table for the regression. A DataFrame when `formatted` is False
        and a `Styler` when True.
    '''
    table = sm.stats.anova_lm(result, typ=typ)
    table.loc['Total'] = table.sum(axis=0, skipna=False)
    table['mean_sq'] = table['sum_sq'] / table['df']
    table = table[['df', 'sum_sq', 'mean_sq', 'F', 'PR(>F)']]

    if formatted:
        format_map = {
            'df': '{:.0f}',
            'sum_sq': '{:,.4g}',
            'mean_sq': '{:.4g}',
            'F': '{:.2f}',
            'PR(>F)': '{:.3f}',
        }
        return table.style.format(format_map)
    else:
        return table

def coeffs(result, conf=0.95, formatted=True):
    """
    The coefficients from regression and their confidence intervals.

    Parameters
    ----------
    result : statsmodels.regression.linear_model.RegressionResults
        Result of calling `fit` on a statsmodels linear regression model.
    conf: float, optional
        Confidence level used to form an interval (decimal).
    formatted : bool, optional
        When True, returns a formatted DataFrame. Set this to False to access
        the calculated values.

    Returns
    -------
    pandas.DataFrame or pandas.io.formats.style.Styler
        Coefficients indexed by name from the model. A DataFrame when `formatted`
        is True, a Styler otherwise.
    """
    alpha = 1 - conf
    values = pd.concat(
        [result.params, result.conf_int(alpha=alpha)],
        axis=1)
    values.columns = ['Coeff', 'lower', 'upper']
    values['t'] = result.tvalues
    values['PR(>|t|)'] = result.pvalues
    exog = result.model.exog
    if exog.shape[1] > 1:
        values['VIF'] = np.nan
        for i in np.arange(exog.shape[1]):
            values.iloc[i, -1] = variance_inflation_factor(exog, i)

    if formatted:
        format_map = {
            'Coeff': '{:.4g}',
            'lower': '{:,.4g}',
            'upper': '{:.4g}',
            't': '{:.2f}',
            'PR(>|t|)': '{:.3f}',
            'VIF': '{:.3g}',
        }
        return values.style.format(format_map)
    else:
        return values

def groups(result, *, value: str, factor: str, conf=0.95, formatted=True):
    """
    The `value` column from a dataframe averaged over `factor`, and annotated
    with confidence intervals per level.

    See [1]_ (pp. 538-541) for construction of t-statistic.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe of measurements and categories.
    value : str
        Name of the column to average into groups.
    factor : str
        Name of the column (categorical) to group by.
    conf : float, optional
        The confidence level used to form an interval (decimal).
    formatted : bool, optional
        When True, returns a formatted DataFrame. Set this to False to access
        the calculated values.

    Returns
    -------
    pandas.DataFrame or pandas.io.formats.style.Styler
        Average values per group with confidence intervals. A DataFrame when
        `formatted` is True, a Styler otherwise.

    References
    ----------
    .. [1]  Saville, David J., and Graham R. Wood.
            Statistical methods: The geometric approach.
            Springer Science & Business Media, 2012.
    """
    alpha = 1 - conf
    df = result.model.data.frame
    groupby = df.groupby(factor)[value]
    cols = ['count', 'mean']
    groups = groupby.agg(cols)
    groups.columns = cols
    cis = _groups_ci(result, value, factor, conf)
    groups.loc[:, ['lower', 'upper']] = cis
    groups = groups.reindex(df.loc[:, factor].unique())

    if formatted:
        format_map = {
            'count': '{:.0f}',
            'mean': '{:,.4g}',
            'lower': '{:.4g}',
            'upper': '{:.4g}',
        }
        return groups.style.format(format_map)
    else:
        return groups

def interactions(df: pd.DataFrame, *, value: str, between: list[str], formatted=True):
    """
    Interaction table.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe of measurements and categories.
    value : str
        Name of the column of measurements.
    between : list[str]
        List of columns to group over before averaging.
    formatted : bool, optional
        When True, returns a formatted DataFrame. Set this to False to access
        the calculated values.

    Returns
    -------
    pandas.DataFrame or pandas.io.formats.style.Styler
        Average values per pair of columns in between. A DataFrame when `formatted`
        is True, a Styler otherwise.
    """
    result = df.groupby([*between])[value].mean().unstack()

    if formatted:
        return result.style.format(precision=4)
    else:
        return result

################################################################################
## Residual analysis
################################################################################

def adequacy(result, formatted=True):
    """
    Table of statistics from a fitted OLS regression (see Returns).

    Parameters
    ----------
    result : statsmodels.regression.linear_model.RegressionResults
        The result of calling `fit` on a statsmodels linear regression model.
    formatted : bool, optional
        When True, returns a formatted DataFrame. Set this to False to access
        the calculated values.

    Returns
    -------
    pandas.DataFrame or pandas.io.formats.style.Styler
        A DataFrame when `formatted` is True, a Styler otherwise. A list of
        statistics from the regression with columns:

        S
            square-root of the mean-squared error
        R-sq
            coefficient of determination
        R-sq (adj)
            coefficient of determination, adjusted for degrees of freedom in model
        F
            F-statistic for the whole model
        PR(>F)
            p-value of the F-statistic
        AIC
            Akaike information criterion
        BIC
            Bayesian information criterion
        N
            number of observations
    """
    data = {
        'S': np.sqrt(result.mse_resid),
        'R-sq': result.rsquared,
        'R-sq (adj)': result.rsquared_adj,
        'F': result.fvalue,
        'PR(>F)': result.f_pvalue,
        'AIC': result.aic,
        'BIC': result.bic,
        'N': int(result.nobs),
    }
    df = pd.DataFrame(
        data,
        index=[''])

    if formatted:
        format_map = {
            'S': '{:.4g}',
            'R-sq': '{:.4g}',
            'R-sq (adj)': '{:.4g}',
            'F': '{:.2f}',
            'PR(>F)': '{:.3f}',
            'AIC': '{:.4g}',
            'BIC': '{:.4g}',
            'N': '{:.0f}',
        }
        return df.style.format(format_map)
    else:
        return df

################################################################################
## Confidence intervals
################################################################################

def _groups_ci(result, value, factor, conf):
    alpha = 1 - conf
    df = result.model.data.frame

    nobs = len(df)
    se_dist = st.t(result.df_resid)
    level_names = df.loc[:, factor].unique()
    n_levels = len(level_names)
    se = np.sqrt(result.mse_resid)
    groups = df.groupby(factor)[value]

    ci = np.zeros([n_levels, 2])
    for i, level in enumerate(level_names):
        nobs_level = groups.count().loc[level]
        half_width = se_dist.ppf(1 - alpha / 2) * se / np.sqrt(nobs_level)
        ci[i, :] = groups.mean().loc[level] + np.array([-half_width, half_width])
    return ci
