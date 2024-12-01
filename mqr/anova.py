"""
Analysis of Variance.

Tools for interpreting ANOVA results.
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

def summary(result, typ=2):
    '''
    The ANOVA table for a regeression.

    Arguments
    ---------
    result -- Result of calling `fit` on a statsmodels model.

    Optional
    --------
    typ (int or str) -- The ANOVA analysis type. Passed to
        `statsmodels.api.stats.anova_lm(..., typ=typ)`.

    Returns
    -------
    (pd.DataFrame) -- The ANOVA (type 2) table for the regression
    '''
    table = sm.stats.anova_lm(result, typ=typ)
    table.loc['Total'] = table.sum(axis=0, skipna=False)
    table['mean_sq'] = table['sum_sq'] / table['df']
    table = table[['df', 'sum_sq', 'mean_sq', 'F', 'PR(>F)']]
    return table

def coeffs(result, conf=0.95):
    """
    The coefficients from regression and their confidence intervals.

    Arguments
    ---------
    result -- Result of calling `fit` on a statsmodels model.

    Optional
    --------
    conf (float) -- Confidence level used to form an interval (decimal).
        (Default 0.95.)

    Returns
    -------
    (pd.DataFrame) -- Coefficients indexed by name from the model.
    """
    alpha = 1 - conf
    values = pd.concat(
        [result.params, result.conf_int(alpha=alpha)],
        axis=1)
    values.columns = ['Coeff', f'[{alpha/2*100:g}%', f'{(1-alpha/2)*100:g}%]']
    values['t'] = result.tvalues
    values['PR(>|t|)'] = result.pvalues
    exog = result.model.exog
    if exog.shape[1] > 1:
        values['VIF'] = np.nan
        for i in np.arange(exog.shape[1]):
            values.iloc[i, -1] = variance_inflation_factor(exog, i)
    return values

def groups(result, *, value: str, factor: str, conf=0.95):
    """
    The `value` column from a dataframe averaged over `factor`, and annotated
    with confidence intervals per level.

    Arguments
    ---------
    df (pd.DataFrame) -- A dataframe of measurements and categories.
    value (str) -- The name of the column to average into groups.
    factor (str) -- The name of the column (categorical) to group by.

    Optional
    --------
    conf (float) -- The confidence level used to form an interval (decimal).
        (Default 0.95.)

    Returns
    -------
    pd.DataFrame -- Average values per group with confidence intervals.
    """
    alpha = 1 - conf
    df = result.model.data.frame
    groupby = df.groupby(factor)[value]
    cols = ['count', 'mean']
    groups = groupby.agg(cols)
    groups.columns = cols
    cis = _groups_ci(result, value, factor, conf)
    groups.loc[:, [f'[{alpha/2*100:g}%', f'{(1-alpha/2)*100:g}%]']] = cis
    groups = groups.reindex(df.loc[:, factor].unique())
    return groups

def interactions(df: pd.DataFrame, *, value: str, between: list[str]):
    """
    Interaction table.

    Arguments
    ---------
    df (pd.DataFrame) -- A dataframe of measurements and categories.
    value (str) -- The name of the column of measurements.
    between (list[str]) -- A list of columns to group over before averaging.

    Returns
    -------
    pd.DataFrame -- Average values per pair of columns in between.
    """
    return df.groupby([*between])[value].mean().unstack()

################################################################################
## Residual analysis
################################################################################

def adequacy(result):
    """
    Table of statistics from a fitted OLS regression (see Returns).

    Arguments
    ---------
    result -- The result of calling `fit` on a statsmodels model.

    Returns
    -------
    pd.DataFrame -- A list of statistics from the regression:
        * S: square-root of the mean-squared error
        * R-sq: coefficient of determination
        * R-sq (adj): coefficient of determination, adjusted for degrees of freedom
          in model
        * N: number of observations
        * JB: Jarque-Berra test statistic
        * p (JB): p-value for JB test (null is zero skewness and zero excess
          kurtosis -- normality)
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
    return pd.DataFrame(
        data,
        index=[''])

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
