"""
=======================================
Data transforms (:mod:`mqr.transforms`)
=======================================

User guide
    :doc:`/user_guide/data-analysis`

Detailed examples
    https://github.com/nklsxn/mqrpy-guide

.. rubric:: Functions

.. autosummary::
    :toctree: generated/

    zscore
"""
from collections.abc import Iterable
import numpy as np
import pandas as pd

def zscore(data, mean=None, stddev=None):
    '''
    Create functions that z-score the given data, per-column.

    The returned functions (`z` or `z_inv`) can take as input a DataFrame whose
    columns are a subset of the columns in `data`, or a Series whose name is one
    of the columns of `data`.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to z-score. Mean and stddev are calculated for each column.
    mean : float or array_like, optional
        Mean to subtract from the data. Defaults to the mean of `data`.
    stddev : float or array_like, optional
        Stddev to divide out of the data. Defaults to the population sttdev of `data`.

    Returns
    -------
    stats : pandas.DataFrame
        Mean and standard deviation values for each column in data (where
        columns are the same as in data).
    z : callable
        A transform from data to its z-scored values.
    z_inv : callable
        A transform from z-scored values to the original space.
    '''
    stats = pd.DataFrame(index=['mean', 'std'], columns=data.columns)

    if mean is None:
        stats.loc['mean'] = data.apply(np.mean)
    else:
        stats.loc['mean'] = mean

    if stddev is None:
        stats.loc['std'] = data.apply(np.std, ddof=0)
    else:
        stats.loc['std'] = stddev

    def z(col):
        mean, std = stats[col.name]
        return (col - mean) / std
    def z_inv(col):
        """
        """
        mean, std = stats[col.name]
        return col * std + mean
    def _z(x):
        if isinstance(x, pd.DataFrame):
            return x.apply(z)
        elif isinstance(x, pd.Series):
            return z(x)
        else:
            raise ValueError('Pass either a DataFrame or a Series.')
    def _z_inv(x):
        if isinstance(x, pd.DataFrame):
            return x.apply(z_inv)
        elif isinstance(x, pd.Series):
            return z_inv(x)
        else:
            raise ValueError('Pass either a DataFrame of a Series.')
    return stats, _z, _z_inv
