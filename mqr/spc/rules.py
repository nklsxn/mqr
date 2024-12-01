from dataclasses import dataclass, field
import functools
import numpy as np
import pandas as pd

def combine(combn_fn, *rules):
    def _combine(control_statistic, control_params):
        alarms = [rule(control_statistic, control_params) for rule in rules]
        return functools.reduce(combn_fn, alarms)
    return _combine

def limits(control_statistic, control_params):
    stat = control_statistic.stat
    nobs = control_statistic.nobs
    lcl = control_params.lcl(nobs)
    ucl = control_params.ucl(nobs)

    return np.logical_or(stat >= ucl, stat <= lcl)

def aofb_nsigma(a, b, n):
    if a > b:
        raise ValueError(f'Cannot detect more than b of b signals (was passed "{a} of {b}").')
    def _rule(control_statistic, control_params):
        stat = control_statistic.stat
        nobs = control_statistic.nobs
        target = control_params.target()
        se = control_params.se(nobs)

        alarms = pd.Series(False, index=stat.index)
        for seq in (stat >= target + n * se).rolling(b):
            if np.sum(seq) >= a:
                alarms[seq.index[-1]] = True
        for seq in (stat <= target - n * se).rolling(b):
            if np.sum(seq) >= a:
                alarms[seq.index[-1]] = True
        return alarms
    return _rule

def n_1side(n):
    def _rule(control_statistic, control_params):
        stat = control_statistic.stat
        target = control_params.target()

        alarms = pd.Series(False, index=stat.index)
        for seq in np.sign(stat - target).rolling(n):
            if (len(seq) == n) and (len(set(seq)) == 1):
                alarms[seq.index[-1]] = True

        return alarms
    return _rule

def n_trending(n):
    def _rule(control_statistic, control_params):
        stat = control_statistic.stat
        target = control_params.target()

        alarms = pd.Series(False, index=stat.index)
        for seq in np.sign(stat.diff()).rolling(n-1):
            if (len(seq) == n-1) and len(set(seq)) == 1:
                alarms[seq.index[-1]] = True

        return alarms
    return _rule
