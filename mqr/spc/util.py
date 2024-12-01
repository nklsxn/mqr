import importlib
import numpy as np
import pandas as pd
import pickle
import scipy

import mqr
# d2_array = np.full([99], np.nan)
# for n in range(2, 101):
#     d2_array[n-2] = mqr.spc.util.d2_fn(n, epsabs=1e-10)
# with open('d2-table.pkl', 'wb') as f:
#     pickle.dump(d2_array, f)

tables = importlib.resources.files('mqr.tables')

with open(tables/'c4-table.pkl', 'rb') as f:
    c4_table = pickle.load(f)

with open(tables/'d2-table.pkl', 'rb') as f:
    d2_table = pickle.load(f)

with open(tables/'d3-table.pkl', 'rb') as f:
    d3_table = pickle.load(f)

def lookup(index, table):
    if isinstance(index, pd.Series):
        return index.apply(lambda n: table[n])
    else:
        return table[index]

def c4(n):
    if np.any(n < 2) or np.any(n > 100):
        raise ValueError('Sample size n must be between 2 and 100.')

    return lookup(n-2, c4_table)

def d2(n):
    if np.any(n < 2) or np.any(n > 100):
        raise ValueError('Sample size n must be between 2 and 100.')

    return lookup(n-2, d2_table)

def d3(n):
    if np.any(n < 2) or np.any(n > 100):
        raise ValueError('Sample size n must be between 2 and 100.')

    return lookup(n-2, d3_table)

def c4_fn(n):
    num = scipy.special.gamma(n / 2) * np.sqrt(2 / (n - 1))
    den = scipy.special.gamma((n - 1) / 2)
    return num / den

def f2(n):
    dist = scipy.stats.norm()
    def _f2(x):
        phi_x = dist.cdf(x)
        return 1 - (1 - phi_x)**n - phi_x**n
    return _f2

def f3(n):
    dist = scipy.stats.norm()
    def _f3(x, y):
        phi_x = dist.cdf(x)
        phi_y = dist.cdf(y)
        return 1 - phi_y**n - (1-phi_x)**n + (phi_y - phi_x)**n
    return _f3

def f3_tr(n):
    dist = scipy.stats.norm()
    sqrt2 = np.sqrt(2)

    def _f3_tr(s, t):
        x = (s - t) / sqrt2
        y = (s + t) / sqrt2
        phi_x = dist.cdf(x)
        phi_y = dist.cdf(y)
        return 1 - phi_y**n - (1-phi_x)**n + (phi_y - phi_x)**n
    return _f3_tr

def d2_integral(n, **quad_kws):
    return scipy.integrate.quad(f2(n), -np.inf, np.inf, **quad_kws)[0]

def d3_integral(n, d2_fn=None, **dblquad_kws):
    integral = scipy.integrate.dblquad(f3_tr(n), 0, np.inf, -np.inf, np.inf, **dblquad_kws)[0]
    d2_val = d2_fn(n) if (d2_fn is not None) else d2(n)
    return np.sqrt(2 * integral - d2_val**2)

def solve_arl(h4, p, lmda, N=20):
    a, b = 0, np.sqrt(h4 * lmda / (2 - lmda))
    scale = (b - a) / 2
    shift = (a + b) / 2

    z, w = scipy.special.roots_legendre(N)
    z = z * scale + shift

    def eta(alpha):
        return alpha * ((1 - lmda) / lmda)**2

    def fn_W(s, t):
        return scipy.stats.ncx2(p, eta(s**2)).pdf(t**2 / lmda**2) * 2 * t

    def fn_g(s):
        return 1

    c = scale / lmda**2

    return mqr.utils.fredholm2(0, fn_W, fn_g, c, z, w)

def solve_h4(arl_0, p, lmda, init_h4=15.0):
    fn = lambda x: solve_arl(x, p, lmda) - arl_0
    result = scipy.optimize.root_scalar(fn, x0=init_h4)
    return result.root, result

def group_consecutives(array, is_consecutive=None):
    if len(array) == 0:
        return []
    if is_consecutive is None:
        is_consecutive = lambda a, b: (b - a == 1)
    groups = []
    acc = [array[0]]
    last = array[0]
    for i in array[1:]:
        if not is_consecutive(last, i):
            groups.append(acc)
            acc = []
        acc.append(i)
        last = i
    groups.append(acc)
    return groups

def alarm_subsets(alarms):
    idx = np.where(alarms)[0]
    groups = group_consecutives(idx)
    return [pd.Index(alarms.index[g]) for g in groups]
