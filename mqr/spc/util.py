"""
===================================
SPC utilities (:mod:`mqr.spc.util`)
===================================

.. currentmodule: mqr.spc.util

.. rubric:: Functions

.. autosummary::
    :toctree: generated/

    c4
    d2
    d3
    c4_fn
    d2_integral
    d3_integral
    solve_arl
    solve_h4
    group_consecutives
    alarm_subsets

"""

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
    """
    Lookup a value in a table by index.

    Parameters
    ----------
    index : int, ndarray, Series
        Index or indices to lookup.
    table : {c4_table, d2_table, d3_table}
        Tabulated data.

    Notes
    -----
    No shift is applied to the index, so ensure the provided index corresponds
    to the desired sample size.

    Returns
    -------
    int, ndarray, Series
        Values from the lookup table, with the same type and dimensions as the
        argument `index`.
    """
    if isinstance(index, pd.Series):
        return index.apply(lambda n: table[n])
    else:
        return table[index]

def c4(n):
    """
    Retrieves tabulated values of the unbiasing constant c4.

    Parameters
    ----------
    n : int, ndarray, Series
        Sample size(s).

    Notes
    -----
    The tabulated values were generated using :func:`c4_fn`.

    Returns
    -------
    int, ndarray, Series
        Values from the c4 table, with the same type and dimensions as the
        argument `index`.
    """
    if np.any(n < 2) or np.any(n > 100):
        raise ValueError('Sample size n must be between 2 and 100.')

    return lookup(n-2, c4_table)

def d2(n):
    """
    Retrieves tabulated values of the unbiasing constant d2.

    Parameters
    ----------
    n : int, ndarray, Series
        Sample size(s).

    Notes
    -----
    The tabulated values were generated using :func:`d2_integral`.

    Returns
    -------
    int, ndarray, Series
        Values from the d2 table, with the same type and dimensions as the
        argument `index`.
    """
    if np.any(n < 2) or np.any(n > 100):
        raise ValueError('Sample size n must be between 2 and 100.')

    return lookup(n-2, d2_table)

def d3(n):
    """
    Retrieves tabulated values of the unbiasing constant d3.

    Parameters
    ----------
    n : int, ndarray, Series
        Sample size(s).

    Notes
    -----
    The tabulated values were generated using :func:`d3_integral`.

    Returns
    -------
    int, ndarray, Series
        Values from the d3 table, with the same type and dimensions as the
        argument `index`.
    """
    if np.any(n < 2) or np.any(n > 100):
        raise ValueError('Sample size n must be between 2 and 100.')

    return lookup(n-2, d3_table)

def c4_fn(n):
    """
    Direct calculation of the c4 unbiasing constant from the gamma function.

    Evaluates:

    .. math::
        c_4(n) = \\frac{ \\Gamma\\left(\\frac{n}{2}\\right)\\sqrt{\\frac{n}{n-1}}}{\\Gamma\\left(\\frac{n-1}{2}\\right)} .

    Parameters
    ----------
    n : int
        Sample size.

    Returns
    -------
    float
        The value of c4 for the given sample size.
    """
    num = scipy.special.gamma(n / 2) * np.sqrt(2 / (n - 1))
    den = scipy.special.gamma((n - 1) / 2)
    return num / den

def f2(n):
    """
    Integrand for the integral defining d2.
    """
    dist = scipy.stats.norm()
    def _f2(x):
        phi_x = dist.cdf(x)
        return 1 - (1 - phi_x)**n - phi_x**n
    return _f2

def f3_tr(n):
    """
    Integrand for the integral in d3 (transformed version).

    The transform is a substitution of variables the rotates the integration
    limits through 45deg (in the x-y plane), removining the dependence of the
    inner integral's limit on the outer integral's variable.
    """
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
    """
    Numerical integration to calculate the d2 unbiasing constant.

    Evaluates:

    .. math::
        d_2(n) = \\int_{-\\infty}^{\\infty} f_2(x; n) dx 

    where

    .. math::
        f_2(x; n) = 1 - (1 - \\Phi(x))^n - \\Phi(x)^n .

    Parameters
    ----------
    n : int
        Sample size.
    **quad_kws
        Keyword args passed as `**quad_kws` to :func:`scipy.integrate.quad`.

    Returns
    -------
    float
    """
    return scipy.integrate.quad(f2(n), -np.inf, np.inf, **quad_kws)[0]

def d3_integral(n, d2_fn=None, **dblquad_kws):
    """
    Numerical integration to calculate the d3 unbiasing constant.

    Evaluates:

    .. math::
        d_3(n) = \\sqrt{ 2\\int_{-\\infty}^{\\infty} \\int_{-\\infty}^{y} f_3(x, y; n) dx dy - d_2^2(n) },

    where

    .. math::
        f_3(x, y; n) = 1 - \\Phi(y)^n - (1 - \\Phi(x))^n + (\\Phi(y) - \\Phi(x))^n.

    Evaluating the integral is slightly quicker with the transform
    (which is a rotation of the plane through 45 degrees)

    .. math::
        x = \\frac{s - t}{\\sqrt{2}}, \\\\
        y = \\frac{s + t}{\\sqrt{2}}

    over :math:`s \\in (-\\infty, \\infty)` and :math:`t \\in [0, \\infty)`.
    The transform's Jacobian has determinant equal to 1, so the integral is
    evaluated as

    .. math::
        d_3(n) = \\sqrt{ 2\\int_{0}^{\\infty} \\int_{-\\infty}^{\\infty}
            f_3\\left(\\frac{s - t}{\\sqrt{2}}, \\frac{s + t}{\\sqrt{2}}; n\\right) dx dy - d_2^2(n) } .

    Parameters
    ----------
    n : int
        Sample size
    d2_fn : Callable[int, float], optional
        Function evaluating the d2 unbiasing constant. Default is
        :func:`mqr.spc.util.d2`.
    **dblquad_kws : dict
        Keyword args passed as `**dblquad_kws` to :func:`scipy.integrate.dblquad`.

    Notes
    -----
    Uses a substitution that removes the limit's dependence on variables.

    Returns
    -------
    float
    """
    integral = scipy.integrate.dblquad(f3_tr(n), 0, np.inf, -np.inf, np.inf, **dblquad_kws)[0]
    d2_val = d2_fn(n) if (d2_fn is not None) else d2(n)
    return np.sqrt(2 * integral - d2_val**2)

def solve_arl(h4, p, lmda, N=20):
    """
    Find the in-control ARL of an MEWMA chart.

    Follows the modified Gauss-Legendre method in [1]_ to evaluate:

    .. math::
        L(\\alpha) = 1 + \\int_0^h L(u)/\\lambda^2 f(u/\\lambda^2 | \\eta(\\alpha)) du

    where

    .. math::
        \\begin{gather}
            h = h_4 \\lambda / (2 - \\lambda),\\text{ and} \\\\
            \\eta(\\alpha) = \\alpha ((1 - \\lambda) / \\lambda)^2.
        \\end{gather}

    The algorithm makes a change of variables
    (:math:`\\alpha \\rightarrow \\alpha^2`), which improves its performance.

    Parameters
    ----------
    h4 : float
        Upper control limit on MEWMA chart.
    p : int
        Dimension of monitored vector.
    lmda : float
        Decay factor for EWMA statistic.
    N : int, optional
        Number of quadrature sample points (Gauss-Legendre).

    Returns
    -------
    float

    References
    ----------
    .. [1]  Knoth, S. (2017).
            ARL numerics for MEWMA charts.
            Journal of Quality Technology, 49(1), 78-89.
    """
    a, b = 0, np.sqrt(h4 * lmda / (2 - lmda))
    scale = (b - a) / 2
    shift = (a + b) / 2

    z, w = scipy.special.roots_legendre(N)
    z = z * scale + shift
    w = w * scale

    def eta(alpha):
        return alpha * ((1 - lmda) / lmda)**2

    def fn_W(s, t):
        return scipy.stats.ncx2(p, eta(s**2)).pdf(t**2 / lmda**2) * 2 * t

    def fn_g(s):
        return 1

    c = 1 / lmda**2

    return mqr.utils.fredholm2(0, fn_W, fn_g, c, z, w)

def solve_h4(arl_0, p, lmda, init_h4=15.0):
    """
    Find the UCL of an MEWMA chart from the in-control ARL.

    Parameters
    ----------
    arl_0 : float
        In-control average run length.
    p : int
        Dimension of monitored vector.
    lmda : float
        Decay factor for EWMA statistic.
    init_h4 : float, optional
        Initial guess for the upper limit.

    Returns
    -------
    scipy.optimize.RootResults
        The result from the root finder. Use `result.converged` to check if a
        result was found, and call `result.root` to get the value of the limit, h4.
        See :func:`scipy.optimize.root_scalar` for more info.
    """
    fn = lambda x: solve_arl(x, p, lmda) - arl_0
    result = scipy.optimize.root_scalar(fn, x0=init_h4)
    return result.root, result

def group_consecutives(array, is_consecutive=None):
    """
    Groups the index of an array to form sets of consecutive values.

    Parameters
    ----------
    array : array_like[T]
        Values with possible consecutive elements.
    is_consecutive : Callable[[T, T], bool]
        A function returning True if its two arguments are consecutive.

    Returns
    -------
    list[T]

    Examples
    --------

    >>> consecutive_rule = lambda a, b: b == a + 1
    >>> mqr.spc.util.group_consecutives(
            np.array([1, 2, 4, 6, 7, 8, 10]),
            consecutive_rule)
    [array([1, 2]), array([4]), array([6, 7, 8]), array([10])]

    """
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
    """
    Group alarms into subsets where indices are consecutive.

    Parameters
    ----------
    alarms : pandas.Series[bool]
        Series of control alarm points.

    Returns
    -------
    list[pandas.Series[bool]]
        Lists of alarms with consecutive indices.
    """
    idx = np.where(alarms)[0]
    groups = group_consecutives(idx)
    return [pd.Index(alarms.index[g]) for g in groups]
