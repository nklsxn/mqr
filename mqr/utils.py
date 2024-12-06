from collections.abc import Iterable
import numpy as np

def make_ordinal(n):
    '''
    Convert an integer into its ordinal representation

    Thanks to: https://stackoverflow.com/a/50992575

    Parameters
    ----------
    n : int
        Number to express in ordinal representation
    '''
    if not np.isclose(n, int(n)):
        number = str(n)
        suffix = 'th'
    elif (11 <= (n % 100) <= 13) or ():
        number = str(int(n))
        suffix = 'th'
    else:
        number = str(int(n))
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(int(n) % 10, 4)]
    return number + suffix

def clip_where(a, a_min, a_max, where):
    '''
    Clamp `a` so that `a_min <= a <= a_max` is true or all true, masked by `where`.

    Parameters
    ----------
    a : number or array_like
        Value or values to clip.
    a_min : number
        Minimum value.
    a_max : number
        Maximum value.
    where : bool or array[bool]
        Clip when this value or corresponding element evaluates to `True`.
    '''
    aa = np.atleast_1d(a).copy()
    if not isinstance(a_min, Iterable):
        a_min = np.full(aa.size, a_min)
    if not isinstance(a_max, Iterable):
        a_max = np.full(aa.size, a_max)
    if not isinstance(where, Iterable):
        where = np.full(aa.size, where)

    if len(aa) != len(where):
        raise ValueError('Lengths of `a` and `where` must be equal.')
    if len(aa) != len(a_min):
        raise ValueError('Lengths of `a` and `a_min` must be equal.')
    if len(aa) != len(a_max):
        raise ValueError('Lengths of `a` and `a_max` must be equal.')

    aa[np.where(where)] = np.clip(aa[np.where(where)], a_min[np.where(where)], a_max[np.where(where)])
    if isinstance(a, Iterable):
        return aa
    else:
        return aa[0]

def fredholm2(t0, fn_K, fn_g, lmda, x, w):
    """
    Solve a Fredholm equation of the second kind.

    Evaluates f at t0 in:

    .. math::

        f(t) = g(t) + λ \\int_Q K(t, s) f(s) ds

    where the definite integral is approximated over the region Q defined by the
    quadrature points `x` and weights `w`.

    Uses the quadrature points and weights provided to form linear equations,
    and then interpolates the result using Nystrom's method. See [1]_ for a summary.

    Parameters
    ----------
    t0 : float
        Point to evaluate the function `f(t)`.
    fn_K : Callable[]
        Function that returns values of `K(t, s)`; see equation above.
    fn_g : Callable[]
        Function that returns values of `g(t)`; see equation above.
    lmbda : float
        Multiple of the integral term; see equation above.
    x : array_like
        Quadrature abcissa points.
    w : array_like
        Quadrature weights.

    References
    ----------
    .. [1]  Press, W. H. (2007).
            Numerical recipes 3rd edition: The art of scientific computing.
            Cambridge university press.
    """

    if len(x) != len(w):
        raise ValueError('Vectors x and w must be the same length.')

    N = len(x)
    g = np.full([N, 1], np.nan)
    K = np.full([N, N], np.nan)
    I = np.eye(N)

    for i in range(N):
        g[i] = fn_g(x[i])
        for j in range(N):
            K[i, j] = w[j] * fn_K(x[i], x[j])

    L = np.linalg.solve(I - lmda * K, g)[:, 0]

    # Nystrom's interpolation
    return fn_g(t0) + lmda * np.sum(w * L * fn_K(t0, x))
