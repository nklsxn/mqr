"""
Design of experiments.

Tools for constructing experiments. The routines here are designed to easily
compose standard experimental designs into more complex designs. The
construction routines call through to pyDOE3 for convenience, though you can
construct custom designs and also call that library directly and pass its
results. If you only need the results of pyDOE3, call it directly. See
[](pydoe3.readthedocs.io).


See `from_fullfact`, `from_fracfact`, `from_ccdesign`, `from_centrepoints` and
to construct standard components from pyDOE3. Function `from_axial` constructs
axial points. Use `from_levels` to construct others (or the ones above
directly), for example to construct a Box-Behnken design (with edge points):
>>> bbdesign = Design.from_levels(['a', 'b', 'c'], pyDOE3.bbdesign(3, 2))

The principle of this library is that many standard designs can be built from
simple elements. For example, a central composite design is the composition of
fractional factorial design, centrepoints and axial points. Concatenate runs
with the `+` operator to a central composite design, in two blocks, like this:
>>> names = ['x1', 'x2', 'x3', 'x4']
>>> generator = 'a b c abc'
>>> centre_pts = 3
>>> frac_fact = Design.from_fracfact(names, generator)
>>> frac_cpts = Design.from_centrepoints(names, centre_pts)
>>> axial = Design.from_axial(names)
>>> axial_cpts = Design.from_centrepoints(names, centre_pts)
>>> design = (frac_fact + frac_cpts) + (axial + axial_cpts).as_block(2)
"""

from dataclasses import dataclass, field
from collections.abc import Iterable
import pyDOE3
import numpy as np
import pandas as pd

@dataclass
class Design:
    """
    An experimental design.

    Designs should normally be constructed using the from_* methods (see
    examples), which wrap calls to the pyDOE3 library. Designs can also be
    constructed manually, either from pyDOE3 or any other method, including
    directly from numpy arrays. See examples below.

    Designs are composable by concatenation with the `+` operator.

    Attributes
    ----------
    names (list[str]) -- Names of variables.
    levels (np.ndarray) -- Two-dimensional array containing the levels for each
        experiment, with a column for each variables and a row for each run.
    runs (np.ndarray) -- Numerical labels for each run. Useful for tracking
        runs after randomisation.
    pttypes (np.ndarray) -- Numerical label for each point type:
        * 0: centre point
        * 1: corner point
        * 2: axial point
    blocks (np.ndarray) -- Numerical label for each block (default 1).

    Examples
    --------
    >>> Design.from_full_fact(['x1', 'x2', 'x3'], [2, 2, 2])
       PtType  Block   x1   x2   x3
    1       1      1 -1.0 -1.0 -1.0
    2       1      1  1.0 -1.0 -1.0
    3       1      1 -1.0  1.0 -1.0
    4       1      1  1.0  1.0 -1.0
    5       1      1 -1.0 -1.0  1.0
    6       1      1  1.0 -1.0  1.0
    7       1      1 -1.0  1.0  1.0
    8       1      1  1.0  1.0  1.0

    >>> d1 = Design.from_fracfact(['x1', 'x2', 'x3', 'x4'], 'a b c abc')
    >>> d2 = Design.from_centrepoints(['x1', 'x2', 'x3', 'x4'], 3)
    >>> d1 + d2.as_block(2)
        PtType  Block   x1   x2   x3   x4
    1        1      1 -1.0 -1.0 -1.0 -1.0
    2        1      1  1.0 -1.0 -1.0  1.0
    3        1      1 -1.0  1.0 -1.0  1.0
    4        1      1  1.0  1.0 -1.0 -1.0
    5        1      1 -1.0 -1.0  1.0  1.0
    6        1      1  1.0 -1.0  1.0 -1.0
    7        1      1 -1.0  1.0  1.0 -1.0
    8        1      1  1.0  1.0  1.0  1.0
    9        0      2  0.0  0.0  0.0  0.0
    10       0      2  0.0  0.0  0.0  0.0
    11       0      2  0.0  0.0  0.0  0.0
    """
    names: list[str]
    levels: pd.DataFrame
    runs: pd.Index
    pttypes: pd.Series
    blocks: pd.DataFrame

    def replicate(self, n, label=None):
        """
        Create a new design with each run replicated `n` times.

        Arguments
        ---------
        n (int) -- The number of replicates to create.

        Returns
        -------
        Design -- A new design that is a replicated version of this one.
        """
        idx = self.runs.repeat(n)
        new_runs = pd.RangeIndex(1, len(self) * n + 1)
        new_levels = self.levels.loc[idx].set_index(new_runs)
        if self.pttypes is not None:
            new_pttypes = self.pttypes.loc[idx].set_axis(new_runs)
        else:
            new_pttypes = None
        new_blocks = self.blocks.loc[idx].set_index(new_runs)
        if label is not None:
            new_blocks[label] = np.tile(np.arange(n), len(self)) + 1
        return Design(
            names=self.names,
            levels=new_levels,
            runs=new_runs,
            pttypes=new_pttypes,
            blocks=new_blocks)

    def as_block(self, level, name='Block'):
        """
        Return the same set of runs with their block label set to `block`.

        Arguments
        ---------
        block (int) -- New block label.

        Returns
        -------
        Design -- A copy of this design with a new block label.
        """
        new_blocks = self.blocks.copy()
        new_blocks[name] = level
        return Design(
            names=self.names,
            levels=self.levels,
            runs=self.runs,
            pttypes=self.pttypes,
            blocks=new_blocks)

    def to_df(self):
        """
        Construct a dataframe representation of the design.

        Returns
        -------
        pd.DataFrame -- The design.
        """

        df = pd.DataFrame(index=self.runs)
        if self.pttypes is not None:
            df['PtType'] = self.pttypes
        df[self.blocks.columns] = self.blocks
        for name in self.names:
            df[name] = self.levels[name]
        return df

    def get_factor_df(self, name, ref_levels=0.0):
        """
        Create a dataframe containing all unique levels in this design for a
        variable, and a reference level for all others.

        Arguments
        ---------
        name (str) -- The factor to isolate.

        Optional
        --------
        ref_levels (float) -- The reference level assigned to all other
            variables. (Default 0.)

        Returns
        -------
        pd.DataFrame -- A dataframe with levels in `name` as rows and
            variable names as columns.
        """
        df = pd.DataFrame(columns=self.names, dtype=np.float64)
        df[name] = np.sort(np.unique(self.levels[name]))
        df.fillna(ref_levels, inplace=True)
        return df

    def randomise_runs(self, order=None):
        """
        Return the same set of runs, randomised over their run labels.

        Optional
        ---------
        preserve_blocks (bool) -- When `True`, randomise runs only within
            blocks, when `False` randomise runs across blocks (blocks will no
            longer be in order). (Default True.)

        Returns
        -------
        Design -- A copy of this design, randomised.
        """
        df = self.to_df()
        rnd = np.random.choice(
            a=df.index,
            size=df.shape[0],
            replace=False)
        result = df.loc[rnd]
        if order is not None:
            result.sort_values(order, inplace=True, kind='stable')
        return Design(
            names=self.names,
            levels=result[self.names],
            runs=result.index,
            pttypes=result['PtType'],
            blocks=result[self.blocks.columns])

    def __add__(self, other):
        """
        Concatenate the runs of another design at the end of this design.

        Arguments
        ---------
        other (Design) -- The design to concatenate.

        Returns
        -------
        Design -- This design and `other` concatenated into one, with
            run labels of `other` offset to continue from the end of this design.
        """
        if self.names != other.names:
            raise AttributeError('Designs must contain the same variables.')

        new_runs = self.runs.append(other.runs + self.runs.max())

        new_levels = pd.concat([self.levels, other.levels], axis=0)
        new_levels.set_index(new_runs, inplace=True)
        new_pttypes = pd.concat([self.pttypes, other.pttypes], axis=0).set_axis(new_runs)
        new_blocks = pd.concat([self.blocks, other.blocks], axis=0)
        new_blocks.set_index(new_runs, inplace=True)

        return Design(
            names=self.names,
            levels=new_levels,
            runs=new_runs,
            pttypes=new_pttypes,
            blocks=new_blocks)

    def __len__(self):
        return len(self.runs)

    def transform(self, **transforms):
        """
        Apply a transform to the levels of this design.

        Arguments
        ---------
        transform (Transform) -- Transform to apply.

        Returns
        -------
        Design -- A copy of this design with new levels.
        """
        new_levels = self.levels.copy()
        for name, tx in transforms.items():
            if callable(tx):
                new_levels[name] = tx(self.levels[name])
            elif isinstance(tx, dict):
                new_levels[name] = np.vectorize(tx.get)(self.levels[name])
            else:
                raise ValueError(f'Unknown transform {tx}.')
        return Design(
            names=self.names,
            levels=new_levels,
            runs=self.runs,
            pttypes=self.pttypes,
            blocks=self.blocks)

    # def __matmul__(self, transform):
    #     """
    #     Apply a transform to the levels of this design.

    #     Parameters
    #     ----------
    #     transform : :class:`Transform`
    #         Transform to apply.

    #     Returns
    #     -------
    #     :class:`Design`
    #         A copy of this design with new levels.
    #     """
    #     return Design(
    #         names=self.names,
    #         levels=transform(self.levels),
    #         runs=self.runs,
    #         pttypes=self.pttypes,
    #         blocks=self.blocks)

    @staticmethod
    def from_levels(names, levels, runs=None):
        """
        Construct a design from an array of levels.

        Arguments
        ---------
        names (list[str]) -- List of variable names.
        levels (np.ndarray) -- Two-dimensional array of levels, with runs in
            rows and variables in columns.

        Optional
        --------
        runs (np.ndarray) -- Array of labels for runs. (Default `None` results
            in labels counting from 1.)
        block (int) -- Block label for the runs in this design. (Default 1.)

        Returns
        -------
        Design -- The new design.
        """
        m, n = levels.shape
        if n != len(names):
            raise AttributeError('Length of `names` must match number of columns in `levels`.')
        if (runs is not None) and len(runs) != m:
            raise AttributeError('Length of `runs` must match number of rows in `levels`.')

        runs = pd.RangeIndex(1, m+1) if (runs is None) else pd.Index(runs)
        levels = pd.DataFrame(levels, index=runs, columns=names)
        blocks = pd.DataFrame(index=runs)
        return Design(
            names=names,
            levels=levels,
            runs=runs,
            pttypes=None,
            blocks=blocks)

    @staticmethod
    def from_fullfact(names, levels, scale_origin=True, pttypes=True):
        """
        Construct a design from `pyDOE3.fullfact(...)`, and centre the level
        values on 0.

        Arguments
        ---------
        names (list[str]) -- List of variable names.
        levels (list[int]) -- A list of counts of levels, passed directly to
            pyDOE3.fullfact(...).

        Optional
        --------
        block (int) -- Block label for the runs in this design. (Default 1.)

        Returns
        -------
        Design -- The new design.
        """
        coded_levels = pyDOE3.fullfact(levels)
        design = Design.from_levels(names, coded_levels)
        if scale_origin:
            value_counts = [len(np.unique(design.levels[name])) for name in design.names]
            mapper = lambda x: Design._scale(len(np.unique(x)))(x)
            design.levels = design.levels.apply(mapper)
        if np.all(np.isclose(levels, 2)):
            design.pttypes = pd.Series(np.ones(len(design), dtype='u1'), design.runs)

        return design

    @staticmethod
    def from_fracfact(names, gen):
        """
        Construct a design from `pyDOE3.fracfact(...)`.

        Arguments
        ---------
        names (list[str]) -- List of variable names.
        gen (str) -- Yates-labelled generators for each variable, separated by
            spaces. Passed directly to pyDOE3.fracfact(...).

        Optional
        --------
        block (int) -- Block label for the runs in this design. (Default 1.)

        Returns
        -------
        Design -- The new design.
        """
        levels = pyDOE3.fracfact(gen)
        design = Design.from_levels(names, levels)
        design.pttypes = pd.Series(np.ones(len(design), dtype='u1'), design.runs)
        return design

    @staticmethod
    def from_ccdesign(names, center=(0, 0), alpha='orthogonal', face='circumscribed'):
        """
        Construct a design from `pyDOE3.ccdesign(...)`.

        Arguments
        ---------
        names (list[str]) -- List of variable names.

        Optional
        --------
        block (int) -- Block label for the runs in this design. (Default 1.)
        alpha (str) -- Passed to `pyDOE3.ccdesign(...)`.
        face (str) -- Passed to `pyDOE3.ccdesign(...)`.
        runs (np.ndarray) -- Array of labels for runs. (Default `None` results

        Returns
        -------
        Design -- The new design.
        """
        levels = pyDOE3.ccdesign(len(names), center=center, alpha=alpha, face=face)
        design = Design.from_levels(names, levels)
        design.pttypes = design.levels.apply(Design._pttype, axis=1).astype('u1')
        return design

    @staticmethod
    def from_centrepoints(names, n):
        """
        Construct a design from runs of centrepoints.

        Arguments
        ---------
        names (list[str]) -- List of variable names.
        n (int) -- Count of runs.

        Optional
        --------
        block (int) -- Block label for the runs in this design. (Default 1.)

        Returns
        -------
        Design -- The new design.
        """
        levels = np.zeros([n, len(names)])
        design = Design.from_levels(names, levels)
        design.pttypes = pd.Series(np.zeros(len(design), dtype='u1'), design.runs)
        return design

    @staticmethod
    def from_axial(names, exclude=None, magnitude=2.0):
        """
        Construct a design from runs of axial points.

        Arguments
        ---------
        names (list[str]) -- List of variable names.

        Optional
        --------
        exclude (list[str] or set[str]) -- Iterable of names to exclude from
            construction (the columns still exist, but no runs are added).
            (Default None.)
        magnitude (float) -- Magnitude of axial points. (Default 2.0.)
        block (int) -- Block label for the runs in this design. (Default 1.)

        Returns
        -------
        Design -- The new design.
        """
        if exclude is None:
            exclude = {}

        n = len(names)
        n_total = n-len(exclude)
        levels = np.zeros([2*n_total, n])
        j = 0
        for i, name in enumerate(names):
            if name not in exclude:
                levels[2*j, i] = -magnitude
                levels[2*j+1, i] = magnitude
                j += 1
        design = Design.from_levels(names, levels)
        design.pttypes = pd.Series(2*np.ones(len(design), dtype='u1'), design.runs)
        return design

    @staticmethod
    def _is_centre(point):
        # Origin is a centre point
        return np.all(np.isclose(point, 0.0))

    @staticmethod
    def _is_corner(point):
        # Same distance from the origin on all axes
        d = np.abs(point)
        return np.all(np.isclose(d.iloc[1:], d.iloc[0]))

    @staticmethod
    def _is_axial(point):
        # One non-zero entry (on an axis)
        return np.sum(~np.isclose(point, 0.0)) == 1

    @staticmethod
    def _pttype(point):
        '''
        NB: only works on non-transformed points. That is:
            - the origin is a centre point
            - any point that is the same distance from the origin on all axes is a corner point
            - any point with only one non-zero entry is an axial point
            - all other points cannot be classified
        '''
        if Design._is_centre(point):
            return 0
        elif Design._is_corner(point):
            return 1
        elif Design._is_axial(point):
            return 2
        else:
            return None

    @staticmethod
    def _scale(level_count):
        def scale(levels):
            s = 1 if level_count % 2 else 2
            scaled = levels * s
            return scaled - scaled.max() / 2
        return scale

    def _repr_(self):
        return self.to_df()

    def _repr_html_(self):
        return self.to_df().to_html()

@dataclass
class Transform:
    @staticmethod
    def from_map(map):
        """
        Construct an affine transform.

        Arguments
        ---------
        level_map (list[dict[float, float]]) -- A list of dictionaries that map
            from an existing level to a new level. Each dict corresponds to a
            variable and has two float keys, corresponding to existing levels
            and mapping to a float that is the new corresponding level. The two
            pairs exactly define an affine Transform. For example, the dict
            `{0: 10, 1: 20}` transforms `0` to `10` and `1` to `20`. All other
            points will be interpolated/extrapolated along a straight line:
            `0.5` transforms to `15`. As a result, the maps expressed in the
            dict need not correspond to the levels in the current design.

        Returns
        -------
        Affine(Transform) -- The transform.
        """
        [(l, lval), (r, rval)] = map.items()
        scale = (rval - lval) / (r - l)
        translate = lval - l * scale
        return Affine(
            scale=scale,
            translate=translate)

@dataclass
class Affine(Transform):
    """
    An Affine transform for transforming the levels in a Design.

    The scale is applied first, then the translation.

    Attributes
    ----------
    scale (np.ndarray) -- A matrix that multiplies experiment levels on the
        right to scale them into a new space.
    translate (np.ndarray) -- A vector with the same dimension as the variable
        space, that offsets the labels after they are scaled by `scale`.
    """
    scale: float
    translate: float

    def __call__(self, level):


        return level * self.scale + self.translate
