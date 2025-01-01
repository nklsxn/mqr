"""
======================================
Design of Experiments (:mod:`mqr.doe`)
======================================

.. currentmodule:: mqr.doe

User guide
    :doc:`/user_guide/design-of-experiments`

Detailed examples
    https://github.com/nklsxn/mqrpy-guide

.. rubric:: Classes
.. autosummary::
    :toctree: generated/

    Design
    Transform
    Affine

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

    Designs should normally be constructed using the from_* methods, which wrap
    calls to the pyDOE3 library. Designs can also be constructed manually,
    either from pyDOE3 or any other method, including directly from numpy arrays.
    See :doc:`/user_guide/design-of-experiments`.

    Designs are composable by concatenation with the `+` operator.

    Attributes
    ----------
    names : list[str]
        Names of variables.
    levels : np.ndarray
        Two-dimensional array containing the levels for each experiment, with a
        column for each variables and a row for each run.
    runs : np.ndarray
        Numerical labels for each run. Useful for tracking runs after randomisation.
    pttypes : array_like
        Numerical label for each point type:
        | 0 : centre point
        | 1 : corner point
        | 2 : axial point

    Examples
    --------
    >>> Design.from_full_fact(['x1', 'x2', 'x3'], [2, 2, 2])
       PtType    x1   x2   x3
    1       1  -1.0 -1.0 -1.0
    2       1   1.0 -1.0 -1.0
    3       1  -1.0  1.0 -1.0
    4       1   1.0  1.0 -1.0
    5       1  -1.0 -1.0  1.0
    6       1   1.0 -1.0  1.0
    7       1  -1.0  1.0  1.0
    8       1   1.0  1.0  1.0

    >>> d1 = Design.from_fracfact(['x1', 'x2', 'x3', 'x4'], 'a b c abc')
    >>> d2 = Design.from_centrepoints(['x1', 'x2', 'x3', 'x4'], 3)
    >>> d1.as_block(1) + d2.as_block(2)
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

        Parameters
        ----------
        n : int
            The number of replicates to create.
        label : str, optional
            Add a set of blocks with this name, labelling replicates by number.

        Notes
        -----
        Resets the `runs` indexing counting from 1.

        Returns
        -------
        :class:`Design`
            A new design that is a replicated version of this one.
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
        Return the same set of runs with a block label set to `level`.

        Parameters
        ----------
        level : int, str
            The label/level of this block.
        name : str
            The name of this collection of blocks. Useful if more than one level
            of blocking is required (ie. nested blocks).
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

        The resulting DataFrame is indexed with runs, has a column for point types
        if they have been defined, has columns for blocks if they are defined,
        and then columns for the inputs levels corresponding to each run/row.

        Returns
        -------
        pandas.DataFrame
            The design.
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

        Parameters
        ----------
        name : str
            The factor to isolate.
        ref_levels : float, optional
            The reference level assigned to all other variables.

        Returns
        -------
        pandas.DataFrame
            A dataframe with levels in `name` as rows and variable names as columns.
        """
        df = pd.DataFrame(columns=self.names, dtype=np.float64)
        df[name] = np.sort(np.unique(self.levels[name]))
        df.fillna(ref_levels, inplace=True)
        return df

    def randomise_runs(self, *order):
        """
        Return the same set of runs, randomised over their run labels.

        Parameters
        ----------
        order : {'PtType', block_name, factor_name}, optional
            Keep the runs ordered by this group(s). When more than one `order`
            is specified, the group labelled by the first element is sorted,
            then the group labelled by the second element is sorted within the
            groups of the first, and so on. The default `None` fully randomises
            the runs.

        Returns
        -------
        :class:`Design`
            A copy of this design, randomised.
        """
        rnd = np.random.choice(
            a=self.runs,
            size=len(self.runs),
            replace=False)
        df = self.to_df().loc[rnd]
        df.sort_values(list(order), inplace=True, kind='stable')
        new_order = df.index
        new_levels = self.levels.loc[new_order]
        if self.pttypes is not None:
            new_pttypes = self.pttypes.loc[new_order]
        else:
            new_pttypes = None
        new_blocks = self.blocks.loc[new_order]
        return Design(
            names=self.names,
            levels=new_levels,
            runs=new_order,
            pttypes=new_pttypes,
            blocks=new_blocks)

    def __add__(self, other):
        """
        Concatenate the runs of another design at the end of this design.

        Parameters
        ----------
        other : :class:`Design`
            The design to concatenate.

        Returns
        -------
        :class:`Design`
            This design and `other` concatenated into one, with run labels of
            `other` offset to continue from the end of this design.
        """
        if self.names != other.names:
            raise ValueError('Designs must contain the same variables.')

        new_runs = self.runs.append(other.runs + self.runs.max())
        if (self.pttypes is None) and (other.pttypes is None):
            new_pttypes = None
        elif (self.pttypes is not None) and (other.pttypes is None):
            new_pttypes = pd.concat([
                self.pttypes,
                pd.Series(np.nan, index=other.runs),
            ], axis=0).set_axis(new_runs)
        elif (self.pttypes is None) and (other.pttypes is not None):
            new_pttypes = pd.concat([
                pd.Series(np.nan, index=self.runs),
                other.pttypes
            ], axis=0).set_axis(new_runs)
        else:
            new_pttypes = pd.concat([self.pttypes, other.pttypes], axis=0).set_axis(new_runs)
        new_levels = pd.concat([self.levels, other.levels], axis=0)
        new_levels.set_index(new_runs, inplace=True)
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
        Apply transforms to the levels of this design.

        Parameters
        ----------
        transforms : {`Transform`, callable, dict}
            Keyword args with keywords corresponding to factor names of the design.
            Not all factors need to be listed/transformed.

            | :class:`Transform`
            |   Apply the given transform to the vector of levels.
            | callable
            |   Apply the given function to the vector of levels.
            | dict
            |   Use each level as a key in this dictionary and replace the level
            |   with the returned value. Each level must be a key in the dictionary.

        Returns
        -------
        :class:`Design`
            A copy of this design with new levels.
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

    @staticmethod
    def from_levels(names, levels, runs=None):
        """
        Construct a design from an array of levels.

        Parameters
        ----------
        names : list[str]
            List of variable names.
        levels : array_like
            Two-dimensional array of levels, with runs in rows and variables in
            columns.
        runs : array_like, optional
            Array of labels for runs. (Default `None` results in labels counting
            from 1.)

        Returns
        -------
        :class:`Design`
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
        Construct a design from `pyDOE3.fullfact(...)`.

        Parameters
        ----------
        names : list[str]
            List of variable names.
        levels : list[int]
            A list of counts of levels, passed directly to `pyDOE3.fullfact(...)`.
        scale_origin : bool, optional
            | `True`
            |   Scales the levels to have integer values centred at the origin.
            | `False`
            |   Levels are set directly from the `pyDOE3` output.

        Notes
        -----
        The point type column is only added when all factors have two levels.

        Returns
        -------
        :class:`Design`
        """
        coded_levels = pyDOE3.fullfact(levels)
        design = Design.from_levels(names, coded_levels)
        if scale_origin:
            value_counts = [len(np.unique(design.levels[name])) for name in design.names]
            mapper = lambda x: Design._scale(len(np.unique(x)))(x)
            design.levels = design.levels.apply(mapper)
        if np.all(np.isclose(levels, 2)) and pttypes:
            design.pttypes = pd.Series(np.ones(len(design), dtype='u1'), design.runs)

        return design

    @staticmethod
    def from_fracfact(names, gen):
        """
        Construct a design from `pyDOE3.fracfact(...)`.

        Parameters
        ----------
        names : list[str]
            List of variable names.
        gen : str
            Yates-labelled generators for each variable, separated by spaces.
            Passed directly to `pyDOE3.fracfact(...)`.

        Returns
        -------
        :class:`Design`
        """
        levels = pyDOE3.fracfact(gen)
        design = Design.from_levels(names, levels)
        design.pttypes = pd.Series(np.ones(len(design), dtype='u1'), design.runs)
        return design

    @staticmethod
    def from_ccdesign(names, center=(0, 0), alpha='orthogonal', face='circumscribed'):
        """
        Construct a design from `pyDOE3.ccdesign(...)`.

        Parameters
        ----------
        names : list[str]
            List of variable names.
        center : tuple, optional
            Passed to `pyDOE3.ccdesign(...)`.
        alpha : str, optional
            Passed to `pyDOE3.ccdesign(...)`.
        face : str, optional
            Passed to `pyDOE3.ccdesign(...)`.

        Returns
        -------
        :class:`Design`
        """
        levels = pyDOE3.ccdesign(len(names), center=center, alpha=alpha, face=face)
        design = Design.from_levels(names, levels)
        design.pttypes = design.levels.apply(Design._pttype, axis=1).astype('u1')
        return design

    @staticmethod
    def from_centrepoints(names, n):
        """
        Construct a design from runs of centrepoints.

        Parameters
        ----------
        names : list[str]
            List of variable names.
        n : int
            Count of runs.

        Returns
        -------
        :class:`Design`
        """
        levels = np.zeros([n, len(names)])
        design = Design.from_levels(names, levels)
        design.pttypes = pd.Series(np.zeros(len(design), dtype='u1'), design.runs)
        return design

    @staticmethod
    def from_axial(names, exclude=None, magnitude=None):
        """
        Construct a design from runs of axial points.

        Parameters
        ----------
        names : list[str]
            List of variable names.
        exclude : list[str] or set[str], optional
            Iterable of names to exclude from construction (the columns still
            exist, but no runs are added).
        magnitude : float, optional
            Magnitude of axial points. Default is sqrt(len(names)).

        Returns
        -------
        :class:`Design`
        """
        if exclude is None:
            exclude = {}

        if magnitude is None:
            magnitude = np.sqrt(len(names))

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
        # Same non-zero distance from the origin on all axes
        d = np.abs(point)
        return (d.iloc[0] > 0) and np.all(np.isclose(d.iloc[1:], d.iloc[0]))

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
            - all other points are not classified
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

        See :doc:`/user_guide/design-of-experiments` for examples.

        Parameters
        ----------
        map : dict[float, float]
            A dictionary that maps from an existing level to a new level.
            The dict has two float keys, corresponding to existing levels and
            mapping to a float that is the new corresponding level. The two
            pairs exactly define an affine Transform. For example, the dict
            `{0: 10, 1: 20}` transforms `0` to `10` and `1` to `20`. All other
            points will be interpolated/extrapolated along a straight line: `0.5`
            transforms to `15`. As a result, the maps expressed in the dict need
            not correspond to the levels in the current design.

        Returns
        -------
        :class:`Affine`
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
    See :doc:`/user_guide/design-of-experiments` for examples.

    Attributes
    ----------
    scale : float
        Multiplies experiment levels.
    translate : float
        Offsets the experiment levels after they are scaled by `scale`.
    """
    scale: float
    translate: float

    def __call__(self, level):
        """
        Applies this transform to level values.

        Parameters
        ----------
        levels : array_like
            Levels to transform, usually a column from :attr:`Design.levels`.

        Returns
        -------
        array_like
            Transformed levels.
        """
        return level * self.scale + self.translate
