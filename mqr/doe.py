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
    levels: np.ndarray
    runs: np.ndarray
    pttypes: np.ndarray
    blocks: np.ndarray

    def replicate(self, n):
        """
        Create a new design with each run replicated `n` times.

        Arguments
        ---------
        n (int) -- The number of replicates to create.

        Returns
        -------
        Design -- A new design that is a replicated version of this one.
        """
        m, _ = self.levels.shape
        return Design(
            names=self.names,
            levels=np.repeat(self.levels, n, axis=0),
            runs=np.arange(m*n)+1,
            pttypes=np.repeat(self.pttypes, n, axis=0),
            blocks=np.repeat(self.blocks, n, axis=0))


    def as_block(self, block):
        """
        Return the same set of runs with their block label set to `block`.

        Arguments
        ---------
        block (int) -- New block label.

        Returns
        -------
        Design -- A copy of this design with a new block label.
        """
        return Design(
            names=self.names,
            levels=self.levels,
            runs=self.runs,
            pttypes=self.pttypes,
            blocks=np.full(self.blocks.shape, block))

    def to_df(self):
        """
        Construct a dataframe representation of the design.

        Returns
        -------
        pd.DataFrame -- The design.
        """
        df = pd.DataFrame(
            {'PtType': self.pttypes, 'Block': self.blocks},
            index=self.runs)
        df[self.names] = self.levels
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
        df[name] = np.sort(np.unique(self.levels[:, 0]))
        df.fillna(ref_levels, inplace=True)
        return df

    def randomise_runs(self, preserve_blocks=True):
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
        rnd = np.empty(self.runs.shape, dtype=int)
        if preserve_blocks:
            for blk in np.unique(self.blocks):
                idx = self.blocks == blk
                rnd[idx] = np.random.choice(
                    a=self.runs[idx],
                    size=np.sum(idx),
                    replace=False)
            assert np.all(self.blocks[rnd-1] == self.blocks)
        else:
            rnd = np.random.choice(
                a=self.runs,
                size=len(self.runs),
                replace=False)

        return Design(
            names=self.names,
            levels=self.levels[rnd-1],
            runs=rnd,
            pttypes=self.pttypes[rnd-1],
            blocks=self.blocks[rnd-1])

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

        run_offset = np.max(self.runs)

        new_levels = np.vstack([self.levels, other.levels])
        new_runs = np.hstack([self.runs, other.runs + run_offset])
        new_pttypes = np.concatenate([self.pttypes, other.pttypes])
        new_blocks = np.concatenate([self.blocks, other.blocks])
        return Design(
            names=self.names,
            levels=new_levels,
            runs=new_runs,
            pttypes=new_pttypes,
            blocks=new_blocks)

    def __matmul__(self, transform):
        """
        Apply a transform to the levels of this design.

        Arguments
        ---------
        transform (Transform) -- Transform to apply.

        Returns
        -------
        Design -- A copy of this design with new levels.
        """
        return Design(
            names=self.names,
            levels=transform(self.levels),
            runs=self.runs,
            pttypes=self.pttypes,
            blocks=self.blocks)

    @staticmethod
    def from_levels(names, levels, runs=None, block=1):
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
        if levels.shape[1] != len(names):
            raise AttributeError('Length of `names` does not match number of columns in `levels`.')
        m = levels.shape[0]
        if runs is None:
            runs = np.arange(m) + 1
        pttypes = np.apply_along_axis(Design._pttype, 1, levels)
        m, _ = levels.shape
        blocks = np.full(m, block)
        return Design(
            names=names,
            levels=levels,
            runs=runs,
            pttypes=pttypes,
            blocks=blocks)

    @staticmethod
    def from_fullfact(names, levels, block=1):
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
        exp_levels = pyDOE3.fullfact(levels)
        for i, lvl in enumerate(levels):
            exp_levels[:, i] = Design._scale(exp_levels[:, i], lvl)
        m = exp_levels.shape[0]
        runs = np.arange(m) + 1
        pttypes = np.full(m, 1)
        blocks = np.full(m, block)
        return Design(
            names=names,
            levels=exp_levels,
            runs=runs,
            pttypes=pttypes,
            blocks=blocks)

    @staticmethod
    def from_fracfact(names, gen, block=1):
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
        m = levels.shape[0]
        runs = np.arange(m) + 1
        pttypes = np.full(m, 1)
        blocks = np.full(m, block)
        return Design(
            names=names,
            levels=levels,
            runs=runs,
            pttypes=pttypes,
            blocks=blocks)

    @staticmethod
    def from_ccdesign(names, block=1, alpha='orthogonal', face='circumscribed'):
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
        n = len(names)
        levels = pyDOE3.ccdesign(n, center=(0, 0), alpha=alpha, face=face) #[2**n:]
        m = levels.shape[0]
        runs = np.arange(m) + 1
        pttypes = np.apply_along_axis(Design._pttype, 1, levels)
        blocks = np.full(m, block)
        return Design(
            names=names,
            levels=levels,
            runs=runs,
            pttypes=pttypes,
            blocks=blocks)

    @staticmethod
    def from_centrepoints(names, n, block=1):
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
        m = levels.shape[0]
        runs = np.arange(m) + 1
        pttypes = np.full(n, 0)
        blocks = np.full(n, block)
        return Design(
            names=names,
            levels=levels,
            runs=runs,
            pttypes=pttypes,
            blocks=blocks)

    @staticmethod
    def from_axial(names, exclude=None, magnitude=2.0, block=1):
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
        m = levels.shape[0]
        runs = np.arange(m) + 1
        pttypes = np.full(m, 2)
        blocks = np.full(m, block)
        return Design(
            names=names,
            levels=levels,
            runs=runs,
            pttypes=pttypes,
            blocks=blocks)

    @staticmethod
    def _pttype(point):
        '''
        NB: only works on non-transformed points. That is:
            - the origin is a centre point
            - any point that is the same distance from the origin on all axes is a corner point
            - any point with only one non-zero entry is an axial point
            - all other points cannot be classified
        '''
        if np.all(np.isclose(point, 0.0)):
            # Centre point
            return 0
        elif np.all(np.isclose(np.abs(point), np.mean(np.abs(point)))):
            # Corner point
            return 1
        elif np.sum(~np.isclose(point, 0.0)) == 1:
            # Axial point
            return 2
        elif np.sum(np.isclose(np.abs(point), 1.0)) == 2:
            # Edge point
            return 3
        else:
            raise AttributeError(f'Could not determine type of point: {point}.')

    @staticmethod
    def _scale(levels, lvl):
        s = 1 if lvl % 2 else 2
        scaled = levels * s
        return scaled - scaled[-1] / 2

    def _repr_(self):
        return self.to_df()

    def _repr_html_(self):
        return self.to_df().to_html()

@dataclass
class Transform:
    @staticmethod
    def from_map(level_map):
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
        n = len(level_map)
        scale = np.empty(n)
        translate = np.empty(n)
        for i, d in enumerate(level_map):
            [(l, lval), (r, rval)] = d.items()
            scale[i] = (rval - lval) / (r - l)
            translate[i] = lval - l * scale[i]
        return Affine(
            scale=scale,
            translate=translate)

    @staticmethod
    def from_categories(category_map):
        """
        Construct an affine transform.

        Arguments
        ---------
        category_map (list[dict[float, object]]) -- A list of dictionaries that
            map from an existing level to a new level. Each dict corresponds to
            a variable and has two float keys, corresponding to existing levels
            and mapping to any object that is the new corresponding level.
            Unlike the Affine transform, the Categorical transform does not
            interpolate/extrapolate values; each level in the design must appear
            in the dict.

        Returns
        -------
        Categorical(Transform) -- The transform.
        """
        return Categorical(category_map)

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
    scale: np.ndarray
    translate: np.ndarray

    def __call__(self, levels):
        return levels * self.scale + self.translate

@dataclass
class Categorical(Transform):
    """
    A transform for categorical levels in a Design.

    .

    Attributes
    ----------
    categories (list[dict]) -- List of maps corresponding to variables; existing
        levels for keys and new categorical levels for values.
    dtype (str) -- Type of the new category values.
    """
    categories: list[dict]
    dtype: str = field(default='<U64')

    def __call__(self, levels):
        if levels.shape[1] != len(self.categories):
            raise AttributeError(f'wrong number of variables (got {levels.shape[1]} columns, expected {len(self.categories)})')

        res = np.empty(levels.shape, dtype=self.dtype)
        for i in range(len(self.categories)):
            tr = lambda x: self.categories[i][x]
            res[:, i] = np.vectorize(tr)(levels[:, i])
        return res
