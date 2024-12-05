import mqr

import numpy as np
import pandas as pd
import pyDOE3
import pytest

def test_Design_replicate():
    names = ['a', 'b']
    levels = [2, 2]
    design = (
        mqr.doe.Design.from_fullfact(names, levels).as_block(1) +
        mqr.doe.Design.from_centrepoints(names, 2).as_block(2))

    replicated = design.replicate(2)
    assert np.all(replicated.runs.values[0::2] == replicated.runs.values[1::2] - 1)
    assert np.all(replicated.pttypes.values[0::2] == replicated.pttypes.values[1::2])
    assert np.all(replicated.levels.values[0::2] == replicated.levels.values[1::2])

    replicated = design.replicate(2, 'Rep')
    assert np.all(replicated.runs.values[0::2] == replicated.runs.values[1::2] - 1)
    assert np.all(replicated.pttypes.values[0::2] == replicated.pttypes.values[1::2])
    assert np.all(replicated.levels.values[0::2] == replicated.levels.values[1::2])
    assert np.all(replicated.blocks['Rep'][0::2] == 1)
    assert np.all(replicated.blocks['Rep'][1::2] == 2)

    levels = [2, 3]
    design = mqr.doe.Design.from_fullfact(names, levels)

    replicated = design.replicate(2)
    assert np.all(replicated.runs.values[0::2] == replicated.runs.values[1::2] - 1)
    assert np.all(replicated.levels.values[0::2] == replicated.levels.values[1::2])

def test_Design_as_block():
    names = ['a', 'b']
    levels = [2, 2]
    design = (
        mqr.doe.Design.from_fullfact(names, levels).as_block(1) +
        mqr.doe.Design.from_centrepoints(names, 2).as_block(2))

    assert np.all(design.as_block(4).blocks == 4)

def test_Design_to_df():
    names = ['a', 'b']
    levels = [2, 2]
    design = (
        mqr.doe.Design.from_fullfact(names, levels).as_block(1) +
        mqr.doe.Design.from_centrepoints(names, 2).as_block(2))
    df = design.to_df()

    assert np.all(df.index == design.runs)
    assert np.all(df['PtType'] == design.pttypes)
    assert np.all(df[['Block']] == design.blocks)
    assert np.all(df.loc[:, 'a':'b'] == design.levels)

def test_Design_get_factor_df():
    names = ['a', 'b']
    levels = [2, 2]
    design = (
        mqr.doe.Design.from_fullfact(names, levels).as_block(1) +
        mqr.doe.Design.from_centrepoints(names, 2).as_block(2))
    factor_df = design.get_factor_df('a', ref_levels=1.23)

    assert list(factor_df['a']) == sorted(design.levels['a'].unique())
    assert all(factor_df['b'] == 1.23)

def test_Design_randomise_runs():
    names = ['a', 'b', 'c']
    levels = [2, 2, 3]
    design = mqr.doe.Design.from_fullfact(names, levels)
    df = design.to_df()

    # complete randomisation
    np.random.seed(0)
    rdesign = design.randomise_runs()
    rdf = rdesign.to_df()

    assert set(design.runs) == set(rdesign.runs)
    assert list(design.runs) != list(rdesign.runs) # True when seeded with 0
    for run in design.runs:
        assert list(df.loc[run]) == list(rdf.loc[run])

    # one level ordering
    np.random.seed(0)
    rdesign = design.randomise_runs('a')
    rdf = rdesign.to_df()

    assert set(design.runs) == set(rdesign.runs)
    assert sorted(rdf['a']) == list(rdf['a'])
    assert list(design.runs) != list(rdesign.runs) # True when seeded with 0
    for run in design.runs:
        assert list(df.loc[run]) == list(rdf.loc[run])

    # two level ordering
    np.random.seed(0)
    rdesign = design.randomise_runs('a', 'b')
    rdf = rdesign.to_df()

    assert set(design.runs) == set(rdesign.runs)
    assert sorted(rdf['a']) == list(rdf['a'])
    for level in rdf['a'].unique():
        bs = rdf.query(f'a == {level}')['b']
        print(bs)
        assert sorted(bs) == list(bs)
    assert list(design.runs) != list(rdesign.runs) # True when seeded with 0
    for run in design.runs:
        assert list(df.loc[run]) == list(rdf.loc[run])

    # complete randomisation
    np.random.seed(0)
    design.pttypes = pd.Series(1, index=design.runs)
    df = design.to_df()
    rdesign = design.randomise_runs()
    rdf = rdesign.to_df()

    assert set(design.runs) == set(rdesign.runs)
    assert list(design.runs) != list(rdesign.runs) # True when seeded with 0
    for run in design.runs:
        assert list(df.loc[run]) == list(rdf.loc[run])

def test_Design_add():
    names = ['a', 'b']
    levels = [2, 2]
    design1 = mqr.doe.Design.from_fullfact(names, levels).as_block(1)
    design2 = mqr.doe.Design.from_centrepoints(names, 3).as_block(2)

    design = design1 + design2
    assert list(design.runs) == list(design1.runs) + list(design2.runs + len(design1))
    assert list(design.pttypes) == list(design1.pttypes) + list(design2.pttypes)
    assert np.all(design.levels.values == pd.concat([design1.levels, design2.levels]).values)
    assert np.all(design.blocks.values == pd.concat([design1.blocks, design2.blocks]).values)

    design1.pttypes = None
    design = design1 + design2
    assert list(design.runs) == list(design1.runs) + list(design2.runs + len(design1))
    assert np.all(np.isnan(design.pttypes[:len(design1)]))
    assert list(design.pttypes[len(design1):] == list(design2.pttypes))
    assert np.all(design.levels.values == pd.concat([design1.levels, design2.levels]).values)
    assert np.all(design.blocks.values == pd.concat([design1.blocks, design2.blocks]).values)

    design1 = mqr.doe.Design.from_fullfact(names, levels).as_block(1)
    design2.pttypes = None
    design = design1 + design2
    assert list(design.runs) == list(design1.runs) + list(design2.runs + len(design1))
    assert list(design.pttypes[:len(design1)] == list(design1.pttypes))
    assert np.all(np.isnan(design.pttypes[len(design1):]))
    assert np.all(design.levels.values == pd.concat([design1.levels, design2.levels]).values)
    assert np.all(design.blocks.values == pd.concat([design1.blocks, design2.blocks]).values)

    design1.pttypes = None
    design2.pttypes = None
    design = design1 + design2
    assert list(design.runs) == list(design1.runs) + list(design2.runs + len(design1))
    assert design.pttypes == None
    assert np.all(design.levels.values == pd.concat([design1.levels, design2.levels]).values)
    assert np.all(design.blocks.values == pd.concat([design1.blocks, design2.blocks]).values)

def test_Design_transform():
    names = ['a', 'b', 'c']
    levels = [2, 2, 2]
    design = mqr.doe.Design.from_fullfact(names, levels).as_block(2)
    
    tr = design.transform(
        a={-1: 5, 1: 15},
        c=lambda x: 2*x)
    exp_levels = np.array([
        [5, 15, 5, 15, 5, 15, 5, 15],
        [-1, -1, 1, 1, -1, -1, 1, 1],
        [-2, -2, -2, -2, 2, 2, 2, 2],
    ]).T

    assert list(tr.runs) == list(design.runs)
    assert list(tr.pttypes) == list(design.pttypes)
    assert np.all(tr.blocks == design.blocks)
    assert np.all(tr.levels.values == exp_levels)

def test_Design_from_levels():
    names = ['a', 'b', 'c']
    levels = np.array([
        [-1, -1, 1, 1],
        [-1, 1, -1, 1],
        [1, -1, -1, 1],
    ]).T
    runs = pd.Index([3, 4, 5, 6])
    
    design = mqr.doe.Design.from_levels(names, levels, runs)

    assert list(design.runs) == list(runs)
    assert np.all(design.levels.values == levels)

def test_Design_from_fullfact():
    names = ['a', 'b']
    levels = [2, 2]    
    exp_runs = list(range(1, 2 * 2 + 1))
    exp_levels = pyDOE3.fullfact(levels)

    design = mqr.doe.Design.from_fullfact(names, levels, scale_origin=True, pttypes=True)
    assert list(design.runs) == exp_runs
    assert np.all(design.pttypes == 1)
    assert np.all(design.levels == exp_levels * 2 - 1)

    design = mqr.doe.Design.from_fullfact(names, levels, scale_origin=True, pttypes=False)
    assert list(design.runs) == exp_runs
    assert design.pttypes == None
    assert np.all(design.levels == exp_levels * 2 - 1)

    design = mqr.doe.Design.from_fullfact(names, levels, scale_origin=False, pttypes=True)
    assert list(design.runs) == exp_runs
    assert np.all(design.pttypes == 1)
    assert np.all(design.levels == exp_levels)

    design = mqr.doe.Design.from_fullfact(names, levels, scale_origin=False, pttypes=False)
    assert list(design.runs) == exp_runs
    assert np.all(design.pttypes == None)
    assert np.all(design.levels == exp_levels)

def test_Design_from_fracfact():
    names = ['a', 'b', 'c']
    gen = 'a b ab'
    exp_runs = list(range(1, 2 * 2 + 1))
    exp_levels = pyDOE3.fracfact(gen)

    design = mqr.doe.Design.from_fracfact(names, gen)
    assert list(design.runs) == exp_runs
    assert np.all(design.pttypes == 1)
    assert np.all(design.levels == exp_levels)

def test_Design_from_ccdesign():
    names = ['a', 'b', 'c']
    exp_runs = list(range(1, 2 ** 3 + 3 * 2 + 1))
    exp_levels = pyDOE3.ccdesign(len(names), center=(0, 0))

    design = mqr.doe.Design.from_ccdesign(names)
    assert list(design.runs) == exp_runs
    assert np.all(design.pttypes[:8] == 1)
    assert np.all(design.pttypes[8:] == 2)
    assert np.all(np.isclose(design.levels, exp_levels))

def test_Design_from_centrepoints():
    names = ['a', 'b', 'c']
    exp_runs = list(range(1, 5))

    design = mqr.doe.Design.from_centrepoints(names, 4)
    assert list(design.runs) == exp_runs
    assert np.all(design.pttypes == 0)
    assert np.all(design.levels == 0)

def test_Design_from_axial():
    names = ['a', 'b', 'c']
    exp_runs = list(range(1, 5))
    exp_levels = pyDOE3.ccdesign(len(names), center=(0, 0))[8:]

    design = mqr.doe.Design.from_axial(names, exclude=['b'])
    assert list(design.runs) == exp_runs
    assert list(design.pttypes == 2)
    assert np.all(design.levels['a'] == np.sqrt(3) * np.array([-1, 1, 0, 0]))
    assert np.all(design.levels['b'] == 0)
    assert np.all(design.levels['c'] == np.sqrt(3) * np.array([0, 0, -1, 1]))

    design = mqr.doe.Design.from_axial(names, exclude=['b'], magnitude=2)
    assert list(design.runs) == exp_runs
    assert list(design.pttypes == 2)
    assert np.all(design.levels['a'] == 2 * np.array([-1, 1, 0, 0]))
    assert np.all(design.levels['b'] == 0)
    assert np.all(design.levels['c'] == 2 * np.array([0, 0, -1, 1]))

    design = mqr.doe.Design.from_axial(names)
    exp_runs = list(range(1, 7))
    assert list(design.runs) == exp_runs
    assert list(design.pttypes == 2)
    assert np.all(design.levels['a'] == np.sqrt(3) * np.array([-1, 1, 0, 0, 0, 0]))
    assert np.all(design.levels['b'] == np.sqrt(3) * np.array([0, 0, -1, 1, 0, 0]))
    assert np.all(design.levels['c'] == np.sqrt(3) * np.array([0, 0, 0, 0, -1, 1]))

def test_Design_is_centre():
    assert mqr.doe.Design._is_centre(pd.Series([0, 0, 0, 0]))
    assert not mqr.doe.Design._is_centre(pd.Series([0, 0, 1]))
    assert not mqr.doe.Design._is_centre(pd.Series([1, 1, 1]))
    assert mqr.doe.Design._is_centre(pd.Series([0]))
    assert not mqr.doe.Design._is_centre(pd.Series([1]))

def test_Design_is_corner():
    assert mqr.doe.Design._is_corner(pd.Series([1, -1, -1, 1]))
    assert not mqr.doe.Design._is_corner(pd.Series([1, -1, -1, 0]))
    assert not mqr.doe.Design._is_corner(pd.Series([0, 0, 0]))
    assert not mqr.doe.Design._is_corner(pd.Series([0]))
    assert mqr.doe.Design._is_corner(pd.Series([1]))

def test_Design_is_axial():
    assert mqr.doe.Design._is_axial(pd.Series([0, 1, 0, 0]))
    assert mqr.doe.Design._is_axial(pd.Series([0, 0, -1, 0]))
    assert not mqr.doe.Design._is_axial(pd.Series([0, 1, -1, 0]))
    assert not mqr.doe.Design._is_axial(pd.Series([0, 1, -1, 0]))
    assert not mqr.doe.Design._is_axial(pd.Series([0]))
    assert mqr.doe.Design._is_axial(pd.Series([1]))

def test_Design_pttype():
    assert mqr.doe.Design._pttype(pd.Series([0, 0, 0])) == 0
    assert mqr.doe.Design._pttype(pd.Series([-1, 1, 1])) == 1
    assert mqr.doe.Design._pttype(pd.Series([-1, 0, 0])) == 2
    assert mqr.doe.Design._pttype(pd.Series([-1])) == 1 # corner takes precedence when only one dimension
    assert mqr.doe.Design._pttype(pd.Series([0])) == 0
    assert mqr.doe.Design._pttype(pd.Series([1, 0, -1])) is None

def test_scale():
    assert list(mqr.doe.Design._scale(2)(np.array([0, 1]))) == list([-1, 1])
    assert list(mqr.doe.Design._scale(3)(np.array([0, 1, 2]))) == list([-1, 0, 1])
    assert list(mqr.doe.Design._scale(4)(np.array([0, 1, 2, 3]))) == list([-3, -1, 1, 3])
    assert list(mqr.doe.Design._scale(5)(np.array([0, 1, 2, 3, 4]))) == list([-2, -1, 0, 1, 2])

def test_Transform_from_map():
    assert list(mqr.doe.Transform.from_map({1: 4, 2: 8})(np.array([0, 1, 2]))) == list([0, 4, 8])
    assert list(mqr.doe.Transform.from_map({0: 4, 2: 8})(np.array([0, 1, 2]))) == list([4, 6, 8])
    assert list(mqr.doe.Transform.from_map({1: 4, 4: -8})(np.array([0, 1, 2]))) == list([8, 4, 0])
