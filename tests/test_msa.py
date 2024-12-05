import mqr

import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def grr_data():
    design = mqr.doe.Design.from_fullfact(
        names=['Part', 'Inspector'],
        levels=[10, 3],
        scale_origin=False)
    design = design.replicate(3, label='Test')
    design.levels = (design.levels + 1).astype(int)
    data = design.to_df()
    data['Impedance'] = np.array([
        37, 38, 37, 42, 41, 43, 30, 31, 31, 42, 43, 42, 28, 30, 29, 42, 42, 43, 25, 26, 27, 40, 40, 40, 25, 25, 25, 35, 34, 34,
        41, 41, 40, 42, 42, 42, 31, 31, 31, 43, 43, 43, 29, 30, 29, 45, 45, 45, 28, 28, 30, 43, 42, 42, 27, 29, 28, 35, 35, 34,
        41, 42, 41, 43, 42, 43, 29, 30, 28, 42, 42, 42, 31, 29, 29, 44, 46, 45, 29, 27, 27, 43, 43, 41, 26, 26, 26, 35, 34, 35,
    ])
    return data

def test_NameMapping_init():
    name_mapping = mqr.msa.NameMapping()
    name_mapping.measurement == 'measurement'
    name_mapping.part == 'part'
    name_mapping.operator == 'operator'

    name_mapping = mqr.msa.NameMapping(
        measurement='m',
        part='p',
        operator='o',)
    name_mapping.measurement == 'm'
    name_mapping.part == 'p'
    name_mapping.operator == 'o'

def test_GRR_init(grr_data):
    names = mqr.msa.NameMapping(
        part='Part',
        operator='Inspector',
        measurement='Impedance',)
    grr = mqr.msa.GRR(grr_data, 3.0, names, include_interaction=True)
    assert grr.formula == 'Impedance ~ C(Part) + C(Inspector) + C(Part):C(Inspector)'
    assert list(grr.counts) == [10, 3]
    assert grr.replicates == 3
    assert grr.model
    assert grr.regression_result

    grr = mqr.msa.GRR(grr_data, 3.0, names, include_interaction=False)
    assert grr.formula == 'Impedance ~ C(Part) + C(Inspector)'
    assert list(grr.counts) == [10, 3]
    assert grr.replicates == 3
    assert grr.model
    assert grr.regression_result

def test_VarianceTable_init(grr_data):
    names = mqr.msa.NameMapping(
        part='Part',
        operator='Inspector',
        measurement='Impedance',)
    grr = mqr.msa.GRR(grr_data, 3.0, names)
    var = mqr.msa.VarianceTable(grr)
    varcomp = var.table['VarComp']

    assert var.grr == grr

def test_VarianceTable_varcomp(grr_data):
    """
    Verifies variance components against Table 8.8 in [1]_.

    References
    ----------
    .. [1]  Montgomery, D. C. (2009).
            Statistical quality control (Vol. 7).
            New York: Wiley.
    """
    names = mqr.msa.NameMapping(
        part='Part',
        operator='Inspector',
        measurement='Impedance',)
    grr = mqr.msa.GRR(grr_data, 3.0, names)
    var = mqr.msa.VarianceTable(grr)
    varcomp = var.table['VarComp']
    
    assert varcomp['Part-to-Part'] == pytest.approx(48.2926, abs=1e-4)
    assert varcomp['Operator'] == pytest.approx(0.5646, abs=1e-4)
    assert varcomp['Operator*Part'] == pytest.approx(0.7280, abs=1e-4)
    assert varcomp['Repeatability'] == pytest.approx(0.5111, abs=1e-4)
    assert varcomp['Gauge RR'] + varcomp['Part-to-Part'] == varcomp['Total']
    assert varcomp['Repeatability'] + varcomp['Reproducibility'] == varcomp['Gauge RR']
    assert varcomp['Operator'] + varcomp['Operator*Part'] == varcomp['Reproducibility']

    grr = mqr.msa.GRR(grr_data, 3.0, names, include_interaction=False)
    var = mqr.msa.VarianceTable(grr)
    varcomp = var.table['VarComp']

    assert varcomp['Gauge RR'] + varcomp['Part-to-Part'] == varcomp['Total']
    assert varcomp['Repeatability'] + varcomp['Reproducibility'] == varcomp['Gauge RR']
    assert varcomp['Operator'] == varcomp['Reproducibility']

def test_VarianceTable_set_discrimination(grr_data):
    names = mqr.msa.NameMapping(
        part='Part',
        operator='Inspector',
        measurement='Impedance',)
    grr = mqr.msa.GRR(grr_data, 3.0, names)
    var = mqr.msa.VarianceTable(grr)

    assert var.discrimination == pytest.approx(54.5, abs=1e-1)

def test_VarianceTable_set_num_distinct_cats(grr_data):
    names = mqr.msa.NameMapping(
        part='Part',
        operator='Inspector',
        measurement='Impedance',)
    grr = mqr.msa.GRR(grr_data, 3.0, names)
    var = mqr.msa.VarianceTable(grr)

    assert round(var.num_distinct_cats) == 7
