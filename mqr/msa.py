"""
============================================
Measurement system analysis (:mod:`mqr.msa`)
============================================

.. currentmodule:: mqr.msa

User guide
    :doc:`/user_guide/measurement-system-analysis`

Detailed examples
    https://github.com/nklsxn/mqrpy-guide

Construction and presentation of gauge repeatability and reproducibility study.

.. rubric:: Construction
.. autosummary::
    :toctree: generated/

    GRR
    NameMapping

.. rubric:: Results
.. autosummary::
    :toctree: generated/

    VarianceTable
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from patsy import ModelDesc
import statsmodels
from statsmodels.formula.api import mixedlm

import mqr

@dataclass
class NameMapping:
    """
    A definition of terms that maps variables in an experiment to standard terms
    in a GRR study.

    Attributes
    ----------
    measurement : str, optional
        Column name referring to the observation/KPI. Default 'measurement'.
    part : str, optional
        Column name referring to the categorical part ID. Default 'part'.
    operator : str, optional
        Column name referring to the categorical operator ID. Default 'operator'.
    replicate : str, optional
        Column name referring to the categorical replicate ID. Default 'replicate'.
    """
    measurement: str = field(default='measurement')
    part: str = field(default='part')
    operator: str = field(default='operator')
    _m: str = field(repr=False, default=None)
    _p: str = field(repr=False, default=None)
    _o: str = field(repr=False, default=None)

    def __init__(self, *, measurement=None, part=None, operator=None):
        """
        Construct NameMapping.

        Parameters
        ----------
        measurement : str, optional
            see Attribute.
        part : str, optional
            see Attribute.
        operation : str, optional
            see Attribute.
        """
        if measurement:
            self.measurement = self._m = measurement
        if part:
            self.part = self._p = part
        if operator:
            self.operator = self._o = operator

@dataclass
class GRR:
    """
    A Gauge Repeatability and Reliability study.

    Constructs a model and calls `statsmodels.formula.api.ols`.

    Attributes
    ----------
    data : pd.DataFrame
        Experiment runs and measurements, including columns for measurement, part,
        operator and replicate. See `NameMapping` for how to name these columns.
    tolerance : np.float64
        Width of the tolerance of the process in the same units as the measurements.
    names : NameMapping
        A name mapping that defines how custom names translate to the standard
        names used in this library. See `mqr.msa.NameMapping`.
    include_interaction : bool
        When `True`, include terms in the ANOVA for the interaction between
        operator and part.
    include_intercept : bool
        When `True`, include terms in the ANOVA for the intercept.
    nsigma : int
        Target capability of the process.
    formula : str, automatic
        Formula passed to statsmodels for regression.
    counts : tuple[int], automatic
        Number of measurements, and number of unique levels in categorical
        variables part, operator and replicate.
    model : automatic
        Linear model for OLS.
    regression_result : statsmodels.regression.linear_model.RegressionResultsWrapper, automatic
        Result of calling `fit()` on the `model`.
    """
    data: pd.DataFrame
    tolerance: np.float64
    names: NameMapping
    include_interaction: bool
    include_intercept: bool
    nsigma: int

    formula: str
    counts: tuple[int]

    model: statsmodels.regression.linear_model.OLS
    regression_result: statsmodels.regression.linear_model.RegressionResultsWrapper

    def __init__(
        self,
        data:pd.DataFrame,
        tolerance:float,
        names:NameMapping=None,
        include_interaction=True,
        include_intercept=True,
        nsigma=6):
        """
        Construct GRR.

        Arguments
        ---------
        data : pd.DataFrame
            See attribute.
        tolerance : float
            See attribute.
        names : mqr.msa.NameMapping, optional
            See attribute. Defaults to `NameMapping()`.
        include_interaction : bool, optional
            See attribute.
        include_intercept : bool, optional
            See attribute.
        nsigma : float
            See attribute.
        """
        self.data = data
        self.tolerance = tolerance
        self.include_interaction = include_interaction
        self.include_intercept = include_intercept
        self.names = names if names is not None else NameMapping()
        self.nsigma = nsigma

        self._configure_counts()
        self._configure_formula()

        self._fit_model(data)

    def _configure_counts(self):
        cols = [self.names._p, self.names._o]
        data = self.data.loc[:, cols]
        self.counts = data.nunique(axis=0)
        reps = len(data) / np.prod(self.counts)
        if reps != int(reps):
            raise ValueError('Unbalanced designs not supported.')
        self.replicates = int(reps)

    def _configure_formula(self):
        name_m = self.names._m
        name_p = self.names._p
        name_o = self.names._o
        combn = '*' if self.include_interaction else '+'
        intercept = '+ 1' if self.include_intercept else '- 1'
        formula = f'{name_m} ~ C({name_p}) {combn} C({name_o}) {intercept}'
        self.formula = ModelDesc.from_formula(formula).describe()

    def _fit_model(self, data):
        self.model = statsmodels.formula.api.ols(self.formula, self.data)
        self.regression_result = self.model.fit()

    def _repr_html_(self):
        return SummaryTable(self)._repr_html_()

class SummaryTable:
    _grr: GRR

    def __init__(self, grr: GRR):
        self._grr = grr

    def _repr_html_(self):
        grr = self._grr
        html = f'''
            <table>
            <thead>
                <caption>Gauge Repeatability and Reliability Study</caption>
            </thead>
            <tbody>
                <thead>
                    <tr>
                        <th scope='col'></th>
                        <th scope='col'>Measurement</td>
                        <th scope='col'>Part</td>
                        <th scope='col'>Operator</td>
                    </tr>
                </thead>
                <tr>
                    <th scope='row'>Variable</th>
                    <td>{grr.names.measurement}</td>
                    <td>{grr.names.part}</td>
                    <td>{grr.names.operator}</td>
                </tr>
                <tr>
                    <th scope='row'>Count</th>
                    <td>{grr.data.shape[0]}</td>
                    <td>{grr.counts[grr.names.part]}</td>
                    <td>{grr.counts[grr.names.operator]}</td>
                </tr>
                <thead><tr></tr></thead>
                <tr>
                    <th scope='row'>Replicates</th>
                    <td>{grr.replicates}</td>
                    <td colspan='2'></td>
                </tr>
                <tr>
                    <th scope='row'>Tolerance</th>
                    <td>{grr.tolerance}</td>
                    <td colspan='2'></td>
                </tr>
                <tr>
                    <th scope='row'>N<sub>&#x03C3;</sub></th>
                    <td>{grr.nsigma}</td>
                    <td colspan='2'></td>
                </tr>
                <thead><tr></tr></thead>
                <tr>
                    <th scope='row'>Formula</th>
                    <td colspan='3'>{grr.formula}</td>
                </tr>
            </tbody>
            </table>
            '''
        return html

class VarianceTable:
    """
    GRR variance components

    Variance components are calculated using the method in [1]_.

    Attributes
    ----------
    grr : :class:`GRR`
        GRR study object used to create the variance components.
    anova_table : pandas.DataFrame, automatic
        ANOVA summary table.
    table : pandas.DataFrame
        Table of variance components from GRR model.
    num_distinct_cat : float
        Number of confidence intervals on part measurements that will span the
        part variability (without overlap). For example, if `num_distinct_cat`
        is 5 then the measurement system can discern five groups of parts.
    discrimination : float
        Discrimination ratio, see [1]_.

    References
    ----------
    .. [1]  Montgomery, D. C. (2009).
            Statistical quality control (Vol. 7).
            New York: Wiley.
    """
    grr: NameMapping
    anova_table: pd.DataFrame
    table: pd.DataFrame
    num_distinct_cats: np.float64
    discrimination: np.float64

    def _table_index(self):
        return [
            'Gauge RR',
            'Repeatability',
            'Reproducibility',
            'Operator',
            'Operator*Part',
            'Part-to-Part',
            'Total']

    def _table_columns(self):
        return [
            'VarComp',
            '% Contribution',
            'StdDev',
            f'StudyVar ({self.grr.nsigma}*SD)',
            '% StudyVar',
            '% Tolerance']

    def _table_styles(self):
        if self.grr.include_interaction:
            reprod_indent = 'th.row3,th.row4'
        else:
            reprod_indent = 'th.row3'
        return [
            {
                'selector': '.row_heading',
                'props': [
                    ('text-align', 'left'),
                ]
            },
            {
                'selector': 'th.row1,th.row2',
                'props': [
                    ('padding-left', '1.5em'),
                ]
            },
            {
                'selector': reprod_indent,
                'props': [
                    ('padding-left', '3em'),
                ]
            }
        ]

    def __init__(self, grr: GRR, typ=2):
        self.grr = grr
        self.anova_table = mqr.anova.summary(
            grr.regression_result,
            typ,
            formatted=False)
        self._varcomp()
        self._calculate_table()
        self._set_discrimination()
        self._set_num_distinct_cats()

    def _varcomp(self):
        name_p = self.grr.names._p
        name_o = self.grr.names._o

        N_r = self.grr.replicates
        N_p = self.grr.counts[name_p]
        N_o = self.grr.counts[name_o]

        anova_table = self.anova_table
        MS_e = anova_table.iloc[-2, 2]
        MS_p = anova_table.iloc[0, 2]
        MS_o = anova_table.iloc[1, 2]
        if self.grr.include_interaction:
            MS_i = anova_table.iloc[2, 2]
        else:
            MS_i = 0

        var = MS_e
        var_i = np.clip((MS_i - MS_e) / N_r, 0, np.inf)
        var_p = (MS_p - MS_i) / (N_r * N_o)
        var_o = (MS_o - MS_i) / (N_r * N_p)

        self._variance_components = var_p, var_o, var_i, var

    def _calculate_table(self):
        var_p, var_o, var_i, var = self._variance_components

        table = pd.DataFrame(
            index=self._table_index(),
            columns=self._table_columns(),
            dtype=np.float64)
        table.iloc[:, 0] = [
            var_o + var_i + var,         # GRR
                var,                     # Repeatability
                var_o + var_i,           # Reproducibility
                    var_o,               # Operator
                    var_i,               # Interaction
            var_p,                       # Part-to-Part
            var_p + var_o + var_i + var, # Total
        ]
        table.iloc[:, 1] = 100 * table.iloc[:, 0] / table.iloc[-1, 0]
        table.iloc[:, 2] = np.sqrt(table.iloc[:, 0])
        table.iloc[:, 3] = self.grr.nsigma * table.iloc[:, 2]
        table.iloc[:, 4] = 100 * table.iloc[:, 3] / table.iloc[-1, 3]
        table.iloc[:, 5] = 100 * table.iloc[:, 3] / self.grr.tolerance

        if not self.grr.include_interaction:
            table.drop(index=['Operator*Part'], inplace=True)

        self.table = table

    def _set_discrimination(self):
        rho_p = self.table.loc['Part-to-Part', '% Contribution'] / 100
        self.discrimination = (1 + rho_p) / (1 - rho_p)

    def _set_num_distinct_cats(self):
        rho_p = self.table.loc['Part-to-Part', '% Contribution'] / 100
        self.num_distinct_cats = np.sqrt(2 * rho_p / (1 - rho_p))

    def _repr_html_(self):
        n_cats = int(np.floor(self.num_distinct_cats))
        html = '<div style="display:flex; flex-direction:column; align-items:flex-start;">'
        html += (
            self.table.style
            .format(formatter='{:.4g}')
            .set_table_styles(self._table_styles())
            ._repr_html_())
        html += f'<div><b>Number of distinct categories:</b> {n_cats:d}</div>'
        html += '</div>'
        return html

# class ConfTable:
#     def _make_conf_int_table(self):
#         p, o, n = self.counts
#         [MS_p, MS_o, MS_i, MS_e] = self._mean_squares
#         var_grr = self.grr_table.loc['Gauge RR', 'VarComp']
#         var_p = self.grr_table.loc['Part-to-Part', 'VarComp']
#         var_tot = self.grr_table.loc['Total', 'VarComp']
#         intervals = _conf_int(p, o, n, MS_p, MS_o, MS_i, MS_e, self.alpha)
#         [
#             (var_p_lower, var_p_upper),
#             (var_grr_lower, var_grr_upper),
#             (var_tot_lower, var_tot_upper),
#             (rho_p_lower, rho_p_upper),
#             (rho_m_lower, rho_m_upper),
#         ] = intervals

#         values = np.array([
#             [var_grr, var_grr + var_grr_lower, var_grr + var_grr_upper],
#             [var_p, var_p + var_p_lower, var_p + var_p_upper],
#             [var_tot, var_tot + var_tot_lower, var_tot + var_tot_upper],
#             [np.nan, rho_m_lower*100, rho_m_upper*100],
#             [np.nan, rho_p_lower*100, rho_p_upper*100],
#         ])
#         table = pd.DataFrame(
#             values,
#             index=GRR._conf_index(),
#             columns=GRR._conf_columns(self.alpha))
#         self.conf_int_table = table

    # @staticmethod
    # def _conf_index():
    #     return [
    #         'Gauge RR',
    #         'Part-to-Part',
    #         'Total',
    #         'Gauge RR (%)',
    #         'Part-to-Part (%)',
    #     ]

    # @staticmethod
    # def _conf_columns(alpha):
    #     return [
    #         f'E(var)',
    #         f'[{100*alpha/2:.3g}%',
    #         f'{100*(1-alpha/2):.3g}%]'
    #     ]


# def _conf_int(p, o, n, MS_P, MS_O, MS_PO, MS_E, alpha=0.05):
#     # Burdick, Richard K., Connie M. Borror, and Douglas C. Montgomery. "A review of methods for measurement systems capability analysis." Journal of Quality Technology 35.4 (2003): 342-354.

#     V_LP, V_UP, V_LM, V_UM, V_LT, V_UT, L_star, U_star = _conf_int_intermediate_vals(p, o, n, MS_P, MS_O, MS_PO, MS_E, alpha)
    
#     var_p_lower = -np.sqrt(V_LP) / (o * n)
#     var_p_upper = np.sqrt(V_UP) / (o * n)

#     var_grr_lower = -np.sqrt(V_LM) / (p * n)
#     var_grr_upper = np.sqrt(V_UM) / (p * n)

#     var_tot_lower = -np.sqrt(V_LT) / (p * o * n)
#     var_tot_upper = np.sqrt(V_UT) / (p * o * n)

#     rho_p_lower = L_p = p * L_star / (p * L_star + o)
#     rho_p_upper = U_p = p * U_star / (p * U_star + o)

#     rho_m_lower = 1 - U_p
#     rho_m_upper = 1 - L_p

#     return [
#         (var_p_lower, var_p_upper),
#         (var_grr_lower, var_grr_upper),
#         (var_tot_lower, var_tot_upper),
#         (rho_p_lower, rho_p_upper),
#         (rho_m_lower, rho_m_upper),
#     ]

# def _conf_int_intermediate_vals(p, o, n, MS_P, MS_O, MS_PO, MS_E, alpha=0.05):
#     # Burdick, Richard K., Connie M. Borror, and Douglas C. Montgomery. "A review of methods for measurement systems capability analysis." Journal of Quality Technology 35.4 (2003): 342-354.
#     # But doesn't match "Introduction to Statistical Quality Control", Montgomery... need to look a bit closer at this.

#     # Distributions
#     F_p = st.chi2(p-1)
#     F_o = st.chi2(o-1)
#     F_i = st.chi2((p-1)*(o-1))
#     F_e = st.chi2(p*o*(n-1))
#     F_po = st.f(p-1, o-1)
#     F_pi = st.f(p-1, (p-1)*(o-1))

#     # Intermediate quantities
#     G_1 = 1 - 1 / F_p.ppf(1-alpha/2)
#     G_2 = 1 - 1 / F_o.ppf(1-alpha/2)
#     G_3 = 1 - 1 / F_i.ppf(1-alpha/2)
#     G_4 = 1 - 1 / F_e.ppf(1-alpha/2)

#     H_1 = 1 / F_p.ppf(alpha/2) - 1
#     H_2 = 1 / F_o.ppf(alpha/2) - 1
#     H_3 = 1 / F_i.ppf(alpha/2) - 1
#     H_4 = 1 / F_e.ppf(alpha/2) - 1

#     G_13 = ((F_pi.ppf(1-alpha/2)-1)**2 - G_1**2 * F_pi.ppf(1-alpha/2)**2 - H_3**2) / F_pi.ppf(1-alpha/2)
#     H_13 = ((1-F_pi.ppf(alpha/2))**2 - H_1**2 * F_pi.ppf(alpha/2)**2 - G_3**2) / F_pi.ppf(alpha/2)

#     V_LP = G_1**2 * MS_P**2 + H_3**2 * MS_PO**2 + G_13 * MS_P
#     V_UP = H_1**2 * MS_P**2 + G_3**2 * MS_PO**2 + H_13 * MS_P**2 * MS_PO

#     V_LM = G_2**2 * MS_O**2 + G_3**2 * (p-1)**2 * MS_PO**2 + G_4**2 * p**2 * (n-1)**2 * MS_E**2
#     V_UM = H_2**2 * MS_O**2 + H_3**2 * (p-1)**2 * MS_PO**2 + H_4**2 * p**2 * (n-1)**2 * MS_E**2

#     V_LT = G_1**2 * p**2 * MS_P**2 + G_2**2 * o**2 * MS_O**2 + G_3**2 * (p*o - p - o)**2 * MS_PO**2 + G_4**2 * (p*o)**2 * (n-1)**2 * MS_E**2
#     V_UT = H_1**2 * p**2 * MS_P**2 + H_2**2 * o**2 * MS_O**2 + H_3**2 * (p*o - p - o)**2 * MS_PO**2 + H_4**2 * (p*o)**2 * (n-1)**2 * MS_E**2

#     L_star_num = MS_P - F_pi.ppf(1-alpha/2) * MS_PO
#     L_star_den = p * (n-1) * F_p.ppf(1-alpha/2) * MS_E + F_po.ppf(1-alpha/2) * MS_O + (p-1) * F_p.ppf(1-alpha/2) * MS_PO
#     L_star = L_star_num / L_star_den

#     U_star_num = MS_P - F_pi.ppf(alpha/2) * MS_PO
#     U_star_den = p * (n-1) * F_p.ppf(alpha/2) * MS_E + F_po.ppf(alpha/2) * MS_O + (p-1) * F_p.ppf(alpha/2) * MS_PO
#     U_star = U_star_num / U_star_den

#     return V_LP, V_UP, V_LM, V_UM, V_LT, V_UT, L_star, U_star
