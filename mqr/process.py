"""
===================================================
Process summary and capability (:mod:`mqr.process`)
===================================================

.. currentmodule:: mqr.process

User guide
    :doc:`/user_guide/summary-capability`

Detailed examples
    https://github.com/nklsxn/mqrpy-guide

Routines for summarising processes and their capability.

See `doc:mqr.process` for more details.

.. rubric:: Classes

.. autosummary::
    :toctree: generated/

    Sample
    Specification
    Capability
    Summary
"""

from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import scipy
import scipy.stats as st
import seaborn as sns
from statsmodels.stats.diagnostic import normal_ad, kstest_normal

import mqr

@dataclass
class Sample:
    """
    Data and descriptive statistics for a single sample from a process.

    Construct using a pandas Series (ie. a column from a dataframe). Intended
    for use by a `Study` object.

    Attributes
    ----------
    name : str
        Name of the KPI or measurement.
    conf : float
        Confidence level to use in confidence intervals.
    data : pd.Series
        Sample measurements.

    ad_stat : float
        Anderson-Darling normality test statistic.
    ad_pvalue : float
        p-value associated with `ad_stat`.
    ks_stat : float
        Kolmogorov-Smirnov goodness of fit (with normal) test statistic.
    ks_pvalue : float
        p-value associated with `ks_stat`.

    nobs : int
        Number of measurements in the sample.
    mean : float
        Sample mean.
    sem : float
        Standard error of mean.
    std : float
        Sample standard deviation.
    var : float
        Sample variance.
    skewness : float
        Skewness.
    kurtosis : float
        Kurtosis.
    minimum : float
        Smallest observation.
    quartile1 : float
        25th percentile observation.
    median : float
        Median observation.
    quartile3 : float
        75th percentile obsevation.
    maximum : float
         Largest observation.
    iqr : float
        Inter-quartile range.

    conf_mean : ConfidenceInterval
        Conf interval on the mean.
    conf_var : ConfidenceInterval
        Conf interval on the variance.
    conf_quartile1 : ConfidenceInterval
        Conf interval on the 25th percentile.
    conf_median : ConfidenceInterval
        Conf interval on the median.
    conf_quartile3 : ConfidenceInterval
        Conf interval on the 75th percentile.
    outliers : array_like
        List of points falling further from a quartile than `1.5 * iqr`.

    Examples
    --------
    In a jupyter notebook, sample summaries are shown as HTML tables:

    >>> data = pd.read_csv(mqr.sample_data('study-random-5x5.csv'))
    >>> mqr.process.Sample(data['KPI1'])

    produces

    +--------------------------------+
    | KPI1                           |
    +================================+
    | Normality (Anderson-Darling).  |
    +--------------+-----------------+
    | Stat         | 0.34261         |
    +--------------+-----------------+
    | P-value      | 0.48588         |
    +--------------+-----------------+
    |                                |
    +--------------+-----------------+
    | N            | 120             |
    +--------------+-----------------+
    |                                |
    +--------------+-----------------+
    | Mean         | 149.97          |
    +--------------+-----------------+
    | StdDev       | 1.1734          |
    +--------------+-----------------+
    | Variance     | 1.3768          |
    +--------------+-----------------+
    | Skewness     | 0.23653         |
    +--------------+-----------------+
    |                                |
    +--------------+-----------------+
    | Kurtosis     | 0.34012         |
    +--------------+-----------------+
    | Minimum      | 147.03          |
    +--------------+-----------------+
    | 1st Quartile | 149.22          |
    +--------------+-----------------+
    | Median       | 149.97          |
    +--------------+-----------------+
    | 3rd Quartile | 150.56          |
    +--------------+-----------------+
    | Maximum      | 153.27          |
    +--------------+-----------------+
    |                                |
    +--------------+-----------------+
    | N Outliers.  | 5               |
    +--------------+-----------------+

    """
    name: str = None
    conf: float = field(default=np.nan, repr=False)
    data: pd.Series = field(default=None, repr=False)

    ad_stat: float = field(default=np.nan, repr=False)
    ad_pvalue: float = field(default=np.nan, repr=False)
    ks_stat: float = field(default=np.nan, repr=False)
    ks_pvalue: float = field(default=np.nan, repr=False)

    nobs: int = 0
    mean: np.float64 = np.nan
    sem: np.float64 = field(default=np.nan, repr=False)
    std: np.float64 = field(default=np.nan, repr=False)
    var: np.float64 = np.nan
    skewness: np.float64 = np.nan
    kurtosis: np.float64 = np.nan
    minimum: np.float64 = np.nan
    quartile1: np.float64 = np.nan
    median: np.float64 = np.nan
    quartile3: np.float64 = np.nan
    maximum: np.float64 = np.nan
    iqr: np.float64 = field(default=np.nan, repr=False)

    conf_mean: mqr.inference.confint.ConfidenceInterval = field(default=None, repr=False)
    conf_std: mqr.inference.confint.ConfidenceInterval = field(default=None, repr=False)
    conf_var: mqr.inference.confint.ConfidenceInterval = field(default=None, repr=False)
    conf_quartile1: mqr.inference.confint.ConfidenceInterval = field(default=None, repr=False)
    conf_median: mqr.inference.confint.ConfidenceInterval = field(default=None, repr=False)
    conf_quartile3: mqr.inference.confint.ConfidenceInterval = field(default=None, repr=False)

    outliers: np.ndarray = field(default=None, repr=False)

    def __init__(self, data, conf=0.95, ddof=1, name=None, num_display_fmt='#.5g'):
        import scipy.stats as st

        if hasattr(data, 'name'):
            self.name = data.name
        elif name is not None:
            self.name = name
        else:
            self.name = 'data'
        self.conf = conf
        self.data = data

        (self.ad_stat, self.ad_pvalue) = normal_ad(data)
        (self.ks_stat, self.ks_pvalue) = kstest_normal(data)

        self.nobs = len(data)
        self.mean = np.mean(data)
        self.sem = st.sem(data)
        self.std = np.std(data, ddof=ddof)
        self.var = np.var(data, ddof=ddof)
        self.skewness = st.skew(data)
        self.kurtosis = st.kurtosis(data)
        self.minimum = np.min(data)
        self.quartile1 = np.quantile(data, 0.25)
        self.median = np.median(data)
        self.quartile3 = np.quantile(data, 0.75)
        self.maximum = np.max(data)
        self.iqr = self.quartile3 - self.quartile1

        self.conf_mean = mqr.inference.mean.confint_1sample(data, conf=conf)
        self.conf_std = mqr.inference.stddev.confint_1sample(data, conf=conf)
        self.conf_var = mqr.inference.variance.confint_1sample(data, conf=conf)

        self.conf_quartile1 = mqr.inference.nonparametric.quantile.confint_1sample(data, q=0.25, conf=conf)
        self.conf_median = mqr.inference.nonparametric.quantile.confint_1sample(data, q=0.5, conf=conf)
        self.conf_quartile3 = mqr.inference.nonparametric.quantile.confint_1sample(data, q=0.75, conf=conf)

        self.outliers = np.concatenate([
            data[data<self.quartile1-1.5*self.iqr],
            data[data>self.quartile3+1.5*self.iqr]])

@dataclass
class Specification:
    """
    Process specification.

    Attributes
    ----------
    target : float
        Design value for the process.
    lsl : float
        Lower specification limit.
    usl : float
        Upper specification limit.
    """
    target: float
    lsl: float
    usl: float

@dataclass
class Capability:
    """
    Process capability values.

    Attributes
    ----------
    sample : Sample
        Sample that has this capability.
    spec : Specification
        Specification for which `sample` has this capability.
    cp : float
        Process potential. The capability of the process if it was centred at
        `Specification.target`.
    cpk : float
        Process capability. The number of standard deviations of process variation
        that fit in the specification, normalised by 3*sigma. Ie. a 6-sigma
        process has capability 2.0.
    defects_st : float
        Short-term defect rate, based on a fitted normal distribution.
    defects_lt : float
        Long-term defect rate, based on a normal distribution with 1.5*stddev
        larger than short-term.

    Examples
    --------
    Construct this object with a sample and a `mqr.process.Specification`:

    .. code-block:: python
        :emphasize-lines: 3

        data = pd.read_csv(mqr.sample_data('study-random-5x5.csv'))
        summary = mqr.process.Summary(data['KPI1'])
        spec = mqr.process.Specification(150, 147, 153)
        mqr.process.Capability(summary['KPI1'], spec)

    In a jupyter notebook, this produces the HTML table below. Iterables of
    `Capability` are shown as the same table with multiple columns.

    +-----------------+----------+
    |                 | KPI1     |
    +=================+==========+
    | USL             | 153.     |
    +-----------------+----------+
    | Target          | 150.     |
    +-----------------+----------+
    | LSL             | 147.     |
    +-----------------+----------+
    |                            |
    +-----------------+----------+
    | Cpk             | 0.844    |
    +-----------------+----------+
    | Cp              | 0.852    |
    +-----------------+----------+
    | Defectsst (ppm) | 1.06e+04 |
    +-----------------+----------+
    | Defectslt (ppm) | 8.83e+04 |
    +-----------------+----------+

    """
    sample: Sample
    spec: Specification

    cp: float
    cpk: float
    defects_st: float
    defects_lt: float

    def __init__(self, sample: Sample, spec: Specification):
        """
        Construct Capability.

        Attributes
        ----------
        sample : mqr.process.Sample
            Set of measurements from KPI.
        spec : mqr.process.Specification
            Specificatino for KPI.

        """
        self.sample = sample
        self.spec = spec

        self.cp = (spec.usl - spec.lsl) / (6 * sample.std)
        self.cpk = np.minimum(spec.usl - sample.mean, sample.mean - spec.lsl) / (3 * sample.std)
        in_spec = np.logical_and(sample.data >= spec.lsl, sample.data <= spec.usl)
        dist = st.norm(sample.mean, sample.std)
        dist_lt = st.norm(sample.mean, 1.5 * sample.std)
        self.defects_st = 1 - (dist.cdf(spec.usl) - dist.cdf(spec.lsl))
        self.defects_lt = 1 - (dist_lt.cdf(spec.usl + 1.5 * sample.std) -
                               dist_lt.cdf(spec.lsl - 1.5 * sample.std))

@dataclass
class Summary:
    """
    Measurements and summary statistics for a set of samples from a process.

    Attributes
    ----------
    data : pd.DataFrame
        Measurements with KPIs in each column, and possibly other columns like
        run lables, operator IDs, etc.
    samples : dict[str, mqr.process.Sample]
        Automatically constructed. Dict of `mqr.process.Sample` for each sample
        in data.
    capabilities : dict[str, Capability]
        Automatically constructed when initialised with `Specification`s. Dict
        of `mqr.process.Capability` for each sample in data.

    Examples
    --------
    Construct this object using a dataframe of measurements, optionally providing
    a list of columns to include:

    >>> data = pd.read_csv(mqr.sample_data('study-random-5x5.csv'))
    >>> mqr.process.Study(data)

    That input is shown in notebooks as an HTML table:

    +--------------+---------+-----------+-----------+----------+----------+
    |              | KPI1    | KPI2      | KPI3      | KPO1     | KPO2     |
    +==============+=========+===========+===========+==========+==========+
    | Normality (Anderson-Darling)                                         |
    +--------------+---------+-----------+-----------+----------+----------+
    | Stat         | 0.34261 | 0.23796   | 1.1874    | 0.19203  | 0.70213  |
    +--------------+---------+-----------+-----------+----------+----------+
    | P-value      | 0.48588 | 0.77835   | 0.0040775 | 0.89417  | 0.065144 |
    +--------------+---------+-----------+-----------+----------+----------+
    +--------------+---------+-----------+-----------+----------+----------+
    | N            | 120     | 120       | 120       | 120      | 120      |
    +--------------+---------+-----------+-----------+----------+----------+
    +--------------+---------+-----------+-----------+----------+----------+
    | Mean         | 149.97  | 20.003    | 14.004    | 160.05   | 4.0189   |
    +--------------+---------+-----------+-----------+----------+----------+
    | StdDev       | 1.1734  | 0.24527   | 0.75643   | 2.0489   | 1.5634   |
    +--------------+---------+-----------+-----------+----------+----------+
    | Variance     | 1.3768  | 0.060156  | 0.57219   | 4.1979   | 2.4443   |
    +--------------+---------+-----------+-----------+----------+----------+
    | Skewness     | 0.23653 | -0.31780  | -0.63437  | -0.12064 | 0.087295 |
    +--------------+---------+-----------+-----------+----------+----------+
    | Kurtosis     | 0.34012 | -0.032159 | 0.37947   | -0.16908 | -0.18817 |
    +--------------+---------+-----------+-----------+----------+----------+
    +--------------+---------+-----------+-----------+----------+----------+
    | Minimum      | 147.03  | 19.234    | 11.639    | 154.89   | -0.37247 |
    +--------------+---------+-----------+-----------+----------+----------+
    | 1st Quartile | 149.22  | 19.833    | 13.642    | 158.87   | 2.9019   |
    +--------------+---------+-----------+-----------+----------+----------+
    | Median       | 149.97  | 20.012    | 14.033    | 160.02   | 3.9264   |
    +--------------+---------+-----------+-----------+----------+----------+
    | 3rd Quartile | 150.56  | 20.173    | 14.481    | 161.35   | 5.2160   |
    +--------------+---------+-----------+-----------+----------+----------+
    | Maximum      | 153.27  | 20.505    | 15.460    | 164.51   | 8.2828   |
    +--------------+---------+-----------+-----------+----------+----------+
    +--------------+---------+-----------+-----------+----------+----------+
    | N Outliers   | 5       | 1         | 4         | 1        | 0        |
    +--------------+---------+-----------+-----------+----------+----------+

    """
    data: pd.DataFrame = field(repr=False)
    samples: dict[str, Sample] = field(repr=False)
    capabilities: dict[str, Capability] = field(repr=False)

    def __init__(self, data, specs=None, conf=0.95, ddof=1):
        if isinstance(data, pd.Series):
            self.data = data.to_frame()
        elif isinstance(data, pd.DataFrame):
            self.data = data
        else:
            raise ValueError('`data` must be a Series or a DataFrame.')

        self.samples = {
            name: Sample(col, conf=conf, ddof=ddof)
            for name, col
            in self.data.items()
        }

        if specs is not None:
            self.capabilities = {
                name: Capability(self.samples[name], spec)
                for name, spec
                in specs.items()
            }
        else:
            self.capabilities = {}

    def __getitem__(self, index):
        return self.samples[index]
