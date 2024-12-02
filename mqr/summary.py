"""
=======================================
Summary Statistics (:mod:`mqr.summary`)
=======================================

.. currentmodule:: mqr.summary

.. rubric:: Classes

.. autosummary::
    :toctree: generated/

    Sample
    Study

"""

import numpy as np
import pandas as pd
import scipy.stats as st

from statsmodels.stats.diagnostic import normal_ad, kstest_normal

import mqr

from dataclasses import dataclass, field

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
    >>> mqr.summary.Sample(data['KPI1'])

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

    num_display_fmt: str = field(default='#.5g', repr=False)

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

        self.num_display_fmt = num_display_fmt

    def _repr_html_(self):
        return html({self.name: self}, self.num_display_fmt)

@dataclass
class Study:
    """
    Measurements and summary statistics for a set of samples from a process.

    Attributes
    ----------
    name : str
        The name of the process or experiment.
    data : pd.DataFrame
        Measurements with KPIs in each column, and possibly other columns like
        run lables, operator IDs, etc.
    measurements : list[str]
        A list of column names to include for descriptive stats.
    samples : dict[str, mqr.process.Sample]
        Automatically constructed. Dict of `mqr.process.Sample` for each KPI in
        the dataframe.
    conf : float
        Confidence level to use for confidence intervals.
    num_display_fmt : int
        The format specifier to use when displaying data as text.

    Examples
    --------
    Construct this object using a dataframe of measurements, optionally providing
    a list of columns to include:

    >>> data = pd.read_csv(mqr.sample_data('study-random-5x5.csv'))
    >>> mqr.summary.Study(
    >>>     data=data,
    >>>     measurements=['KPI1', 'KPI2', 'KPI3', 'KPO1', 'KPO2'],
    >>>     conf=0.98)

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
    name: str
    data: pd.DataFrame = field(repr=False)
    measurements: list[str] = field(repr=False)
    samples: dict[str, Sample] = field(repr=False)
    conf: float = field(repr=False)

    num_display_fmt: str = field(repr=False)

    def __init__(self, data:pd.DataFrame, measurements:list[str]=None, conf=0.95, ddof=1, num_display_fmt='#.5g'):
        if not isinstance(data, pd.DataFrame):
            raise ValueError('`data` must be a DataFrame.')

        try:
            self.name = data.name
        except:
            self.name = 'Dataset'

        self.data = data
        if measurements is not None:
            self.measurements = measurements
        else:
            self.measurements = data.columns
        self.samples = {
            name: Sample(data[name], conf=conf, ddof=ddof)
            for name in self.measurements}
        self.conf = conf

        self.num_display_fmt = num_display_fmt

    def get_data(self, exclude_inputs=True):
        """
        Get source Dataframe, optionally showing only measurement columns.

        Parameters
        ----------
        exclude_inputs : bool, optional
            When `True`, only show columns listed in `measurements`.
        """
        if exclude_inputs:
            return self.data[self.measurements]
        else:
            return self.data

    def __getitem__(self, index):
        return self.samples[index]

    def _repr_html_(self):
        caption = f'Study - {self.name}'
        return html(self.samples, self.num_display_fmt)

def html(samples, display_fmt, caption=None):
    def join(s):
        return ''.join(s)

    def th(scope='col'):
        def _th(s):
            return f'<th scope="{scope}">{s}</th>'
        return _th

    def td(s):
        return f'<td>{s}</td>'

    def fmt_g(value):
        return f'{value:{display_fmt}}'

    caption = '' if caption is None else caption
    
    col_headers = [n for n in samples.keys()]

    ad_stat = [a.ad_stat for a in samples.values()]
    ad_pvalue = [a.ad_pvalue for a in samples.values()]
    ks_stat = [a.ks_stat for a in samples.values()]
    ks_pvalue = [a.ks_pvalue for a in samples.values()]

    nobs = [a.nobs for a in samples.values()]
    mean = [a.mean for a in samples.values()]
    std = [a.std for a in samples.values()]
    var = [a.var for a in samples.values()]
    skewness = [a.skewness for a in samples.values()]
    kurtosis = [a.kurtosis for a in samples.values()]
    minimum = [a.minimum for a in samples.values()]
    quartile1 = [a.quartile1 for a in samples.values()]
    median = [a.median for a in samples.values()]
    quartile3 = [a.quartile3 for a in samples.values()]
    maximum = [a.maximum for a in samples.values()]

    outliers = [len(a.outliers) for a in samples.values()]

    html = f'''
    <table>
        <caption>{caption}</caption>
        <thead>
            <tr>
                <th scope="col"></th>
                {join(map(th(), col_headers))}
            </tr>
        </thead>
        <tbody>
            <tr>
                <th colspan={len(col_headers)+1} style="text-align:left;">Normality (Anderson-Darling)</th>
            </tr>
            <tr>
                <th scope="row">Stat</th>
                {join(map(td, map(fmt_g, ad_stat)))}
            </tr>
            <tr>
                <th scope="row">P-value</th>
                {join(map(td, map(fmt_g, ad_pvalue)))}
            </tr>

            <thead><tr></tr></thead>
            <tr>
                <th scope="row">N</th>
                {join(map(td, nobs))}
            </tr>

            <thead><tr></tr></thead>
            <tr>
                <th scope="row">Mean</th>
                {join(map(td, map(fmt_g, mean)))}
            </tr>
            <tr>
                <th scope="row">StdDev</th>
                {join(map(td, map(fmt_g, std)))}
            </tr>
            <tr>
                <th scope="row">Variance</th>
                {join(map(td, map(fmt_g, var)))}
            </tr>
            <tr>
                <th scope="row">Skewness</th>
                {join(map(td, map(fmt_g, skewness)))}
            </tr>
            <tr>
                <th scope="row">Kurtosis</th>
                {join(map(td, map(fmt_g, kurtosis)))}
            </tr>

            <thead><tr></tr></thead>
            <tr>
                <th scope="row">Minimum</th>
                {join(map(td, map(fmt_g, minimum)))}
            </tr>
            <tr>
                <th scope="row">1st Quartile</th>
                {join(map(td, map(fmt_g, quartile1)))}
            </tr>
            <tr>
                <th scope="row">Median</th>
                {join(map(td, map(fmt_g, median)))}
            </tr>
            <tr>
                <th scope="row">3rd Quartile</th>
                {join(map(td, map(fmt_g, quartile3)))}
            </tr>
            <tr>
                <th scope="row">Maximum</th>
                {join(map(td, map(fmt_g, maximum)))}
            </tr>

            <thead><tr></tr></thead>
            <tr>
                <th scope="row">N Outliers</th>
                {join(map(td, outliers))}
            </tr>
        </tbody>
        <tfoot>
        </tfoot>
    </table>
    '''
    return html
