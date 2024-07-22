"""
Summary statistics for samples of data.
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

    Arguments
    ---------
    name (str) -- Name of the KPI or measurement.
    conf (float) -- Confidence level to use in confidence intervals.
    data (pd.Series) -- Sample measurements.

    ad_stat(float) -- Anderson-Darling normality test statistic.
    ad_pvalue (float) -- p-value associated with `ad_stat`.
    ks_stat (float) -- Kolmogorov-Smirnov goodness of fit (with normal) test statistic.
    ks_pvalue (float) -- p-value associated with `ks_stat`.
    nobs (int) -- Number of measurements in the sample.

    mean (float) -- Sample mean.
    sem (float) -- Standard error of mean.
    std (float) -- Sample standard deviation.
    var (float) -- Sample variance.
    skewness (float) -- Skewness.
    kurtosis (float) -- Kurtosis.
    minimum (float) -- Smallest observation.
    quartile1 (float) -- 25th percentile observation.
    median (float) -- Median observation.
    quartile3 (float) -- 75th percentile obsevation.
    maximum (float) --  Largest observation.
    iqr (float) -- Inter-quartile range.

    conf_mean (ConfidenceInterval) -- Conf interval on the mean.
    conf_var (ConfidenceInterval) -- Conf interval on the variance.
    conf_quartile1 (ConfidenceInterval) -- Conf interval on the 25th percentile.
    conf_median (ConfidenceInterval) -- Conf interval on the median.
    conf_quartile3 (ConfidenceInterval) -- Conf interval on the 75th percentile.
    outliers (np.ndarray) -- List of points falling further from a quartile than `1.5 * iqr`.
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
    # conf_std: mqr.confint.ConfidenceInterval = field(default=None, repr=False)
    conf_var: mqr.inference.confint.ConfidenceInterval = field(default=None, repr=False)
    conf_quartile1: mqr.inference.confint.ConfidenceInterval = field(default=None, repr=False)
    conf_median: mqr.inference.confint.ConfidenceInterval = field(default=None, repr=False)
    conf_quartile3: mqr.inference.confint.ConfidenceInterval = field(default=None, repr=False)

    outliers: np.ndarray = field(default=None, repr=False)

    def __init__(self, data, conf=0.95, ddof=1, name=None):
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
        # self.conf_std = mqr.inference.std.confint_1sample(data, conf=conf)
        self.conf_var = mqr.inference.variance.confint_1sample(data, conf=conf)

        self.conf_quartile1 = mqr.inference.nonparametric.quantile.confint_1sample(data, q=0.25, conf=conf)
        self.conf_median = mqr.inference.nonparametric.quantile.confint_1sample(data, q=0.5, conf=conf)
        self.conf_quartile3 = mqr.inference.nonparametric.quantile.confint_1sample(data, q=0.75, conf=conf)

        self.outliers = np.concatenate([
            data[data<self.quartile1-1.5*self.iqr],
            data[data>self.quartile3+1.5*self.iqr]])

    # def _repr_html_(self):
        # return html(self)

@dataclass
class Study:
    """
    Measurments and summary statistics for a set of samples from a process.

    Construct this object using a dataframe of measurements, optionally providing
    a list of columns to exclude from statistics (ie. columns that are not
    measurements):
    >>> df = pd.read_csv('process_measurements.csv')
    >>> Study(df, ['RUNID', 'OPERATOR', 'BLOCK'])

    Fields
    ------
    name (str) -- The name of the process or experiment.
    data (pd.DataFrame) -- Measurements with KPIs in each column, and possibly
        other columns like run lables, operator IDs, etc.
    exclude (list[str]) -- A list of column names to exclude from descriptive stats.
    samples (dict[str, mqr.process.Sample]) -- Automatically constructed. Dict
        of `mqr.process.Sample` for each KPI in the dataframe.
    conf (float) -- Confidence level to use for confidence intervals.
    num_display_fmt (int) -- The format specifier to use when displaying data as text.
    """
    name: str
    data: pd.DataFrame = field(repr=False)
    measurements: list[str] = field(repr=False)
    samples: dict[str, Sample] = field(repr=False)
    conf: float = field(repr=False)

    num_display_fmt: str = field(repr=False)

    def __init__(self, data:pd.DataFrame, measurements:list[str]=None, conf=0.95, ddof=1, num_display_fmt='#.5g'):
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
        if exclude_inputs:
            return self.data[self.measurements]
        else:
            return self.data

    def __getitem__(self, index):
        return self.samples[index]

    def _repr_html_(self):
        return html(self, self.num_display_fmt)

def html(d:Study, display_fmt):
    conf = d.conf
    fmt = display_fmt

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

    def fmt_conf(value, interval):
        return (
            f'{value:{display_fmt}} '
            f'<font color="gray">({interval[0]:{display_fmt}}, '
            f'{interval[1]:{display_fmt}}) '
            f'<sup>{conf*100:.0f}%</sup></font>')
        # return f'{value:.{prec}g} [{values[0]:.{prec}g}, {values[1]:.{prec}g}]'


          # <th scope="row">{np.floor(d.conf_quartile1[1]*100):.0f}% CI</th>
          # <td>{d.conf_quartile1[0][0]:.{prec}g}</td>
          # <td>{d.conf_quartile1[0][1]:.{prec}g}</td>


    caption = 'Study' if d.name is None else f'Study - {d.name}'
    
    col_headers = [name for name in d.samples.keys()]

    ad_stat = [a.ad_stat for a in d.samples.values()]
    ad_pvalue = [a.ad_pvalue for a in d.samples.values()]
    ks_stat = [a.ks_stat for a in d.samples.values()]
    ks_pvalue = [a.ks_pvalue for a in d.samples.values()]

    nobs = [a.nobs for a in d.samples.values()]
    mean = [a.mean for a in d.samples.values()]
    std = [a.std for a in d.samples.values()]
    var = [a.var for a in d.samples.values()]
    skewness = [a.skewness for a in d.samples.values()]
    kurtosis = [a.kurtosis for a in d.samples.values()]
    minimum = [a.minimum for a in d.samples.values()]
    quartile1 = [a.quartile1 for a in d.samples.values()]
    median = [a.median for a in d.samples.values()]
    quartile3 = [a.quartile3 for a in d.samples.values()]
    maximum = [a.maximum for a in d.samples.values()]
    
    # conf_level: float = np.nan
    conf_mean = [a.conf_mean for a in d.samples.values()]
    # conf_std: tuple = None
    # conf_var: tuple = None
    # conf_quartile1: tuple = None
    # conf_median: tuple = None
    # conf_quartile3: tuple = None

    outliers = [len(a.outliers) for a in d.samples.values()]

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
