"""
SPC Types

.. :currentmodule:`mqr.spc`

This module has types for creating control parameters and statistics.
"""

import abc
from dataclasses import asdict, dataclass, field
import matplotlib.pyplot as plt
from mqr.spc.util import c4, d2, d3
import numpy as np
import pandas as pd
import scipy

@dataclass
class ControlStatistic:
    """
    Monitored statistic of a controlled process.

    Parameters
    ----------
    stat: pandas.Series
        Monitored statistic, indexed by sample.
    nobs: pandas.Series
        Number of observations used to construct the corresponding sample.
    """
    stat: pd.Series = field(repr=False)
    nobs: pd.Series = field(repr=False)

    def __post_init__(self):
        if len(self.stat) != len(self.nobs):
            raise ValueError('Series stat and nobs must be the same length.')

@dataclass
class ControlParams:
    """
    Base class of all control methods.

    Other types should implement at least these functions to be monitored with
    SPC rules.
    """

    @abc.abstractmethod
    def statistic(self, samples):
        """
        Calculates the control statistic from samples.

        Parameters
        ----------
        samples : pandas.DataFrame
            Observations to be controlled. The index is a sample number, which
            some methods (eg. EWMA) depend on. The columns are dependent on the
            monitoring method, for example, they could be the observations in a
            fixed-size sample.

        Returns
        -------
        :class:`ControlStatistic`
            Statistics and sample sizes for this method, based on samples. Has
            the same index as the argument `samples`.
        """
        pass

    @abc.abstractmethod
    def target(self):
        """
        Process target value for this monitoring method.

        Returns
        -------
        float
            Target value.
        """
        pass

    @abc.abstractmethod
    def lcl(self, nobs):
        """
        Calculates the lower control limits for samples with sizes in `nobs`.

        Parameters
        ----------
        nobs : pandas.Series
            Sample sizes used to construct the statistic that will be monitored
            with this limit. The index must match the index of the `samples`
            argument passed to `ControlParams.statistic`.

        Returns
        -------
        pandas.Series
            Lower limits for each sample. Has the same index as the argument `nobs`.
        """
        pass

    @abc.abstractmethod
    def ucl(self, nobs):
        """
        Calculates the upper control limits for samples with sizes in `nobs`.

        Parameters
        ----------
        nobs : pandas.Series
            Sample sizes used to construct the statistic that will be monitored
            with this limit. The index must match the index of the `samples`
            argument passed to `ControlParams.statistic`.

        Returns
        -------
        pandas.Series
            Upper limits for each sample. Has the same index as the argument `nobs`.
        """
        pass

    def asdict(self):
        """
        Returns a dict representation of this class.

        This is a convenience method for serialisation. It allows users to
        store control parameters that are based on historical data, ready for
        later use. The resulting dictionary can be passed to the class's
        constructor to reproduce an identical object.
        """
        return asdict(self)

@dataclass
class ShewhartParams(ControlParams):
    """
    Base class of Shewhart-type control methods.

    Other types should implement at least these functions to be monitored with
    SPC rules expressed in terms of "sigma"s, or standard deviations of the
    sampling distribution used by the control method.
    """
    @abc.abstractmethod
    def se(self, nobs):
        """
        Calculates the standard error of samples with sizes `nobs`.

        Parameters
        ----------
        nobs : pandas.Series
            Sample sizes used to construct the statistic that will be monitored
            with this limit. The index must match the index of the `samples`
            argument passed to `ControlParams.statistic`.

        Notes
        -----
        This function calculates the standard deviation of the sampling
        distribution for the statistic that is being monitored. For example,
        in an XBar chart, `se` returns the standard deviation of the sampling
        distribution of the mean, ie. the standard error of the mean.

        Returns
        -------
        pandas.Series
            Magnitude of one standard deviation of the sampling distribution of
            this statistic for each sample size in `nobs`. Has the same index as
            the argument `nobs`.
        """
        pass

@dataclass
class XBarParams(ShewhartParams):
    """
    XBar (sample mean) control parameters.

    Use this class to construct traditional XBar charts. In addition to
    construction using class attributes, `XBarParams` can be constructed from
    sample standard deviation, sample range or historical data.

    Attributes
    ----------
    centre : float
        Centreline of the process distribution.
    sigma : float
        Standard deviation of the process distribution.
    nsigma : float, optional
        Distance in standard errors of the upper and lower control limits from
        the centreline.
    name : str, optional
        Name of the control method.
    """
    centre: float = field(repr=True)
    sigma: float = field(repr=True)
    nsigma: float = field(default=3, repr=False)

    name: str = field(default='XBar', repr=False)

    def statistic(self, samples):
        """
        XBar statistic; the sample mean.

        Parameters
        ----------
        samples : pandas.DataFrame
            Samples indexed by sample ID/number with columns for each observation.
            See :meth:`mqr.spc.ControlParams.statistic`.

        Returns
        -------
        :class:`ControlStatistic`
            Means of `samples`.
        """
        return ControlStatistic(
            stat=samples.mean(axis=1),
            nobs=samples.apply(len, axis=1))

    def se(self, nobs):
        """
        Standard error of xbar (standard error of the mean).

        Parameters
        ----------
        nobs : pandas.Series
            Number of observations in each sample. Index of `nobs` corresponds
            to the index of the argument `samples` passed to `statistic`.

        Returns
        -------
        pandas.Series
            Standard error of the mean for samples with sizes `nobs`.
        """
        return self.sigma / np.sqrt(nobs)

    def target(self):
        """
        Expected value of the sample mean.

        Returns
        -------
        float
        """
        return self.centre

    def lcl(self, nobs):
        """
        Calculates the lower control limits for samples with sizes in `nobs`.

        Parameters
        ----------
        nobs : pandas.Series
            Number of observations in each sample. Index of `nobs` corresponds
            to the index of the argument `samples` passed to `statistic`.

        Returns
        -------
        pandas.Series
        """
        return self.centre - self.nsigma * self.se(nobs)

    def ucl(self, nobs):
        """
        Calculates the upper control limits for samples with sizes in `nobs`.

        Parameters
        ----------
        nobs : pandas.Series
            Number of observations in each sample. Index of `nobs` corresponds
            to the index of the argument `samples` passed to `statistic`.

        Returns
        -------
        pandas.Series
        """
        return self.centre + self.nsigma * self.se(nobs)

    @staticmethod
    def from_stddev(centre, s_bar, nobs, nsigma=3):
        """
        Constructs XBarParams from an average sample stddev.

        Parameters
        ----------
        centre : float
            Process target mean.
        s_bar : float
            Average sample stddev from a reference in-control process.
        nobs : int
            Size (fixed) of samples used to calculate (s_bar).
        nsigma : float
            Number of stderr to set control limits.
        """
        return XBarParams(centre, s_bar / c4(nobs), nsigma, 'XBar(S)')

    @staticmethod
    def from_range(centre, r_bar, nobs, nsigma=3):
        """
        Constructs XBarParams from an average range.

        Using sample range to estimate process standard deviation can be
        inefficient. See :doc:`/notes/stddev-efficiency`.

        Parameters
        ----------
        centre : float
            Process target mean.
        r_bar : float
            Average sample range from a reference in-control process.
        nobs : int
            Size (fixed) of samples used to calculate (`r_bar`).
        nsigma : float
            Number of stderr to set control limits.
        """
        return XBarParams(centre, r_bar / d2(nobs), nsigma, 'XBar(R)')

    @staticmethod
    def from_data(samples, method='s_bar', nsigma=3):
        """
        Constructs XBarParams from reference samples.

        Reference samples are usually taken from an historical, in-control process.

        .. note::
            Using sample range to estimate process standard deviation can be
            inefficient. See :doc:`/notes/stddev-efficiency`.

        Parameters
        ----------
        samples : pandas.DataFrame
            Reference samples. Rows represent samples and columns represent
            observations.
        method : {'s_bar', 'r_bar'}, optional
            | 's_bar'
            |   Estimates process standard deviation from the sample standard
                deviation. Produces the traditional XBar-S chart.
            | 'r_bar'
            |   Estimates process standard deviation from the sample range.
                Produces the traditional XBar-R chart.
                See notes about efficiency :doc:`/notes/stddev-efficiency`.
        nsigma : float, optional
            Number of stderr to set control limits.
        """
        if method == 's_bar':
            centre = np.mean(samples, axis=1).mean()
            s_bar = np.std(samples.values, ddof=1, axis=1).mean()
            nobs = samples.shape[1]
            return XBarParams.from_stddev(centre, s_bar, nobs, nsigma)
        elif method == 'r_bar':
            centre = samples.mean(axis=1).mean()
            r_bar = np.ptp(samples, axis=1).mean()
            nobs = samples.shape[1]
            return XBarParams.from_range(centre, r_bar, nobs, nsigma)
        else:
            raise ValueError(f'Method {method} not supported.')

@dataclass
class RParams(ShewhartParams):
    """
    R (sample range) control parameters.

    Use this class to construct traditional R charts. In addition to
    construction using class attributes, `RParams` can be constructed from
    sample range or historical data.

    .. note::
        Using sample range to estimate process standard deviation can be
        inefficient. See :doc:`/notes/stddev-efficiency`.

    Attributes
    ----------
    centre : float
        Expected value of the process range.
    sigma : float
        Standard deviation of the process.
    nsigma : float, optional
        Distance in standard errors (of the range) of the upper and lower
        control limits from the centreline.
    name : str, optional
        Name of the control method.
    """
    centre: float = field(repr=True)
    sigma: float = field(repr=True)
    nsigma: float = field(repr=False, default=3)

    name: str = field(default='R', repr=False)

    def statistic(self, samples):
        """
        R statistic; the sample range.
        """
        return ControlStatistic(
            stat=np.ptp(samples, axis=1),
            nobs=samples.apply(len, axis=1))

    def se(self, nobs):
        """
        Standard error of the range.
        """
        return d3(nobs) * self.sigma

    def target(self):
        """
        Expected value of the sample range.
        """
        return self.centre

    def lcl(self, nobs):
        """
        Calculates the lower control limits for samples with sizes in `nobs`.
        """
        return np.clip(self.centre - self.nsigma * self.se(nobs), 0, np.inf)

    def ucl(self, nobs):
        """
        Calculates the upper control limits for samples with sizes in `nobs`.
        """
        return self.centre + self.nsigma * self.se(nobs)

    @staticmethod
    def from_range(r_bar, nobs, nsigma=3):
        """
        Constructs RParams from process range.

        Parameters
        ----------
        r_bar : float
            Expected value of process range.
        nobs : int
            Sample size used to estimate r_bar.
        nsigma : float
            Number of stderr to set control limits.
        """
        return RParams(r_bar, r_bar / d2(nobs), nsigma)

    @staticmethod
    def from_data(samples, nsigma=3):
        """
        Constructs RParams from reference samples.

        Reference samples are usually taken from an historical, in-control process.

        Parameters
        ----------
        samples : pandas.DataFrame
            Reference samples.
        nsigma : float
            Number of stderr to set control limits.
        """
        r_bar = np.ptp(samples, axis=1).mean()
        nobs = samples.shape[1]
        return RParams(r_bar, r_bar / d2(nobs), nsigma)

@dataclass
class SParams(ShewhartParams):
    """
    S (sample standard deviation) control parameters.

    Use this class to construct traditional S charts. In addition to
    construction using class attributes, `SParams` can be constructed from
    historical data.

    Attributes
    ----------
    centre : float
        Expected value of the process stddev.
    nsigma : float, optional
        Distance in standard errors (of the stddev) of the upper and lower
        control limits from the centreline.
    name : str, optional
        Name of the control method.
    """
    centre: float = field(repr=False)
    nsigma: int = field(repr=False, default=3)

    name: str = field(default='S', repr=False)

    def statistic(self, samples):
        """
        S statistic; the sample standard deviation.
        """
        return ControlStatistic(
            stat=samples.std(axis=1, ddof=1),
            nobs=samples.apply(len, axis=1))

    def se(self, nobs):
        """
        Standard error of the standard deviation
        """
        return self.centre * np.sqrt(1 - c4(nobs)**2) / c4(nobs)

    def target(self):
        """
        Expected value of the standard deviation.
        """
        return self.centre

    def lcl(self, nobs):
        """
        Calculates the lower control limits for samples with sizes in `nobs`.
        """
        return np.clip(self.target() - self.nsigma * self.se(nobs), 0, np.inf)

    def ucl(self, nobs):
        """
        Calculates the upper control limits for samples with sizes in `nobs`.
        """
        return self.target() + self.nsigma * self.se(nobs)

    @staticmethod
    def from_data(samples, nsigma=3):
        """
        Constructs SParams from reference samples.

        Reference samples are usually taken from an historical, in-control process.

        Parameters
        ----------
        samples : pandas.DataFrame
            Reference samples.
        nsigma : float
            Number of stderr to set control limits.
        """
        nobs = samples.shape[1]
        centre = samples.std(axis=1, ddof=1).mean()
        return SParams(centre, nsigma)

@dataclass
class EwmaParams(ControlParams):
    """
    EWMA control parameters.

    Use this class to construct EWMA charts for the mean of a sample. In addition
    to construction using class attributes, `EwmaParams` can be constructed from
    historical data.

    Attributes
    ----------
    mu_o : float
        Desired process mean.
    sigma : float
        Process standard deviation.
    lmda : float
        Decay rate.
    L : float
        Width of the control limits, in multiples of the standard deviation of
        the smoothed mean.
    steady_state : bool, optional
        Whether the process has already decayed to the steady state control limits.
    name : str, optional
        Name of the control method.

    Notes
    -----
    This monitoring strategy tracks a statistic based on either an observation
    or a sample. When the sample size is 1, the statistic is the EWMA of those
    samples. When the sample size is > 1, the statistic is the EWMA of the mean
    of those samples.

    The parameter sigma is an estimate of the standard deviation of the process,
    but the control limits are calculated from the standard error of the mean.
    When the sample size is 1, the standard error of the mean equals the standard
    deviation of the process.

    When the `EwmaParams` are created from data, `nobs` in the historical data
    need not match the sample size used to create the tracked statistic. The
    control limits will be scaled appropriately for the given sample size.
    """

    mu_0: float = field()
    sigma: float = field()
    lmda: float = field()
    L: float = field()

    steady_state: bool = field(default=False, repr=False)
    name: str = field(default='EWMA', repr=False)

    def statistic(self, samples):
        """
        Exponentially weighted average of the samples.

        Samples is first averaged over axis 1, so that when samples.shape[1] is
        2 or more, the statistic is tracking the sample mean.

        Parameters
        ----------
        samples : pandas.DataFrame
            Samples in rows, with one or more columns representing observations
            in each sample.
        """
        samples_z0 = pd.concat([pd.Series(self.mu_0), samples.mean(axis=1)])
        ewm = samples_z0.ewm(alpha=self.lmda, adjust=False).mean()
        return ControlStatistic(
            stat=ewm.iloc[1:],
            nobs=samples.apply(len, axis=1))

    def target(self):
        """
        Desired process mean.
        """
        return self.mu_0

    def lcl(self, nobs):
        """
        Calculates the lower control limits for samples with sizes in `nobs`.

        Parameters
        ----------
        nobs : pandas.Series
            Sample sizes used to construct the statistic at each sample index.

        Notes
        -----
        The argument `nobs` must be the size of the samples used to construct
        the statistic at each index.
        """
        idx = nobs.index.to_series()
        stderr = self.sigma / np.sqrt(nobs)
        if self.steady_state:
            sqrt_term = np.sqrt(self.lmda / (2 - self.lmda))
        else:
            sqrt_term = np.sqrt(self.lmda / (2 - self.lmda) * (1 - (1 - self.lmda)**(2 * idx)))
        return self.mu_0 - self.L * stderr * sqrt_term

    def ucl(self, nobs):
        """
        Calculates the upper control limits for samples with sizes in `nobs`.

        Parameters
        ----------
        nobs : pandas.Series
            Sample sizes used to construct the statistic at each sample index.

        Notes
        -----
        The argument `nobs` must be the size of the samples used to construct
        the statistic at each index.
        """
        idx = nobs.index.to_series()
        stderr = self.sigma / np.sqrt(nobs)
        if self.steady_state:
            sqrt_term = np.sqrt(self.lmda / (2 - self.lmda))
        else:
            sqrt_term = np.sqrt(self.lmda / (2 - self.lmda) * (1 - (1 - self.lmda)**(2 * idx)))
        return self.mu_0 + self.L * stderr * sqrt_term

    @staticmethod
    def from_stddev(mu_0, s_bar, nobs, lmda, L, steady_state=False):
        """
        Constructs EwmaParams from an average sample stddev.

        Parameters
        ----------
        mu_0 : float
            Process target mean.
        s_bar : float
            Average sample stddev from a reference in-control process.
        nobs : int
            Size (fixed) of samples used to calculate (s_bar).
        lmda : float
            Smoothing factor.
        L : float
            Width of the limits in multiples of the stddev of the smoothed mean.
        steady_state : bool, optional
            Whether the process has already decayed to the steady state control limits.
        """
        return EwmaParams(
            mu_0=mu_0,
            sigma=s_bar/c4(nobs),
            lmda=lmda,
            L=L,
            steady_state=steady_state)

    @staticmethod
    def from_data(samples, lmda, L, steady_state=False):
        """
        Constructs an instance of `EwmaParams` from reference samples.

        Reference samples are usually taken from an historical, in-control process.

        Parameters
        ----------
        samples : pandas.DataFrame
            Reference samples where each row is a sample and each column is an
            observation.
        lmda : float
            Smoothing factor.
        L : float
            Width of the limits in multiples of the stddev of the smoothed mean.
        steady_state : bool, optional
            Whether the process has already decayed to the steady state control limits.
        """
        N, nobs = samples.shape
        mu_0 = samples.mean(axis=1).mean()
        s_bar = np.std(samples.values, ddof=1, axis=1).mean()
        return EwmaParams.from_stddev(mu_0, s_bar, nobs, lmda, L, steady_state)

@dataclass
class MewmaParams(ControlParams):
    """
    MEWMA (multivaraite-EMWA) control parameters.

    Tools for calculating the `limit` based on ARL and h4 are provided in
    :func:`mqr.spc.solve_arl` and :func:`mqr.spc.solve_h4`.

    Attributes
    ----------
    mu : array_like
        Desired process output means.
    cov : array_like
        Covariance of process outputs.
    lmda : float
        Smoothing factor.
    limit : float
        Limit for the smoothed statistic, at the steady state.
    name : str, optional
        Name of the control method.

    Notes
    -----
    This class uses dataframe columns to represent process outputs rather than
    observations within a sample. That is, each row contains measurements that
    were taken at the same time (or equivalent), and columns contain measurements
    for the different process dimensions.
    """

    mu: np.ndarray = field(repr=False)
    cov: np.ndarray = field(repr=False)
    lmda: float = field(repr=False)
    limit: float = field(repr=False)

    name: str = field(default='Multivariate EWMA', repr=False)

    def statistic(self, samples):
        """
        Exponentially weighted average of the samples.

        Each row is a sample, and each column is a dimension in the process.

        Parameters
        ----------
        samples : pandas.DataFrame
        """
        init = pd.DataFrame(self.mu[None, :], columns=samples.columns)
        samples_0 = pd.concat([init, samples])
        z = (samples_0 - self.mu).ewm(alpha=self.lmda, adjust=False).mean().iloc[1:, :]
        t2 = pd.Series(index=samples.index)
        for i, idx in enumerate(samples.index):
            t2.iloc[i] = self._t2_stat(z.values[i].T, idx)
        return ControlStatistic(
            stat=t2,
            nobs=samples.apply(len, axis=1))

    def target(self):
        """
        Desired distance (a function of the norm) of the process from `mu`.

        Always zero.
        """
        return 0

    def lcl(self, nobs):
        """
        Calculates the lower control limits for samples with sizes in `nobs`.

        Always None, since MEWMA tracks a norm, which is always non-negative.
        """
        return None

    def ucl(self, nobs):
        """
        Calculates the upper control limits for samples with sizes in `nobs`.
        """
        return pd.Series(self.limit, index=nobs.index)

    def _cov_z(self, i):
        return self.lmda / (2 - self.lmda) * (1 - (1 - self.lmda)**(2 * i)) * self.cov

    def _t2_stat(self, z, i):
        return z @ np.linalg.inv(self._cov_z(i)) @ z[:, None]

    @staticmethod
    def from_data(samples, limit, lmda):
        """
        Constructs MewmaParams from reference samples.

        Reference samples are usually taken from an historical, in-control process.

        Parameters
        ----------
        samples : pandas.DataFrame
            Reference samples, with a sample per row and process dimensions in
            columns.
        limit : float
            Upper control limit. See :func:`mqr.spc.solve_arl` and
            :func:`mqr.spc.solve_h4`, which are tools for calculating this limit.
        lmda : float
            Smoothing factor.
        """
        mu = samples.mean(axis=0).values
        cov = samples.cov(ddof=1).values
        return MewmaParams(mu, cov, lmda, limit)
