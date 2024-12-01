import abc
from dataclasses import asdict, dataclass, field
import matplotlib.pyplot as plt
from mqr.spc.util import c4, c4_fn, d2, d3
import numpy as np
import pandas as pd
import scipy

@dataclass
class ControlStatistic:
    stat: pd.Series = field(repr=False)
    nobs: pd.Series = field(repr=False)

    def __post_init__(self):
        if len(self.stat) != len(self.nobs):
            raise ValueError('Series stat and nobs must be the same length.')

@dataclass
class ControlParams:
    @abc.abstractmethod
    def statistic(self, samples):
        pass

    @abc.abstractmethod
    def target(self):
        pass

    @abc.abstractmethod
    def lcl(self, nobs):
        pass

    @abc.abstractmethod
    def ucl(self, nobs):
        pass

    def asdict(self):
        return asdict(self)

@dataclass
class ShewhartParams(ControlParams):
    @abc.abstractmethod
    def se(self, nobs):
        pass

@dataclass
class XBarParams(ShewhartParams):
    centre: float = field(repr=True)
    sigma: float = field(repr=True)
    nsigma: float = field(default=3, repr=False)

    name: str = field(default='XBar', repr=False)

    def statistic(self, samples):
        return ControlStatistic(
            stat=samples.mean(axis=1),
            nobs=samples.apply(len, axis=1))

    def se(self, nobs):
        return self.sigma / np.sqrt(nobs)

    def target(self):
        return self.centre - self.nsigma * self.se(nobs)

    def ucl(self, nobs):
        return self.centre + self.nsigma * self.se(nobs)

    @staticmethod
    def from_stddev(centre, s_bar, nobs, nsigma=3):
        return XBarParams(centre, s_bar / c4(nobs), nsigma, 'XBar(S)')

    @staticmethod
    def from_range(centre, r_bar, nobs, nsigma=3):
        return XBarParams(centre, r_bar / d2(nobs), nsigma, 'XBar(R)')

    @staticmethod
    def from_data(samples, method='s_bar', nsigma=3):
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
    centre: float = field(repr=True)
    sigma: float = field(repr=True)
    nsigma: float = field(repr=False)

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
        return RParams(r_bar, r_bar / d2(nobs), nsigma=3)

    @staticmethod
    def from_data(samples, nsigma=3):
        r_bar = np.ptp(samples, axis=1).mean()
        nobs = samples.shape[1]
        return RParams(r_bar, r_bar / d2(nobs), nsigma)

@dataclass
class SParams(ShewhartParams):
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
        nobs = samples.shape[1]
        centre = samples.std(axis=1, ddof=1).mean()
        return SParams(centre, nsigma)

@dataclass
class EwmaParams(ControlParams):
    mu_0: float = field()
    sigma: float = field()
    lmda: float = field()
    L: float = field()

    steady_state: bool = field(default=False, repr=False)
    name: str = field(default='EWMA', repr=False)

    def statistic(self, samples):
        samples_z0 = pd.concat([pd.Series(self.mu_0), samples.mean(axis=1)])
        ewm = samples_z0.ewm(alpha=self.lmda, adjust=False).mean()
        return ControlStatistic(
            stat=ewm.iloc[1:],
            nobs=samples.apply(len, axis=1))

    def target(self):
        return self.mu_0

    def lcl(self, nobs):
        Ns = nobs.index.to_series()
        stderr = self.sigma / np.sqrt(nobs)
        if self.steady_state:
            sqrt_term = np.sqrt(self.lmda / (2 - self.lmda))
        else:
            sqrt_term = np.sqrt(self.lmda / (2 - self.lmda) * (1 - (1 - self.lmda)**(2 * Ns)))
        return self.mu_0 - self.L * stderr * sqrt_term

    def ucl(self, nobs):
        Ns = nobs.index.to_series()
        stderr = self.sigma / np.sqrt(nobs)
        if self.steady_state:
            sqrt_term = np.sqrt(self.lmda / (2 - self.lmda))
        else:
            sqrt_term = np.sqrt(self.lmda / (2 - self.lmda) * (1 - (1 - self.lmda)**(2 * Ns)))
        return self.mu_0 + self.L * stderr * sqrt_term

    @staticmethod
    def from_data(samples, lmda, L):
        N, nobs = samples.shape
        mu_0 = samples.mean(axis=1).mean()
        sigma = np.std(samples.values, ddof=1) / c4_fn(np.prod(samples.shape))
        return EwmaParams(mu_0, sigma, lmda, L)

@dataclass
class MewmaParams(ControlParams):
    mu: np.ndarray = field(repr=False)
    cov: np.ndarray = field(repr=False)
    lmda: float = field(repr=False)
    limit: float = field(repr=False)

    name: str = field(default='Multivariate EWMA', repr=False)

    def statistic(self, samples):
        init = pd.DataFrame(self.mu[None, :], columns=samples.columns)
        samples_0 = pd.concat([init, samples])
        z = samples_0.ewm(alpha=self.lmda, adjust=False).mean().iloc[1:, :]
        t2 = pd.Series(index=samples.index)
        for i in range(samples.shape[0]):
            t2.iloc[i] = self._t2_stat(z.values[i].T, i)
        return ControlStatistic(
            stat=t2,
            nobs=samples.apply(len, axis=1))

    def target(self):
        """
        Desired distance (norm) of the process from `mu`.

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
        return self.lmda / (2 - self.lmda) * (1 - (1 - self.lmda)**(2 * i+1)) * self.cov

    def _t2_stat(self, z, i):
        z_0 = (z - self.mu)[:, None]
        return z_0.T @ np.linalg.inv(self._cov_z(i)) @ z_0

    @staticmethod
    def from_data(samples, limit, lmda):
        mu = samples.mean(axis=0).values
        cov = samples.cov(ddof=1).values
        return MewmaParams(mu, cov, lmda, limit)
