"""
Result type and presentation of confidence intervals.
"""

from mqr.inference.lib.util import bounded_error_msg

from dataclasses import dataclass
import numpy as np
import scipy.stats as st
import warnings

import matplotlib.pyplot as plt

@dataclass
class ConfidenceInterval:
    """
    Result type for confidence interval calculations.

    Attributes
    ----------
    name: str
        Statistic on which the confidence interval was calculated.
    method: str
        Statistical method for determining the interval.
    value: float
        Value of the bounded statistic.
    lower: float
        Lower limit of the interval.
    upper: float
        Upper limit of the interval.
    conf: float
        Confidence that determines the width of the interval.
    bounded: {'both', 'below', 'above'}
        Which sides of the statistic are bounded by this interval.

    Notes
    -----
    The type is iterable, and iterates over the lower and upper bounds (in that
    order).

    Examples
    --------
    The results of confidence interval calculations will be return like this.

    >>> import mqr
    >>> ci = mqr.inference.confint.ConfidenceInterval(
    >>>     'name', 'method',
    >>>     value=1.0, lower=0.0, upper=2.0,
    >>>     conf=0.98, bounded='both')
    >>> print(ci)
    ConfidenceInterval(
        name='test', method='method',
        value=1.0, lower=0.0, upper=2.0,
        conf=0.98, bounded='both')

    Iterate over the lower and upper bounds:

    >>> lower, upper = ci
    >>> lower, upper
    (0.0, 2.0)

    When displayed in a jupyter notebook, this class is shown as an HTML table:

    +-----------+-----------+-----------+
    | | **Confidence Interval**         |
    | | name                            |
    | | 98% (fisher-z)                  |
    +===========+===========+===========+
    | **value** | **lower** | **upper** |
    +-----------+-----------+-----------+
    | 1.0       | 0.0       | 2.0       |
    +-----------+-----------+-----------+
    """

    name: str
    method: str
    value: np.float64
    lower: np.float64
    upper: np.float64
    conf: np.float64
    bounded: str

    def __iter__(self):
        """
        Iterator over the lower and upper bounds of the confidence interval.
        """
        return iter((self.lower, self.upper))

    def _html(self):
        alpha = 1 - self.conf
        if self.bounded == 'both':
            left_alpha = f'{alpha*100/2:g}%'
            right_alpha = f'{(1-alpha/2)*100:g}%'
        elif self.bounded == 'above':
            left_alpha = ''
            right_alpha = f'{(1-alpha)*100:g}%'
        elif self.bounded == 'below':
            left_alpha = f'{alpha*100:g}%'
            right_alpha = ''
        else:
            raise ValueError(bounded_error_msg(bounded))

        return f'''
        <table>
        <thead>
            <tr style='padding-bottom:0px;'>
                <th scope="col" colspan=3 style="text-align:left; padding-bottom:0px;">Confidence Interval</th>
            </tr>
            <tr style='padding-top:0px; padding-bottom:0px;'>
                <td colspan=3 style='text-align: left; padding-top:0px; padding-bottom:0px;'>{self.name}</td>
            </tr>
            <tr style='padding-top:0px;'>
                <td style='text-align:left; padding-top:0px' colspan=3>{self.conf*100:g}% ({self.method})</td>
            </tr>
        </thead>
        <tbody>
            <tr>
                <th scope='col' style='text-align: left;'>value</th>
                <th scope='col' style='text-align: left;'>lower</th>
                <th scope='col' style='text-align: left;'>upper</th>
            </tr>
            <tr>
                <td style='text-align: left;'>{self.value:g}</td>
                <td style='text-align: left;'>{self.lower:g}</td>
                <td style='text-align: left;'>{self.upper:g}</td>
            </tr>
        </tbody>
        </table>
        '''

    def _repr_html_(self):
        return self._html()
