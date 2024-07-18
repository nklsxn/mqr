"""
Result type and presentation of confidence intervals.
"""

from dataclasses import dataclass
import numpy as np
import scipy.stats as st
import warnings

import matplotlib.pyplot as plt

@dataclass
class ConfidenceInterval:
    """
    Result of calculating a confidence interval

    Attributes
    ----------
    name (str) -- Statistic on which the confidence interval was calculated.
    value (float) -- Value of the statistic.
    lower (float) -- Lower limit of the interval.
    upper (float) -- Upper limit of the interval.
    conf (float) -- Confidence dictating the width of the interval.

    Notes
    -----
    Printed as an HTML table in notebooks. Printed as a text table on the command line.

    Is iterable, which results in an iterator over the lower and upper bounds of
    the interval:
    >>> lower, upper = ConfidenceInterval(...)
    """

    name: str
    value: np.float64
    lower: np.float64
    upper: np.float64
    conf: np.float64

    def __iter__(self):
        return iter((self.lower, self.upper))

    def as_text(self):
        import mqr.styles
        from rich import table, text
        from rich.table import box, Table, Style

        title = (
            text.Text('Confidence Interval', style=Style(bold=True)) +
            text.Text('\n'+self.name, style=Style(bold=True, color='grey50')))
        table = Table(
            title=title,
            title_justify='left',
            box=mqr.styles.default_table_box(),
            show_header=True,
            pad_edge=False,
            collapse_padding=True)
        alpha = 1 - self.conf
        if np.isfinite(self.lower) and np.isfinite(self.upper):
            left_alpha = f'{alpha*100/2:g}%'
            right_alpha = f'{(1-alpha/2)*100:g}%'
        elif not np.isfinite(self.lower):
            left_alpha = ''
            right_alpha = f'{(1-alpha)*100:g}%'
        elif not np.isfinite(self.upper):
            left_alpha = f'{alpha*100:g}%'
            right_alpha = ''
        else:
            raise AttributeError('neither `lower` nor `upper` limit is defined')
        table.add_column('value', justify='left')
        table.add_column(f'[{left_alpha}', justify='left', header_style=Style(bold=False))
        table.add_column(f'{right_alpha}]', justify='right', header_style=Style(bold=False))
        table.add_row(
            text.Text(f'{self.value:g}', style=Style(bold=True)),
            f'{self.lower:g}',
            f'{self.upper:g}')

        return table._repr_mimebundle_([], [])['text/plain']

    def _html(self):
        alpha = 1 - self.conf
        if np.isfinite(self.lower) and np.isfinite(self.upper):
            left_alpha = f'{alpha*100/2:g}%'
            right_alpha = f'{(1-alpha/2)*100:g}%'
        elif not np.isfinite(self.lower):
            left_alpha = ''
            right_alpha = f'{(1-alpha)*100:g}%'
        elif not np.isfinite(self.upper):
            left_alpha = f'{alpha*100:g}%'
            right_alpha = ''
        else:
            raise AttributeError('neither `lower` nor `upper` limit is defined')

        return f'''
        <table>
        <thead>
            <tr>
                <th scope="col" colspan=3 style="text-align: left; padding-bottom: 0px;">Confidence Interval</th>
            </tr>
            <tr style='padding-top: 0px;'>
                <td colspan=3 style='text-align: left; padding-top: 0px'>{self.name}</td>
            </tr>
        </thead>
        <tbody>
            <tr>
                <th scope='col' style='text-align: left;'>value</th>
                <th scope='col' style='text-align: left;'>[{left_alpha}</th>
                <th scope='col' style='text-align: right;'>{right_alpha}]</th>
            </tr>
            <tr>
                <td style='text-align: left;'>{self.value:g}</td>
                <td style='text-align: left;'>{self.lower:g}</td>
                <td style='text-align: right;'>{self.upper:g}</td>
            </tr>
        </tbody>
        </table>
        '''

    def _repr_html_(self):
        return self._html()

    def _repr_pretty_(self, p, cycle):
        return p.text(self.as_text())
