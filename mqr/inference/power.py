"""
Result type and presentation for power calculations.
"""

from dataclasses import dataclass
import numbers
import numpy as np
import mqr.styles

@dataclass
class TestPower:
    """
    Result of a sample size calculation for an hypothesis test.

    Arguments
    ---------
    name (str) -- Description of the hypothesis test.
    alpha (np.float64) -- Required significance level.
    beta (np.float64) -- Complement of the required power: 1 - power.
    effect (np.float64) -- Required effect size of the test.
    alternative (str) -- Sense of the test alternative. One of "two-sided",
        "less" or "greater".
    method (str) -- Name of the hypothesis test method, if applicable. For
        example, the t-test.
    sample_size (int) -- Lower bound on the sample size to achieve the above
        parameters.
    """
    name: str
    alpha: np.float64
    beta: np.float64
    effect: np.float64
    alternative: str
    method: str
    sample_size: int

    def as_text(self):
        from rich import table, color, text
        from rich.table import box, Table, Style

        table = Table(
            title=(
                text.Text('Sample Size', style=Style(bold=True)) +
                text.Text('\n'+self.name, style=Style(bold=True, color='grey50'))),
            title_style=Style(bold=True),
            title_justify='left',
            show_header=False,
            box=mqr.styles.default_table_box(),
            pad_edge=False,
            collapse_padding=True)
        if isinstance(self.effect, numbers.Number):
            effect_str = f'{self.effect:g}'
        else:
            effect_str = str(self.effect)
        table.add_row('alpha', f'{self.alpha:g}')
        table.add_row('beta', f'{self.beta:g}')
        table.add_row('effect', effect_str)
        table.add_row('alternative', self.alternative)
        table.add_row('method', self.method)
        table.add_section()
        table.add_row('sample size', f'{self.sample_size:g}', style=Style(bold=True))

        return table._repr_mimebundle_([], [])['text/plain']

    def _html(self):
        if isinstance(self.effect, numbers.Number):
            effect_str = f'{self.effect:g}'
        else:
            effect_str = str(self.effect)

        return f'''
        <table>
        <thead>
            <tr>
                <th scope="col" colspan=2 style="text-align: left; padding-bottom: 0px;">Sample Size</th>
            </tr>
            <tr style='padding-top: 0px;'>
                <td colspan=2 style='text-align: left; padding-top: 0px'>{self.name}</td>
            </tr>
        </thead>
        <tbody>
            <tr>
                <th scope='row' style='text-align: left;'>alpha</td>
                <td style='text-align: left;'>{self.alpha:g}</td>
            </tr>
            <tr>
                <th scope='row' style='text-align: left;'>beta</td>
                <td style='text-align: left;'>{self.beta:g}</td>
            </tr>
            <tr>
                <th scope='row' style='text-align: left;'>effect</td>
                <td style='text-align: left;'>{effect_str}</td>
            </tr>
            <tr>
                <th scope='row' style='text-align: left;'>alternative</td>
                <td style='text-align: left;'>{self.alternative}</td>
            </tr>
            <tr>
                <th scope='row' style='text-align: left;'>method</td>
                <td style='text-align: left;'>{self.method}</td>
            </tr>
            <thead><tr/></thead>
            <tr>
                <th scope='row' style='text-align: left;'>sample size</td>
                <td style='text-align: left;'>{self.sample_size:g}</td>
            </tr>
        </tbody>
        </table>
        '''

    def _repr_html_(self):
        return self._html()

    def _repr_pretty_(self, p, cycle):
        return p.text(self.as_text())
