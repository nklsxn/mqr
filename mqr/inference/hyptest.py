"""
Result type and presentation for hypothesis tests.
"""

from dataclasses import dataclass, field
import numpy as np
import scipy

def _hyptest_table_styles():
        return [
            {
                'selector': '.row_heading',
                'props': [
                    ('text-align', 'left'),
                ]
            },
        ]

@dataclass
class HypothesisTest:
    """
    Result of an hypothesis test.

    Attributes
    ----------
    description (str) -- Description of the statistic on which the test is performed.
    alternative (str) -- Sense of the alternative hypothesis. One of "two-sided",
        "less" or "greater".
    method (str) -- The name of the test method, eg. "Kruskall-Wallace".
    sample_stat (str) -- The name of the statistic from the sample on which the
        test is performed. For example, when performing an hypothesis test on the
        mean of a sample, `sample_stat` is the mean. This is independent of the method.
    sample_stat_target (str) -- The hypothesised value of the sample statistic.
        For example, when performing an hypothesis test on the mean of a sample,
        `sample_stat_target` is the hypothesised mean that appears in the null-hypothesis.
    sample_stat_value (np.float64) -- The actual value of the sample statistic
        calculated from the sample.
    stat (np.float64) -- The test statistic. For example, in an hypothesis test
        on the mean of a sample, `stat` might be the score on a student's-t
        distribution.
    pvalue (np.float64) -- The p-value associated with the test statistic.

    null (str) -- Automatically generated. A string representation of the null-hypothesis.
    alt (str) -- Automatically generated. A string representation of the alternative-hypothesis.
    """
    description: str
    alternative: str
    method: str
    sample_stat: str
    sample_stat_target: str
    sample_stat_value: np.float64
    stat: np.float64
    pvalue: np.float64

    null: str = None
    alt: str = None

    def __post_init__(self):
        self.null = self._null_hypothesis()
        self.alt = self._alt_hypothesis()

    def as_text(self):
        import mqr.styles
        from rich import table, text
        from rich.table import box, Table, Style

        title = (
            text.Text('Hypothesis Test', style=Style(bold=True)) +
            text.Text('\n'+self.description, style=Style(bold=True, color='grey50')))
        table = Table(
            title=title,
            title_justify='left',
            show_header=False,
            box=mqr.styles.default_table_box(),
            pad_edge=False,
            collapse_padding=True)
        table.add_row('method', self.method)
        table.add_row('null-hyp', self.null)
        table.add_row('alt-hyp', self.alt)
        table.add_section()
        table.add_row('statistic', f'{self.stat:g}')
        table.add_row('p-value', f'{self.pvalue:g}', style=Style(bold=True))

        return table._repr_mimebundle_([], [])['text/plain']
        
    def _null_hypothesis(self):
        import numbers
        if isinstance(self.sample_stat_target, numbers.Number):
            fmt = 'g'
        else:
            fmt = 's'
        return f'{self.sample_stat} == {self.sample_stat_target:{fmt}}'

    def _alt_hypothesis(self):
        import numbers

        alt_sym = None
        if self.alternative == 'two-sided':
            alt_sym = '!='
        elif self.alternative == 'less':
            alt_sym = '<'
        elif self.alternative == 'greater':
            alt_sym = '>'
        else:
            raise RuntimeError(f'Invalid alternative "{self.alternative}". Use "two-sided" (default), "less", or "greater".')

        if isinstance(self.sample_stat_target, numbers.Number):
            fmt = 'g'
        else:
            fmt = 's'

        return f'{self.sample_stat} {alt_sym} {self.sample_stat_target:{fmt}}'

    def _html(self):
        return f'''
        <table>
        <thead>
            <tr>
                <th scope="col" colspan=2 style="text-align: left; padding-bottom: 0px;">Hypothesis Test</th>
            </tr>
            <tr style='padding-top: 0px;'>
                <td colspan=2 style='text-align: left; padding-top: 0px'>{self.description}</td>
            </tr>
        </thead>
        <tbody>
            <tr>
                <th scope="row">method</th>
                <td>{self.method}</th>
            </tr>
            <tr>
                <th scope="row">H<sub>0</sub></th>
                <td>{self.null}</td>
            </tr>
            <tr>
                <th scope="row">H<sub>1</sub></th>
                <td>{self.alt}</td>
            </tr>
            <thead><tr/></thead>
            <tr>
                <th scope="row">statistic</th>
                <td>{self.stat:g}</td>
            </tr>
            <tr>
                <th scope="row">p-value</th>
                <td>{self.pvalue:g}</td>
            </tr>
        </tbody>
        </table>
        '''

    def _repr_html_(self):
        return self._html()

    def _repr_pretty_(self, p, cycle):
        return p.text(self.as_text())
