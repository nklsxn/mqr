from dataclasses import dataclass, field
import numpy as np
import scipy

import mqr.inference.lib.util as util

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
    description : str
        Description of the statistic on which the test is performed.
    alternative : {'two-sided', 'less', 'greater'}
        Sense of the alternative hypothesis.
    method : str
        Name of the test method, eg. 'Kruskall-Wallace'.
    sample_stat : str
        Name of the statistic from the sample on which the test is performed.
        For example, when performing an hypothesis test on the mean of a sample,
        `sample_stat` is the mean. This is independent of the method.
    sample_stat_target : str
        Hypothesised value of the sample statistic. For example, when performing
        an hypothesis test on the mean of a sample, `sample_stat_target` is
        the hypothesised mean that appears in the null-hypothesis. Expressed as
        a string because some hypothesis tests target non-numeric properties,
        like distributions.
    sample_stat_value : float
        Actual value of the sample statistic calculated from the sample. Use
        `None` when the `sample_stat_target` is non-numeric.
    stat : float
        Test statistic. For example, in an hypothesis test on the mean of a
        sample, `stat` might be the score on a student's-t distribution.
    pvalue : float
        p-value associated with the test statistic.
    null : None
        Automatically generated. A string representation of the null-hypothesis.
    alt : None
        Automatically generated. A string representation of the alternative-hypothesis.

    Notes
    -----
    This class is iterable, and iterates over the test statistic and p-value (in
    that order).

    Examples
    --------
    All hypothesis tests return a type constructed like this.

    >>> import mqr
    >>> hyptest = mqr.inference.hyptest.HypothesisTest(
    >>>     description='Example mean test',
    >>>     alternative='less',
    >>>     method='Example students-t',
    >>>     sample_stat='mean(x)',
    >>>     sample_stat_target='1.6',
    >>>     sample_stat_value=1.4,
    >>>     stat=-2.828427124746192,
    >>>     pvalue=0.02371032779215975)
    >>> hyptest
    HypothesisTest(
        description='Example mean test',
        alternative='less',
        method='Example students-t',
        sample_stat='mean(x)',
        sample_stat_target='1.6',
        sample_stat_value=1.4,
        stat=-2.828427124746192,
        pvalue=0.02371032779215975,
        null='mean(x) == 1.6',
        alt='mean(x) < 1.6')

    Iterate over the test statistic and p-value:

    >>> stat, pvalue = hyptest
    >>> stat, pvalue
    (-2.828427124746192, 0.02371032779215975)

    When displayed in a jupyter notebook, the example here will be shown as an
    HTML table:

    +------------------------+---------------------+
    | | **Hypothesis Test**                        |
    | | Example mean test                          |
    +========================+=====================+
    | **method**             | Example students-t  |
    +------------------------+---------------------+
    | **H**:sub:`0`          | mean(x) == 1.6      |
    +------------------------+---------------------+
    | **H**:sub:`1`          | mean(x) < 1.6       |
    +------------------------+---------------------+
    | **statistic**          | -2.828427124746192  |
    +------------------------+---------------------+
    | **p-value**            | 0.02371032779215975 |
    +------------------------+---------------------+
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

    def __iter__(self):
        """
        Iterator over the test statistic and p-value.
        """
        return iter((self.stat, self.pvalue))
        
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
            raise RuntimeError(util.alternative_error_msg(self.alternative))

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
