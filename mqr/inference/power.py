from dataclasses import dataclass
import numbers
import numpy as np

@dataclass
class TestPower:
    """
    Result of a sample size calculation for an hypothesis test.

    Attributes
    ----------
    name : str
        Description of the hypothesis test.
    alpha : float
        Required significance level.
    beta : float
        Complement of the required power: 1 - power.
    effect : float
        Required effect size of the test.
    alternative : {'two-sided', 'less', 'greater'}
        Sense of the test alternative.
    method : str
        Name of the hypothesis test method, if applicable. For example, 't-test'.
    sample_size : int
        Lower bound on the sample size to achieve the above parameters.

    Examples
    --------
    Calculations solving for power or sample size produce types like this.

    >>> mqr.inference.power.TestPower(
    >>>     name="PowerName",
    >>>     alpha=0.01,
    >>>     beta=0.10,
    >>>     effect=1.5,
    >>>     alternative="greater",
    >>>     method="TestMethod",
    >>>     sample_size=45)
    TestPower(
        name='PowerName',
        alpha=0.05, beta=0.1, effect=1.5,
        alternative='greater', method='TestMethod',
        sample_size=45)

    In jupyter notebooks, power calculations are rendered as an HTML table:

    +-----------------+------------+
    | | **Test Power**             |
    | | PowerName                  |
    +=================+============+
    | **alpha**       | 0.05       |
    +-----------------+------------+
    | **beta**        | 0.1        |
    +-----------------+------------+
    | **effect**      | 1.5        |
    +-----------------+------------+
    | **alternative** | greater    |
    +-----------------+------------+
    | **method**      | TestMethod |
    +-----------------+------------+
    | **sample size** | 45         |
    +-----------------+------------+

    """
    name: str
    alpha: np.float64
    beta: np.float64
    effect: np.float64
    alternative: str
    method: str
    sample_size: int

    def _html(self):
        if isinstance(self.effect, numbers.Number):
            effect_str = f'{self.effect:g}'
        else:
            effect_str = str(self.effect)

        return f'''
        <table>
        <thead>
            <tr>
                <th scope="col" colspan=2 style="text-align: left; padding-bottom: 0px;">Test Power</th>
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
