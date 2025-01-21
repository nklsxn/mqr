---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: '1.1'
    jupytext_version: 1.1.0
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{code-cell} ipython3
: tags: [remove-input, remove-output]

import numpy as np
import pandas as pd
import scipy.stats as st

import mqr
from mqr.plot import Figure
from mqr.nbtools import hstack, vstack, grab_figure
```

Measurement System Analysis
===========================

Related modules
: <project:#mqr.msa> (and <project:#mqr.plot.msa>)

Detailed examples
: <https://github.com/nklsxn/mqr-guide>


MQR can automatically configure an ANOVA to perform a crossed GRR study.
Data should be arranged in a DataFrame with columns for the measurement, part identifier and operator identifier.
These columns can have any names, which will be configured in the next step.
The DataFrame can have other columns too, which will be ignored.
```{code-cell} ipython3
data = pd.read_csv(mqr.sample_data('grr.csv'))
data.head()
```


## Setting up a problem

Define the following parameters:
* tolerance,
* study variation,
* column name mapping.

MQR uses the column name mapping to configure the GRR regression.

```{code-cell} ipython3
usl = 10.5
lsl = 7.5
tol = usl - lsl
sv = 6.0
include_interaction = True

name_mapping = mqr.msa.NameMapping(
    measurement='Height',
    part='Part',
    operator='Operator')
```

Construct the GRR study using <project:#mqr.msa.GRR>.
The result is shown in notebooks as a table that summarises the inputs,
and shows the formula that MQR passed to statsmodels to calculate the regression.
```{code-cell} ipython3
grr = mqr.msa.GRR(
    data,
    tolerance=tol,
    names=name_mapping,
    include_interaction=include_interaction,
    nsigma=sv)
grr
```


## Checking the regression model

Use the regression result (`GRR.regression_result`) to analyse the model fit.
The tools from `mqr.anova` show a summary of the contrasts and the adquacy of the fit.
The residuals can be reviewed as demonstrated in [Analysing residuals](/user_guide/regression-anova).

In the summary below, the interaction is not significant, and the GRR should be re-run without it.
This example continues with the interaction included.

```{code-cell} ipython3
result = grr.regression_result

with Figure(5, 4, 2, 2) as (fig, ax):
    mqr.plot.regression.residuals(result.resid, result.fittedvalues, axs=ax)
    plot = mqr.nbtools.grab_figure(fig)

vstack(
    mqr.anova.adequacy(result),
    plot,
    mqr.anova.summary(result),
)
```

## Analysing results

The variance table can be shown by constructing the type <project:#mqr.msa.VarianceTable>.

```{code-cell} ipython3
mqr.msa.VarianceTable(grr)
```

A graphical representation of the GRR study can be displayed using <project:#mqr.plot.msa.grr>.
```{code-cell} ipython3
with Figure(10, 6, m=3, n=2) as (fig, axs):
    mqr.plot.msa.grr(grr, axs)
```
