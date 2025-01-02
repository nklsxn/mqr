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

Process Summary and Capability
==============================

Related modules
:   <project:#mqr.plot.ishikawa><br>
    <project:#mqr.process> (and <project:#mqr.plot.process>)<br>
    <project:#mqr.plot.correlation>

Detailed examples
:   <https://github.com/nklsxn/mqrpy-guide>


## Fishbone diagram

A fishbone diagram can be constructed with the tools in matplotlib,
though constructing all the lines by hand is a bit tedious,
so MQR includes a convenience function to create the plot.

The plot can create only two levels of "bones", including the "spine".
The top-level result is given in the argument `problem`
and the two levels of causes are given in the argument `causes`.

This is a general diagram.
```{code-cell} ipython3
problem = 'Problem'
causes = {
    'Method': ['Time consumption', 'Cost', 'Procedures', 'Inefficient process', 'Sampling'],
    'Machine': ['Faulty equipment', 'Compatibility'],
    'Material': ['Raw materials', 'Supplier', 'Shortage'],
    'Measurement': ['Calibration', 'Performance', 'Bad measurements'],
    'Environment': ['Bad conditions'],
    'People': ['Lack of training', 'Managers', 'Labor shortage', 'Procedures', 'Sales strategy'],
}

with Figure(8, 5) as (fig, ax):
    mqr.plot.ishikawa(problem, causes, ax=ax)
```

A more specific example for, say, the variability in the yield strength in a dimension
of an injection molded plastic part might be like this.
```{code-cell} ipython3
problem = '$\\mathbf{var}\\ \\sigma_y$'
causes = {
    'method': [
        'gate location', 'shear rate', 'hold pressure',
        'hold time', 'gate-wall ratio', 'location in batch mould'],
    'machine': [
        'time since service', 'time since sensor calibration',
        'warm-up time', 'mold surface texture'],
    'material': ['pellet material', 'colour (dye)', 'supplier'],
    'measurement': ['gauge capability', 'gauge calibration'],
    'environment': ['humidity at pellet hopper', 'ambient temp (cooling time)'],
    'people': ['removal technique', 'removal force'],
}

with Figure(8, 5) as (fig, ax):
    mqr.plot.ishikawa(problem, causes, ax=ax)
```

Often the defaults and matplotlib's spacing algorithms will present a tidy diagram.
The Ishikawa defaults in MQR can be viewed at
```{code-cell} ipython3
mqr.plot.defaults.Defaults.ishikawa
```
For this diagram,
split the long lines and increase the bone-space and cause-space.
```{code-cell} ipython3
causes = {
    'method': [
        'gate location', 'shear rate', 'hold pressure',
        'hold time', 'gate-wall ratio', 'location in\nbatch mould'],
    'machine': [
        'time since service', 'time since\nsensor calibration',
        'warm-up time', 'mold surface texture'],
    'material': ['pellet material', 'colour (dye)', 'supplier'],
    'measurement': ['gauge capability', 'gauge calibration'],
    'environment': ['humidity at pellet hopper', 'ambient temp\n(cooling time)'],
    'people': ['removal technique', 'removal force'],
}

ishikawa_kws = {'bone_space': 15.0, 'cause_space': 2.0}

with Figure(8, 5) as (fig, ax):
    mqr.plot.ishikawa(problem, causes, ax, ishikawa_kws)
```

## Summary statistics

Summary statistics are organised into the following types.

<project:#mqr.process.Summary>
: A set of samples from a process, optionally including specifications for the 
  dimensions which are used to calculate capability.

<project:#mqr.process.Specification>
: A target, and lower and upper limits.

<project:#mqr.process.Sample>
: Measurements from a single dimension in a process/product shown with a set of
  common descriptive statistics.

<project:#mqr.process.Capability>
: Formed from a sample and a specification, contains process potential and capability.


### Summary
Of those types, [`Summary`](#mqr.process.Summary) and [`Specification`](#mqr.process.Summary)
must be constructed manually.
The other two are created automatically when `Summary` is created with `Specification`s.

Create a summary of a process from a DataFrame.
```{code-cell} ipython3
data = pd.read_csv(mqr.sample_data('study-random-5x5.csv'))
data.head()
```

Pass the measurements from the DataFrame to the <project:#mqr.process.Summary> constructor.
```{code-cell} ipython3
summary = mqr.process.Summary(data.loc[:, 'KPI1':'KPO2'])
summary
```

Samples are automatically created as part of a `Summary`,
and can be retrieved using indexing syntax.
```{code-cell} ipython3
vstack(
    f'median = {summary['KPO1'].median}',
    summary['KPI2'],
)
```

### Specifications and capability

If specifications (<project:#mqr.process.Specification>) are passed to `Summary`
then capabilities will be included in the summary.
They are stored in the `Summary.capabilities` attribute.
```{code-cell} ipython3
specs = {
    'KPI1': mqr.process.Specification(150, 145, 155),       # cp=cpk~1.33
    'KPI2': mqr.process.Specification(20.25, 19.00, 21.50), # cp~1.67, cpk~1.33
    'KPI3': mqr.process.Specification(14.00, 11.72, 16.28), # cp=cpk~1.00
    'KPO1': mqr.process.Specification(160, 148, 172),       # cp=cpk~2.00
    'KPO2': mqr.process.Specification(2, -7.6, 11.6),       # cp~2.00, cpk~1.67
}

summary = mqr.process.Summary(data.loc[:, 'KPI1':'KPO2'], specs)
summary.capabilities
```

### Graphical summaries
These routines (from <project:#mqr.plot.process>) operate on the types from <project:#mqr.process>.

The histogram and box plot for a sample and also a confidence interval for its mean
can be shown on a plot with 
```{code-cell} ipython3
with Figure(5, 5, 3, 1, height_ratios=(4, 1, 1)) as (fig, axs):
    mqr.plot.process.summary(summary['KPO1'], axs)
```

If a summary was created with specifications, MQR can show sample capabilities graphically.
If the target capability is different from 2.0 (the default), pass the target as `cp`,
which truncates the tails of the fitted density function so that if it fits in the tolerance region,
then the sample has the specified capability.
```{code-cell} ipython3
with Figure(6, 10, 5) as (fig, axs):
    for ax, name in zip(axs, ['KPI1', 'KPI2', 'KPI3', 'KPO1', 'KPO2']):
        mqr.plot.process.capability(summary, name, cp=5/3, ax=ax)
```

## Correlation
Use <project:#mqr.plot.correlation> to show a detailed correlation plot.
The plot includes
histograms on the diagonal,
scatter plots and fitted lines on the lower triangle,
and statistics on the upper triangle.
Pass `show_conf=True` to include confidence intervals on the correlation coefficients.
Significant correlations are shown in bold.
```{code-cell} ipython3
with Figure(6, 6, 5, 5) as (fig, axs):
    mqr.plot.correlation.matrix(data.loc[:, 'KPI1':'KPO2'], axs, show_conf=True)
```
