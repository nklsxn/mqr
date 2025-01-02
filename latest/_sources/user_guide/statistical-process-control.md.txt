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
from matplotlib import pyplot as plt

import mqr
from mqr.plot import Figure
from mqr.nbtools import hstack, vstack, grab_figure
```

Statistical Process Control
===========================

Related modules
: [](#mqr.spc) (and [](#mqr.plot.spc))

Detailed examples
: <https://github.com/nklsxn/mqrpy-guide>

MQR provides classes for statistical process control. While the tools are designed
for use with the plotting functions, they can also be used independently of control charts,
for example in scripts that detect alarm conditions and take some automated action
like sending an email, displaying on a dashboard, etc.


## Background

Traditionally, charts based on range and standard deviation have been defined
and used together. For example, X-bar and R charts (both based on sample range)
are often defined together. In this module, charts are treated separately so
that [XBarParams](#mqr.spc.XBarParams) is used with R and S charts.
[XBarParams](#mqr.spc.XBarParams) has methods to construct its parameters
from either range or standard deviation.

When control charts were originally developed, they were drawn by hand. To make
drawing practical, the historical chart parameters (control, upper and lower
limits) were defined in terms of tabulated constants. For example, the upper control limit
of an XBar chart that was based on the sample range was often written with a constant
$A_2$ as $\mathrm{UCL} = \overline{\overline{x}} + A_2 \overline{R}$, where $A_2$
is $3/(d_2(n)\sqrt{n})$, which includes a factor to estimate process standard
deviation from sample range ($1 / d_2(n)$), the scale of the standard error of the mean
($1 / \sqrt{n}$), and the width of the limits ($3$ standard errors).
Since this library plots charts automatically, it does not use or provide a way to
calculate those traditional constants. Instead, it calculates all parameters
directly, using lookup tables for pre-calculated unbiasing constants.

If necessary, the traditional constants are easy to calculate from standard errors
and the unbiasing constants implemented in MQR (see <project:#mqr.spc.util>).
For details on those traditional parameters, see [^1].


## Data Handling

The routines in `mqr.spc` expect sample data to be passed in a specific way.
For now, all sample sizes must be the same, though the sample sizes used to construct parameters
need not match the size of the samples used to monitor a process in production.
Data should be formatted into pandas DataFrames with sample labels in the index.
That is, if a process is sampled twice per day, then the first sample on the first day
would be in the first row, the second sample that afternoon would be in the second row,
the sample from the next morning would be in row three, and so on.
Data should be formatted as follows.
* For XBarParams, RParams, SParams and EwmaParams each column corresponds to an observation in a sample.
* For MewmaParams each column corresponds to a measurements from a dimensions/KPIs.

This sample data is provided with MQR.
It is the format that should be used with XBarParams, RParams, etc.
It shows 20 samples labelled from "0" to "19" in the index.
Each sample has eight observations labelled in the columns as "x1" to "x8".
```{code-cell} ipython3

pd.read_csv(mqr.sample_data('spc.csv'))
```

## Control parameters

The main type for statistical quality control in MQR is [ControlParams](#mqr.spc.ControlParams).
ControlParams represent the target and limits of a process,
and also contain enough information to calculate the monitored statistic from a set of samples.
All control chart functions, therefore, required control parameters
corresponding to the chart.
The class [XBarParams](#mqr.spc.XBarParams) is an example of ControlParams and
represents the parameters required to monitor the sample mean (x-bar) of a process.

It is common to monitor in-control processes by comparing their statistics
against historical statistics that were calculated from known-good
processes or from reference values.
`ControlParams` can be used to represent those historical parameters,
or when historical parameters are not available,
`ControlParams` can be created from a production sample.

Many control strategies monitor sample statistics.
X-bar charts track the sample mean, while R charts track the sample range.
The monitored statistic is called the [ControlStatistic](#mqr.spc.ControlStatistic) in MQR.
Objects of this type are not normally calculated directly, but from
[ControlParams.statistic](#mqr.spc.ControlParams.statistic).
This example calculates the sample means of four samples, each of size three.
```{code-cell} ipython3

values = np.array([
    [4.3, 1.7, 6.2],
    [5.4, 5.4, 3.6],
    [6.8, 2.8, 7.1],
    [4.4, 4.2, 2.9],
])
sample_data = pd.DataFrame(values, index=[5, 6, 7, 8])

# process target is 4.5, process stddev (not stderr) is 1.8.
params = mqr.spc.XBarParams(4.5, 1.8)

# in this case, the statistic does not depend on the target or stddev,
# but does depend on the sample size.
ctrl_stat = params.statistic(sample_data)
ctrl_stat.stat
```

The ControlStatistic has the same units and is compared directly against the control limits.
The control limits often depend on the sample size.
For example, the XBar chart has limits that depend on the standard error of the mean,
and so they depend on the sample size.
The XBar example above has the following limits.
```{code-cell} ipython3

params.lcl(nobs=3), params.ucl(nobs=3)
```
The control statistic is required to plot a control chart, and must be
passed to plotting functions along with the control parameters.

The class [ShewhartParams](#mqr.spc.ShewhartParams) is a subclass of
[ControlParams](#mqr.spc.ControlParams) for strategies/charts whose
control limits are based on the standard error of their statistic.
For example, XBarParams, RParams and SParams are all ShewhartParams.


### Predefined control parameters

These are the instances of [ControlParams](#mqr.spc.ControlParams):<br>
[](#mqr.spc.XBarParams), [](#mqr.spc.RParams), [](#mqr.spc.SParams),
[](#mqr.spc.EwmaParams), [](#mqr.spc.MewmaParams).

New control parameters can be defined by subclassing
[ControlParams](#mqr.spc.ControlParams) or
[ShewhartParams](#mqr.spc.ShewhartParams).
New control parameters defined this was are supported by the plotting and alarming routines below.


## Alarm rules

In MQR, alarm rules encode the conditions that a control statistic must satisfy
in order to be considered out-of-control, and therefore warrant corrective action.

Alarm rules are not a special type in MQR, but instead are simple functions/Callables
with the signature:
```
(ControlStatistic, ControlParams) -> pandas.Series[bool]
```
where `Callable` is any object that can be called like a function,
ie. with arguments between parentheses.
The resulting pandas Series is an indexed list of True/False values,
where `True` indicates that the statistic has triggered the alarm at that index.
The convention (which is followed by all pre-defined rules) is that the alarm
marks the last index of any subset that violates the rule.
If subsets overlap, then each ending point will be marked, even if they are consecutive.
The alarms do not, on the other hand, mark all the points that contributed to the alarm.

The pre-defined alarms are all functions that return rules,
which means they are functions that create functions.
This can be confusing if you haven't seen it before, but their use is straigh-forward;
see the examples below and in the [API Reference](#mqr.spc.rules).
The pre-defined alarms are as follows:
[](#mqr.spc.rules.limits),
[](#mqr.spc.rules.aofb_nsigma),
[](#mqr.spc.rules.n_1side),
[](#mqr.spc.rules.n_trending).

This example shows a point that violates the limit of an XBar chart
(the data is here, the chart is shown in the next section).
```{code-cell} ipython3

mean_hist = 10.0
std_hist = 0.002
params = mqr.spc.XBarParams(mean_hist, std_hist)

# Load sample data (which is in-control)
df = pd.read_csv(mqr.sample_data('spc.csv'))
# Add an out-of-control point
df.loc[len(df)] = [10.0019, 10.0027, 10.0038, 10.0036, 9.9973, 10.0037, 10.0028, 10.0017]

statistic = params.statistic(df)
rule = mqr.spc.rules.limits()

alarms = rule(statistic, params)
alarms.tail()
```

### Combining rules

MQR provides a mechanism ([](#mqr.spc.rules.combine)) to build more complex alarm logic from simple elements.
Any number of alarms can be combined with a logical operator to produce a new alarm.
More information and examples are in the [API Reference](#mqr.spc.rules.combine).

For example, this example creates a rule that triggers an alarm when
* the statistic is less that the LCL or greater than the UCL, or
* the statistic has 3 of 4 consecutive points greater than 2 standard deviations from the target, or
* the statistic has 5 values either monotonic increasing or monotonic decreasing.
```{code-cell} ipython3

rule = mqr.spc.rules.combine(
    np.logical_or,
    mqr.spc.rules.limits(),
    mqr.spc.rules.aofb_nsigma(3, 4, 2),
    mqr.spc.rules.n_trending(5)
)
```
This rule is the same, except that it triggers an alarm only when
all three of those conditions are true at the same time.
```{code-cell} ipython3

rule = mqr.spc.rules.combine(
    np.logical_and,                     # AND instead of OR
    mqr.spc.rules.limits(),
    mqr.spc.rules.aofb_nsigma(3, 4, 2),
    mqr.spc.rules.n_trending(5)
)
```

### Custom rules

Any function that has the signature
```
(ControlStatistic, ControlParams) -> pandas.Series[bool]
```
can be used as an alarm rule.
All custom defined rules with this signature will work with the combination functions,
and will work with the plotting routines.
All pre-defined control rules are defined as functions.
The control rules are defined here, for reference:
<https://github.com/nklsxn/mqr/blob/master/mqr/spc/rules.py>.

Note that the index of the output series must match the index of the ControlStatistic argument.


## Control charts

To plot ControlStatistics against ControlParameters, use [](#mqr.plot.spc.chart).
This chart shows the sample data from [](#alarm-rules) above.
```{code-cell} ipython3
with Figure(6, 2) as (fig, ax):
    mqr.plot.spc.chart(statistic, params, ax=ax)
```

Alarm rules are shown as an overlay on the statistic.
The overlay shows a red point and a red-shaded region over the statistic plot.
The last point in the example above triggers an alarm according to the [`limits`](#mqr.spc.rules.limits) rule.
```{code-cell} ipython3
rule = mqr.spc.rules.limits()

with Figure(6, 2) as (fig, ax):
    mqr.plot.spc.chart(statistic, params, ax=ax)
    mqr.plot.spc.alarms(statistic, params, rule, ax=ax)
```

The same data will trigger the [`n_trending(4)`](#mqr.spc.rules.n_trending) rule.
This example demonstrates that only the last of the points in a violating subset is highlighted.
```{code-cell} ipython3
rule = mqr.spc.rules.n_trending(4)

with Figure(6, 2) as (fig, ax):
    mqr.plot.spc.chart(statistic, params, ax=ax)
    mqr.plot.spc.alarms(statistic, params, rule, ax=ax)
```

See more examples in [](#mqr.spc.rules).


## Historical vs. live data

ControlParams represent the target and limits of an in-control process.
Ideally, ControlParams should be created from historical or reference values for the process.
For example, XBarParams would be created from an historical process target and standard deviation directly.
That is true for all the types: the class attributes are the reference parameters.
```{code-cell} ipython3
mqr.spc.XBarParams(4.3, 0.21)
```

To create reference parameters from historical samples, use the `.from_data` methods, like this.
```{code-cell} ipython3
data = pd.read_csv(mqr.sample_data('spc.csv'))
mqr.spc.XBarParams.from_data(data)
```
:::{caution}
The `.from_data` methods can be used to create ControlParams from live data,
but be aware that the limits are then 3$\sigma$ (by default) from the mean of the live data.
As a result, the live data will rarely (about 0.03% of points, since the data were assumed to be normal)
trigger an alarm on its own limits.

ControlParams should only be created this way if the process is known to be in-control.
:::

ControlParams can also be created from the statistics of historical samples.
For example, RParams can be created from an historical sample range like this.
```{code-cell} ipython3
mqr.spc.RParams.from_range(2, nobs=5)
```
The resulting RParams has a target sample range of 2,
and is parameterised on a process standard deviation of $\sigma \approx 0.86$,
which was calculated from the sample range.
If the process standard deviation is known, use it directly:
```{code-cell} ipython3
mqr.spc.RParams(2, 0.86)
```

### Saving ControlParams

After ControlParams have been created they can be serialised to disk/file in two ways.
1. Use python's `pickle` library to serialise the object to an efficient binary format.
   Pickle data contains enough information to recreate the object directly,
   with no further type information.
1. Use `mqr.spc.ControlParams.asdict` to serialise the object to a dict and then JSON.
   The object can be recreated by passing the resulting dictionary to the corresponding constructor.

This is an example of pickling and then reconstructing an instance of RParams.
```{code-cell} ipython3
import pickle

params = mqr.spc.RParams(2, 0.86)
params_bin = pickle.dumps(params)
obj = pickle.loads(params_bin)

vstack(
    params_bin,
    obj,
)
```

This is an example of serliasing to JSON then reconstructing the same instance of RParams.
```{code-cell} ipython3
import json

json_str = json.dumps(params.asdict())
params_dict = json.loads(json_str)
obj = mqr.spc.RParams(**params_dict)

vstack(
    json_str,
    params_dict,
    obj,
)
```
Note that there is not enough information in the JSON representation to know
which type is being constructed.
Usually, the JSON will be stored in a way that makes it clear how it should be reconstructed.
That is, the JSON will be stored at a location that contains only RParams, or
the type name ("mqr.spc.RParams") could be stored as metadata with the JSON.


## References

[^1]:   Montgomery, D. C. (2009).
        Statistical quality control (Vol. 7).
        New York: Wiley.
