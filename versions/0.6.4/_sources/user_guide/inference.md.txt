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

Basic Statistical Inference
===========================

Related modules
: <project:#mqr.inference><br>
  <project:#mqr.inference.nonparametric><br>
  <project:#mqr.plot.confint>

Detailed examples
: <https://github.com/nklsxn/mqr-guide>


The inference module is largely an interface to other libraries.
The module contains functions that calculate **sample size**,
**confidence intervals** and **hypothesis tests**.
Where a function calls another library to calculate a result,
the function's docstring shows the library and function that is called.

## Structure of the <tt>inference</tt> module

The structure of the inference module (like others in MQR) is supposed to read a bit like a menu.
The basic (ie. not ANOVA/regression) parametric routines are organised like this.
```
mqr.inference
├── dist
│   └── test_1sample
├── correlation
│   ├── confint
│   ├── test
│   └── test_diff
├── mean
│   ├── size_1sample
│   ├── size_2sample
│   ├── size_paired
│   ├── confint_1sample
│   ...
│   └── test_paired
└── stddev, variance, proportion, rate
    ├── size_1sample
    ...
    └── test_2sample
```

The non-parametric routines are organised like this.
```
mqr.inference.nonparametric
├── dist
│   ├── test_1sample
│   └── test_2sample
├── correlation
│   └── test
├── median
│   ├── test_1sample
│   └── test_nsample
├── quantile
│   ├── confint_1sample
│   └── test_1sample
└── variance
    └── test_nsample
```

## Result types

All sample-size, confidence interval and hypothesis test results are shown
as tables in notebooks.

### Sample size
Sample-size calculations have <project:#mqr.inference.power.TestPower> type,
shown in a notebook as:
```{code-cell} ipython3
test_size = mqr.inference.proportion.size_2sample(
    p1=0.2,
    p2=0.4,
    alpha=0.05,
    beta=0.2,
    alternative='less')
test_size
```
The `TestPower` result has fields that correspond to the values in the table:
```{code-cell} ipython3
print(test_size)
print(test_size.sample_size)
```

### Confidence intervals
Confidence interval calculations have <project:#mqr.inference.confint.ConfidenceInterval> type,
which are shown as:
```{code-cell} ipython3
ci = mqr.inference.rate.confint_1sample(
    count=80,
    n=100,
    meas=1.0,
    conf=0.98)
ci
```
Like `TestPower` results, confidence intervals are an object with fields corresponding to tabulated values.
Confidence intervals are also iterable, to allow the bounds to be easily unpacked.
```{code-cell} ipython3
lower, upper = ci
print(f'lower={lower}, upper={upper}')
print(f'bounded={ci.bounded}')
print(f'value={ci.value}')
```

### Hypothesis tests
Hypothesis tests have type <project:#mqr.inference.hyptest.HypothesisTest>,
and they are shown like this.
```{code-cell} ipython3
np.random.seed(0)

test = mqr.inference.nonparametric.median.test_nsample(
    st.uniform().rvs(20) - 0.4,
    st.uniform().rvs(20) + 0.2,
    st.uniform().rvs(20) * 2,
    method='kruskal-wallis',
    alternative='two-sided')
test
```
All the table rows are available as fields in the hypothesis test objects.
```{code-cell} ipython3
print(f'null-hypothesis: {test.null}')
print(f'statistic={test.stat}')
print(f'p-value={test.pvalue}')
```
