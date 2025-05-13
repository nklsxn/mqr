User Guide
==========

:::{toctree}
:maxdepth: 1
:hidden:

user_guide/summary-capability
user_guide/data-analysis
user_guide/design-of-experiments
user_guide/statistical-process-control
user_guide/plots
:::

The library covers tools used in the following topics.

## Defining a problem
Section
: <project:user_guide/summary-capability.md>

Shows how to describe a process using statistics calculated from samples of process
input and output. Also, provides tools to quickly calculate and plot process capability,
a measure of the precision and accuracy of a process.


## Measurement and analysis
Section
: <project:user_guide/data-analysis.md>

Covers basic inference, regression and ANOVA, and measurement system analysis.
The Inference section relates to confidence intervals, hypothesis tests and
sample-size calculations for single samples and comparisons between samples.
The Regression and ANOVA section further covers statistical inference with
linear models that include categorical and quantitative variables,
and systems with multiple inputs.
The Measurement System Analysis (MSA) section describes an application of ANOVA
that quantifies how much variance is contributed by gauges and operators to the total variance in
measurements from a product sample.


## Making improvements
Section
: <project:user_guide/design-of-experiments.md>

Discusses how to collect the right data in controlled experiments
to answer particular questions using regression and ANOVA.
This is important because regression and ANOVA cannot identify
the structure of a physical system unless the structure is reflected in the data.


## Controlling process
Section
: <project:user_guide/statistical-process-control.md>

Shows how to monitor a stable process to observe deviations before they result in defects.


## Plotting and notebook tools
Section
: <project:user_guide/plots.md>

MQR has a module dedicated to plots.
The submodules of the plotting module correspond to other modules in the library.
This section describes how the plotting code is structured and
the tools MQR provides for displaying plots in notebooks.

## Imports

The pages in this user guide include example code.
To ensure that the examples are up to date,
they are automatically executed and their output is shown in a cell below the code.
All pages that include code make the following imports before any code is run.
(The plotting pages make a few further imports; see _Additional imports_ at the top of <project:/user_guide/plots.md>.)

```
import numpy as np
import pandas as pd
import scipy.stats as st

import mqr
from mqr.plot import Figure
from mqr.nbtools import hstack, vstack, grab_figure
```
