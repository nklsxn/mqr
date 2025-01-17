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

Plots
=====

:::{toctree}
:maxdepth: 1
:hidden:

plotting-primer.md
plotting-customisation.md
:::

Related modules
: <project:#mqr.plot><br>
  <project:#mqr.nbtools>

Detailed examples
: <https://github.com/nklsxn/mqrpy-guide>

Additional imports
: In addition to the user-guide [imports](/user-guide.md#imports),
  the pages (Plots, Plotting Primer and Customising MQR plots) also require the following imports.
```{code-cell} ipython3
from matplotlib import pyplot as plt
import seaborn as sns
```

<!--
## Plotting module

The library never renders a plot automatically, but instead expects users to
provide axes to draw into. This choice means the plotting libraries have no
side-effects, and it also allow the to change layout and plotting backend easily.

In the example notebooks, plots are wrapped in a `with plot.Figure(...)` context
manager, which creates figures, shows them, then closes them automatically. The
`(fig, ax)` that the context manager creates is the return value of
`matplotlib.pyplot.subplots(...)`. The context manager reduces the boilerplate
code required for the user (especially those unfamiliar with matplotlib) to
show a plot, and helps with a few other activities, like changing backends and
saving files. It is possible, for example, to quickly switch from showing plots
in a notebook to writing them into a backend that produces images for Word or
PGF/TikZ for LaTeX.

Of course, you can always create and manage axes directly, by calling
`ax, fig = matplotlib.pyplot.subplots(...)` and passing `ax` to the plotting
routines.
-->


## Basics
The summaries and example notebooks use two libraries: `matplotlib` and `seaborn`.
* **matplotlib** provides all the basic plotting tools, like figures and axes.
  It also provides various plot types (see <project:/user_guide/plotting-primer.md#elements> section).
* **seaborn** builds on matplotlib to provide more sophisticated plots, mostly related to visualising statistics.
  It also provides alternatives to some matplotlib plots.


### Getting help
Both libraries provide excellent documentation,
detailing how to use the libraries and showing examples.
There are two quick ways to access documentation.
1. In the Jupyter notebook, type the name of the function and parentheses,
   place the cursor somewhere between the parentheses, then type `shift-tab`.
   Eg. type `sns.boxplot(...)`, then while the cursor is somewhere in `(...)` type `shift-tab`.
1. Go to the websites below, and use the search function on their website to find a function, example, etc.

**matplotlib** <https://matplotlib.org/stable/><br>
**seaborn** <https://seaborn.pydata.org/index.html>


### Matplotlib figures and axes
Matplotlib plots are based on figures and axes.
A **figure** holds one or more **axes**.
For more info on figures and axes, see <https://matplotlib.org/stable/users/explain/axes/axes_intro.html>.

Create a figure with a given size like this:
```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6, 3))
plt.show(fig)
plt.close(fig)
```
In this guide and the example notebooks, the axes and figure variables are stored and used explicitly.
For example, `ax.plot(...)` is usually used to draw a line, and
`sns.histplot(..., ax=ax)` would be used to create a histogram.

Lots of examples on the internet will not assign the result of calling `subplots`.
If you do not specify an axis to plot onto, matplotlib will plot onto the most recently created axis.
We suggest keeping track of the axes you create by assigning them to variables like we do in this guide,
then plotting into them explicitly.
Explicit code is easier to read and easier to fix when there are problems.


### Figure context manager
MQR provides a wrapper around subplot creation.
The wrapper is written as a `with`-block called `Figure`.

The `with` construct is a python feature called a _context manager_.
It helps automatically initialise and then destroy resources (the figure in this case).
The `Figure` block creates a figure with the arguments supplied,
and does a few other things to automate common actions to make nice looking figures in notebooks.
The code it executes is here: [](https://github.com/nklsxn/mqr/blob/master/mqr/plot/figure.py).

You can use either construct: `fig, ax = matplotlib.subplots(...)` or `with Figure(...) as (fig, ax): ...`.

```{code-cell} ipython3
with Figure() as (fig, ax):
    pass
```


### MQR notebook tools
The module <project:#mqr.nbtools> can show matplotlib plots alongside other objects.
To capture the matplotlib output, use <project:#mqr.nbtools.grab_figure>, as below.
The result of `grab_figure` is an HTML image element with the figure's data embedded directly as a png image.

After grabbing the image data, the figure is destroyed and matplotlib will not render it (unless `suppress=False`).
Instead, call display on the returned HTML, or combine it with other HTML (like in <project:#mqr.nbtools.hstack>, etc).

For multiple plots, use `matplotlib.pyplot.subplots`,
because it has comprehensive features for showing plots next to each other (like shared axes and height/width ratios).
Grabbing a figure is useful for placing figures next to `DataFrame`s and other objects that implement `_repr_html_`.

```{code-cell} ipython3
line_xs = np.linspace(0, 10)
line_ys = line_xs ** 2

point_xs = np.linspace(0, 10, 11)
point_ys = point_xs ** 2

with Figure(4, 4) as (fig, ax):
    ax.plot(
        line_xs, # x values
        line_ys, # y values
        linewidth=0.5, color='C0')
    ax.plot(
        point_xs,
        point_ys,
        linewidth=0, color='C1', marker='o', fillstyle='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plot = mqr.nbtools.grab_figure(fig)

table_data = pd.DataFrame(
    data={'x': np.linspace(0, 10, 11),
          'y': np.linspace(0, 10, 11)**2})

mqr.nbtools.vstack(
    '#### Parabola',
    mqr.nbtools.hstack(
        plot,
        table_data
    )
)
```


## Directory
MQR constructs the following common plots for convenience.
All plots that require multiple axes flatten the `ax` argument before use,
so any dimensions that multiply to give the value of "No. axes" below will work,
excluding <project:#mqr.plot.correlation.matrix>, which must be exactly N by N.

Some common plots are not listed here, because there are already implementations in matplotlib and seaborn.
For example, MQR does not implement scatter plots, line plots, box plots or histograms.
The [](project:/user_guide/plotting-primer.md) shows how to construct those directly from matplotlib or seaborn.
[matplotlib plot types](inv:matplotlib:std#plot_types) and
[seaborn examples](inv:seaborn:std#examples/index) for examples of other plots.


```{rubric} Process
```
| Function | No. axes | Description |
|:---      |:---      |:---         |
|<project:#mqr.plot.ishikawa>| 1 | Fishbone/Ishikawa diagram |
|<project:#mqr.plot.correlation.matrix>| N by N | Matrix of scatter plots, histrograms and correlation statistics |
|<project:#mqr.plot.confint>| 1 | Interval and points showing a confidence interval and hypothesised value |
||
|<project:#mqr.plot.process.summary>| 3 | For a sample: histogram, boxplot and confidence interval of the mean |
|<project:#mqr.plot.process.pdf>| 1 | Probability density of a process estimating its yield |
|<project:#mqr.plot.process.tolerance>| 1 | Shaded region representing a tolerance |
|<project:#mqr.plot.process.capability>| 1 | For a process: a histrogram, pdf and tolerance overalyed, showing capability. |

```{rubric} Regression
```
| Function | No. axes | Description |
|:---      |:---      |:---         |
|<project:#mqr.plot.regression.residuals>| 4 | For an ols result: a tableau of the above three plots, and a probability plot |
|<project:#mqr.plot.regression.influence> | 1 | For an OLS result: bar graphs of an influence statistic |
|<project:#mqr.plot.regression.res_probplot> | 1 | For OLS residuals: normal probability plot |
|<project:#mqr.plot.regression.res_histogram>| 1 | For OLS residuals: histogram with overlayed density |
|<project:#mqr.plot.regression.res_v_obs>| 1 | For OLS residuals: residuals vs. observations |
|<project:#mqr.plot.regression.res_v_fit>| 1 | For OLS residuals: residuals vs. fitted values |
|<project:#mqr.plot.regression.res_v_factor>| 1 | For OLS residuals: residuals vs. factor data |

```{rubric} ANOVA
```
| Function | No. axes | Description |
|:---      |:---      |:---         |
|<project:#mqr.plot.anova.main_effects>| N | Main effects for experimental data, for N effects |
|<project:#mqr.plot.anova.interactions>| N | Interactions between independent variables, for N interactions |
|<project:#mqr.plot.anova.model_means>| 1 | Means for each treatment combinations, as confidence bars |

```{rubric} Measurement system analysis
```
| Function | No. axes | Description |
|:---      |:---      |:---         |
|<project:#mqr.plot.msa.grr>| 6 | A tableau of the above six plots |
|<project:#mqr.plot.msa.bar_var_pct>| 1 | For a GRR: bar graph of percent contributions from variances |
|<project:#mqr.plot.msa.box_measurement_by_part>| 1 | For a GRR: box-plot of measurements by part |
|<project:#mqr.plot.msa.box_measurement_by_operator>| 1 | For a GRR: box-plot of measurments by operator |
|<project:#mqr.plot.msa.xbar_operator>| 1 | For a GRR: Xbar-chart for each operator |
|<project:#mqr.plot.msa.r_operator>| 1 | For a GRR: R-chart for each operator |
|<project:#mqr.plot.msa.line_part_operator_intn>| 1 | For a GRR: the part-operator interactions |

```{rubric} Statistical process control
```
| Function | No. axes | Description |
|:---      |:---      |:---         |
|<project:#mqr.plot.spc.chart>| 1 | Chart of a monitored statistics |
|<project:#mqr.plot.spc.alarms>| 1 | Chart overlay for out-of-control statistics |
|<project:#mqr.plot.spc.oc>| 1 | For a sample: an operating characteristic curve |


```{rubric} Other tools
```
| Class | Description |
|:---   |:---         |
|<project:#mqr.plot.Figure> | A context manager that wraps a call to `matplotlib.pyplot.subplots`, showing then closing the figure |
|<project:#mqr.nbtools.grab_figure> | A routine that renders a figure as _png_ into an HTML component, then closes the figure |
