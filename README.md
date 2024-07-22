mqr-quality
-----------
Tools for learning and practising manufacturing quality with python.

Installation
------------
`pip install mqr-quality`  

Import the package using:  
`import mqr`

See `examples` directory in the source code for notebooks with sample code.

Introduction
------------
This library is a toolkit for manufacturing quality activities using python.
Most of the tools are related to quite general statistical methods that are
commonly used in fields outside manufacturing.

Most of this library falls into one of two categories:
1. **Code that automates commonly used plots and tables.** These are elementary,
  in that they don't require much special knowledge of the subject area and are
  easy to interpret, but they require an understanding of libraries like numpy,
  pandas, matplotlib, etc, and quite a few lines of boilerplate code. The goal
  of including these in the library is to make common plotting and descriptive
  activities fast.
2. **Code that provides a uniform interface to functionality in other libraries.**
  Existing libraries provide good coverage of hypothesis tests, particularly
  numpy, scipy and statsmodels. However, because the tests come from statistical
  literature, and are organised varously by name, purpose or application, they
  can be difficult to navigate, especially for students with limited experience
  with python. Further, each library has a different interface. The purpose of
  wrapping these tools is (1) to organise tests by goal or application (eg.
  tests on means, tests on proportions), and (2) to provide a uniform interface
  that is easy for students to navigate and use in jupyter notebooks. The `doe`
  module is another example. It provides an interface for designing experiments.
  It can be used with `pyDOE3` (and provides convenience functions for that),
  but also provides extra features that help with the practicalties of designing
  experiments, collecting and analysing data. For example, experimental designs
  can be easily built up in a few lines of code by composing smaller sets of
  runs.
3. **Several example notebooks showing how to do common quality tasks.** The
  examples (in the `examples` directory) provide a summary of the functionality
  and arrangement of this library, and also demonstrations of some common
  activities in quality improvement. For example, the `basic-inference` notebook
  shows how the interface to hypothesis tests are arranged, while the `probability`
  notebook gives examples of calculating useful probabilities using the very
  good scipy library.

Overall, the code is intended to reduce the burden of understanding python and
its various libraries, so the user can focus on understanding processes,
designing experiments, and analysing results.

**A note on wrapped code**: excluding a few tests that we implemented (because
we couldn't find an existing implementation), the confidence interval code and
hypothesis tests call through to other libraries. We think there are benefits to
wrapping the interfaces when using a notebook. However, if you use these tests
outside of a notebook, for example in scripts or automated routines, or if you
need more advanced functionality that we didn't expose in our interface, you
should look up the original functions and then call the statistical libraries
directly.

Code Organisation
-----------------
The code is organised into the following modules.

- `summary`: summary statistics
- `process`: describing and presenting processes, including capability
- `inference`: basic confidence intervals, hypothesis testing and sample size
    calculation
- `anova`: tools for working with ANOVA calculations and results (statsmodels
    provides the regression)
- `msa`: measurement system analysis; an automatically constructed ANOVA for
    gauge reproducibility and repeatability studies
- `control`: plots for showing processes under statistical control
- `doe`: tools for designing experiments
- `plot`: various plots (below)

The plotting module has routines for generating a range of plots. The plots are
mostly arranged like this:

- `plot.Figure` (context manager): create subplots (calls `matplotlib.pyplot.subplots`)
- `plot.confint` (function): visually show the process of an hypothesis test on
  the mean of a data set
- `plot.ishikawa` (function): draw an Ishikawa (fishbone) diagram
- `plot.summary` (function): draw a histogram, box plot and confidence interval
  with shared x-axes
- `plot.grouped_df` (function): draw data from a dataframe grouped by column and
  drawn adjacent to each other, left to right
- `plot.control` (module): draw X-bar and R charts
- `plot.correlation` (module): plot a detailed correlation matrix, including
  histograms, scatter plots with best-fit lines, and confidence intervals on coefficients
- `plot.msa` (module): plot the results of measurement system analysis, including
  a GRR summary
- `plot.probplot` (module): probability plots of subsets of data, with shared
  summary statistics
- `plot.process` (module): graphically represent a process, including capability
- `plot.regression` (module): draw residuals from a regression analysis

Inference module
----------------
The inference module is mostly an interface to other libraries. The module contains
functions that calculate **sample size**, **confidence intervals** and
**hypothesis tests**.

The functions are arranged with the following naming scheme.  
Parametric tests:
* sample size: `mqr.inference.<statistic>.size_<sample>(...)`
* confidence interval: `mqr.inference.<statistic>.confint_<sample>(...)`
* hypothesis test: `mqr.inference.<statistic>.test_<sample>(...)`

Non-parametric tests have the same form, but are in the `nonparametric` module:
* hypothesis test: `mqr.inference.<statistic>.test_<sample>(...)`
* etc.

Where:
* `<statistic>` is the statistic of interest, for example, "mean", "correlation"
  coefficient, etc.
* `<sample>` is a description of the samples involved in the calculation, for
  example "1sample", "nsample". Some routines don't have a sample description.

For example, hypothesis tests that deal with the difference between the means of
two unpaired samples are at:  
`mqr.inference.mean.test_2sample(...)`

Plotting module
---------------
The library never renders a plot automatically, but instead expects users to
provide axes to draw into. This choice means the plotting libraries have no
side-effects, and it also allow the to change layout and plotting backend easily.

In the example notebooks, plots are wrapped in a `with plot.Figure(...)` context
manager, which creates figures, shows them, then closes them automatically. The
`(fig, ax)` that the context manager creates is the return value of
`matplotlib.pyplot.subplots(...)`. The context manager reduces the boilerplate
code required for the user (especially those unfamiliar with matplotlib) to
show a plot, and helps with a few other activities, like changing backends and
saving files. It is possilbe, for example, to quickly switch from showing plots
in a notebook to writing them into a backend that produces images for Word or
PGF/TikZ for LaTeX.

Of course, you can always create and manage axes directly, by calling
`ax, fig = matplotlib.pyplot.subplots(...)` and passing `ax` to the plotting
routines.

License
-------
This package is provided under the BSD License (3-clause).

Credit
------
Copyright (c) 2024 Nikolas Crossan, Kevin Otto

Supported by the University of Melbourne  
Department of Mechanical Engineering
