Data Analysis
=============

:::{toctree}
:maxdepth: 1
:hidden:

inference.md
regression-anova.md
measurement-system-analysis.md
:::

## Hypothesis testing and confidence intervals
Section
: <project:inference.md>

Presents tools that are used to infer underlying characteristics from sampled measurements.
The tools include confidence intervals and hypothesis tests, and also sample-size calculations.
Both the characteristics of a single sample and also comparisons between samples are covered.

There is overlap between that applications of these routines and ANOVA.
As a guide, the `inference` module handles statements about single populations,
comparisons between two populations, and simple non-parametric comparisons between multiple populations.
More detailed structues like data from factorial experiments, split-plots designs, etc.
are handled with ANOVA and regression techniques.

## ANOVA and regression
Section
: <project:regression-anova.md><br>

Describes how to create linear models that explain the effect of many factors on a measurable response,
including factors that are quantitative/continuous and also factors that are categorical (eg. one tool vs. another).
MQR does not perform regression itself, but presents results in a consistent way.
There are several tools in python to run regression; this guide uses `statsmodels` in examples.
Also describes the analysis of how well a model explains the observations from which it was created.
The mathematical tools are *ordinary least squares* (OLS) and *analysis of variance* (ANOVA),
which can be viewed as a particular setup of an OLS problem.
This section shows how to organise data ready for regression,
and also describes the tools that MQR provides to analyse regression results.

## Measurement system analysis
Section
: <project:measurement-system-analysis.md><br>

Describes how to quantify the contribution of multiple sources of
variability that contribute to uncertain measurements.
The goal of measurement system analysis is to verify that
a gauge and measurement process is sufficiently precise to
(a) observe the changes that have to be made to improve a product or process and
(b) keep both stable.
