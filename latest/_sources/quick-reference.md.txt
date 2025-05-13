Quick Reference
===============

Links to features organised by application.

## Quality

:::{list-table}
:header-rows: 1

*   - Application
    - User Guide
    - Examples
    - API Reference

*   - Process capability
    - <project:user_guide/summary-capability.md#summary-statistics>
    - 
    - <project:#mqr.process> <project:#mqr.plot.process>

*   - Measurement system analysis
    - <project:#user_guide/measurement-system-analysis>
    - [Notebook --- GRR](https://github.com/nklsxn/mqr-guide/tree/master/notebooks/6-GRR.ipynb)
    - <project:#mqr.msa> <project:#mqr.plot.msa>

*   - Correlation analysis
    - <project:user_guide/summary-capability.md#correlation>
    - 
    - <project:#mqr.plot.correlation>

*   - Power and sample-size calculation
    - <project:user_guide/inference.md#sample-size>
    - 
    - <project:#mqr.inference>

*   - Confidence intervals
    - <project:user_guide/inference.md#confidence-intervals>
    - [Notebook --- Confidence Intervals](https://github.com/nklsxn/mqr-guide/tree/master/notebooks/1-ConfidenceIntervals.ipynb)
    - <project:#mqr.inference>

*   - Hypothesis tests
    - <project:user_guide/inference.md#hypothesis-tests>
    - [Notebook --- Hypothesis Tests](https://github.com/nklsxn/mqr-guide/tree/master/notebooks/2-HypothesisTesting.ipynb)
    - <project:#mqr.inference> <project:#mqr.inference.nonparametric>
:::


## Reliability
See the comprehensive [reliability](inv:reliability:std#index) library for:
* parametric and non-parametric lifetime modelling,
* accelerated life testing,
* repairable systems,
* reliability testing,
* models of physical failure: stress-cycles, stress-strain, stress-life, creep, acceleration factors, etc., and
* censored lifetime testing data.

MQR and `reliability` can be used together,
and improvement efforts can just as easily target product reliability as product quality.


## Design, Development, Improvement

:::{list-table}
:header-rows: 1

*   - Application
    - User Guide
    - Examples
    - API Reference

*   - Design of experiments (factorial, central composite, etc.)
    - <project:user_guide/design-of-experiments.md>
    - [Notebook --- Design of experiments](https://github.com/nklsxn/mqr-guide/tree/master/notebooks/7-DOE.ipynb)
    - <project:#mqr.doe>

*   - Power and sample-size calculations
    - <project:user_guide/inference.md#sample-size>
    - 
    - <project:#mqr.inference>

*   - Confidence intervals
    - <project:user_guide/inference.md#confidence-intervals>
    - [Notebook --- Confidence Intervals](https://github.com/nklsxn/mqr-guide/tree/master/notebooks/1-ConfidenceIntervals.ipynb)
    - <project:#mqr.inference>

*   - Hypothesis tests
    - <project:user_guide/inference.md#hypothesis-tests>
    - [Notebook --- Hypothesis Tests](https://github.com/nklsxn/mqr-guide/tree/master/notebooks/2-HypothesisTesting.ipynb)
    - <project:#mqr.inference> <project:#mqr.inference.nonparametric>

*   - ANOVA, regression
    - <project:user_guide/regression-anova.md>
    - [Notebook --- ANOVA](https://github.com/nklsxn/mqr-guide/tree/master/notebooks/3-ANOVA.ipynb)
      [Notebook --- Fitted Line Regression](https://github.com/nklsxn/mqr-guide/tree/master/notebooks/4-FittedLineRegression.ipynb)
      [Notebook --- Multiple Regression](https://github.com/nklsxn/mqr-guide/tree/master/notebooks/5-MultipleRegression.ipynb)
    - <project:#mqr.anova> <project:#mqr.plot.anova> <project:#mqr.plot.regression>
:::

MQR uses [pyDOE3](inv:pyDOE3:std#index) to construct experiments.
The library pyDOE3 defines even more experimental designs
that can be easily incorporated into MQR tools, or created directly.


## Process Monitoring and Control

:::{list-table}
:header-rows: 1

*   - Application
    - User Guide
    - Examples
    - API Reference

*   - Acceptance sampling, AQL, LTPD, OC-curves
    - <project:user_guide/statistical-process-control.md#acceptance-sampling>
    - [Notebook --- Acceptance Sampling](https://github.com/nklsxn/mqr-guide/tree/master/notebooks/9-AcceptanceSampling.ipynb)
    - <project:#mqr.plot.spc.oc>

*   - Control charts
    - <project:user_guide/statistical-process-control.md>
    - [Notebook --- Statistical Quality Control](https://github.com/nklsxn/mqr-guide/tree/master/notebooks/8-StatisticalQualityControl.ipynb)
    - <project:#mqr.spc> <project:#mqr.plot.spc>

*   - Customisable monitoring rules
    - <project:user_guide/statistical-process-control.md#alarm-rules>
    - [Notebook --- Statistical Quality Control](https://github.com/nklsxn/mqr-guide/tree/master/notebooks/8-StatisticalQualityControl.ipynb)
    - <project:#mqr.spc.rules> <project:#mqr.plot.spc>
:::
