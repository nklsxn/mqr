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
import statsmodels

import mqr
from mqr.plot import Figure
from mqr.nbtools import hstack, vstack, grab_figure
```

Regression and ANOVA
====================

Related modules
: <project:#mqr.anova> (and <project:#mqr.plot.anova>, <project:#mqr.plot.regression>)

Detailed examples
: <https://github.com/nklsxn/mqrpy-guide>


MQR provides tools to analyse the results of ordinary least squares (OLS) and analysis of variance (ANOVA).
The examples in this guide and example notebooks use [statsmodels](https://www.statsmodels.org)
to compute the regressions.

The examples below use the `formula` API in statsmodels, which extracts data from a DataFrame by name.
In all OLS and ANOVA cases below, the problem is eventually represented (internally, by statsmodels)
as the matrix equation
```{math}
  y = A x,
```
where $x$ is unknown and the problem is overdetermined ($A$ is full-rank and has more rows than columns).
But, statsmodels also provides an interface to create models by passing the $y$ and $A$ directly;
see [statsmodels.regression.linear_model.OLS](https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLS.html).

The following examples rely on this initial data, constructed like a $2^2$ full-factorial experiment with 5 replicates.
```{code-cell} ipython3
np.random.seed(0)

data = np.vstack([
    np.tile([1, 3], 10),
    np.tile([0, 0, 2, 2], 5)]).T
levels = pd.DataFrame(data, columns=['X', 'Y'])

interact = levels.copy()
f_interact = lambda x, y: x + 2 * y - x * y
interact['Z'] = f_interact(interact['X'], interact['Y'])
interact['Z'] += st.norm(0, 0.5).rvs(len(levels))
```

That data looks like this, in 3D.
The blue mesh is the actual function,
the orange dots are the samples from the function with added noise, and
the blue dots are the means for each experimental configuration.
The black dots mark the four corner points of the experiment, projected onto the bottom of the plot.
The contours on the bottom plane are contour lines of the blue mesh.
```{code-cell} ipython3
: tags: [remove-input]

Xs = np.linspace(0.5, 3.5)
Ys = np.linspace(-0.5, 2.5)
X, Y = np.meshgrid(Xs, Ys)
Z = f_interact(X, Y)

zmin, zmax = np.min(Z)-0.5, np.max(Z)+0.5

subplot_kw = {'projection': '3d'}
with Figure(5, 5, subplot_kw=subplot_kw) as (fig, ax):
    ax.plot_surface(X, Y, Z, edgecolor='C0', lw=0.5, rstride=4, cstride=4, alpha=0.0, color='C0')
    ax.plot(xs=interact['X'], ys=interact['Y'], zs=interact['Z'], linewidth=0, marker='.', color='C1')
    
    ax.contour(X, Y, Z, offset=np.min(Z)-0.5)
    ax.plot(1, 0, zs=zmin, color='k', marker='o')
    ax.plot(3, 0, zs=zmin, color='k', marker='o')
    ax.plot(1, 2, zs=zmin, color='k', marker='o')
    ax.plot(3, 2, zs=zmin, color='k', marker='o')
    
    ax.plot(1, 0, zs=f_interact(1, 0), color='C0', marker='o')
    ax.plot(3, 0, zs=f_interact(3, 0), color='C0', marker='o')
    ax.plot(1, 2, zs=f_interact(1, 2), color='C0', marker='o')
    ax.plot(3, 2, zs=f_interact(3, 2), color='C0', marker='o')
    
    ax.view_init(20, 240, 0)
    ax.set(
        xlim=(0.5, 3.5),
        ylim=(-0.5, 2.5),
        zlim=(zmin, zmax),
        xlabel='X',
        ylabel='Y',
        zlabel='Z')
```

## Analysing residuals
MQR automates plots that help to check the assumptions of OLS.
They are
* normal probability plot,
* histogram with fitted normal PDF,
* residuals versus observation order, and
* residuals versus fitted value.

These plots can be shown together as a group by calling <project:#mqr.plot.regression.residuals>.
Pass an array of four axes.
Below a 2-by-2 array is used.
```{code-cell} ipython3
model = statsmodels.formula.api.ols('Z ~ X + Y', interact)
result = model.fit()

with Figure(5, 4, 2, 2) as (fig, axs):
    mqr.plot.regression.residuals(result.resid, result.fittedvalues, axs=axs)
```
In this case
* the normal probability plot shows the data has a heavy right tail,
* the histogram looks bi-modal,
* the residual versus observation index seems to show high and low grouping, and
* the residual versus fitted value shows at least two groups,
  a high group (first and last fitted value) and a low group (inner two fitted values).

This is caused by the interaction which could be captured with the model "Z ~ X + Y + X:Y".

Plots for residual-versus-factor can be shown with:
```{code-cell} ipython3
with Figure(5, 2, 1, 2) as (fig, axs):
    mqr.plot.regression.res_v_factor(result.resid, interact['X'], axs[0])
    mqr.plot.regression.res_v_factor(result.resid, interact['Y'], axs[1])
```

## Analysing categorical data

### Main effects
To show main effects, use <project:#mqr.plot.anova.main_effects>.
```{code-cell} ipython3
with Figure(5, 2, 1, 2) as (fig, axs):
    mqr.plot.tools.sharey(fig, axs)
    mqr.plot.anova.main_effects(
        interact,
        response='Z',
        factors=['X', 'Y'],
        axs=axs)
```

### Interactions
Use <project:#mqr.anova.interactions> to produce an interaction table,
and use <project:#mqr.plot.anova.interactions> to show interaction plots.
This example shows an interaction table between 'X' and 'Y',
and also a plot for the same interaction.
When there are more factors, the interactions table will average over the other factors,
whereas the interaction plot can plot each factor on a new axis, showing each factor's
interaction with the `group` factor ('Y' in this case).
```{code-cell} ipython3
interaction_table = mqr.anova.interactions(interact, value='Z', between=['X', 'Y'])
with Figure(3, 2) as (fig, axs):
    mqr.plot.anova.interactions(
        interact,
        response='Z',
        group='Y',
        factors=['X'],
        axs=axs)
    plot = grab_figure(fig)

hstack(interaction_table, plot)
```

Including the interaction in the model gives the following result.
The model components are now significant,
(though the main effects should be treated with care
since it might not be clear in practise how much of the main effect is actually interaction).
The residuals now look normally distributed,
there is no obvious pattern in the residual-versus-observation plot, and
the residual-versus-fitted-value plot shows group residuals distributed around their means of zero.
```{code-cell} ipython3
model = statsmodels.formula.api.ols('Z ~ X * Y', interact)
result = model.fit()

with Figure(5, 4, 2, 2) as (fig, axs):
    mqr.plot.regression.residuals(result.resid, result.fittedvalues, axs)
    plot = grab_figure(fig)

vstack(
    mqr.anova.summary(result),
    plot
)
```

### Model means
The means of experimental levels (eg. the mean at each corner point in a full-factorial experiment)
can be shown in a table with <project:#mqr.anova.groups>, and
can be plot with confidence intervals using <project:#mqr.plot.anova.model_means>.
The confidence intervals are constructed from the variance estimate that ANOVA makes from the regression error space.
The routine calculates averages over the factors that are not listed.
```{code-cell} ipython3
groups_x = mqr.anova.groups(result, value='Z', factor='X')
groups_y = mqr.anova.groups(result, value='Z', factor='Y')

with Figure(5, 2, 1, 2) as (fig, axs):
    mqr.plot.tools.sharey(fig, axs)
    mqr.plot.anova.model_means(result, response='Z', factors=['X', 'Y'], axs=axs)
    plot = grab_figure(fig)

vstack(
    hstack(groups_x, mqr.nbtools.Line.VERTICAL, groups_y),
    plot
)
```

## ANOVA/regression tables
MQR collects statistics about the fitted model into three tables.
* The <project:#mqr.anova.adequacy> table shows statistics on how well the model fits the data.
  R-squared and adjusted r-squared values are in this table.
* The <project:#mqr.anova.summary> table shows statistics related to the model components (contrasts).
  F-statistics and p-values for the group means and interactions are in this table.
* The <project:#mqr.anova.coeffs> table shows the coefficients of the terms in the model.
  t-statistics and p-values for the coefficients, and the variance inflation factor
  are all shown in this table.

```{code-cell} ipython3
vstack(
    'Adequacy',
    mqr.anova.adequacy(result),
    mqr.nbtools.Line.HORIZONTAL,
    'Summary',
    mqr.anova.summary(result),
    mqr.nbtools.Line.HORIZONTAL,
    'Coefficients',
    mqr.anova.coeffs(result)
)
```

The `summary` table shows the interaction is significant, as expected.
The main effects should be analysed separately now to determine
how much effect was due to the interaction.
The group means should also be reviewed.
