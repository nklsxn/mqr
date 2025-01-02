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

Design of Experiments
=====================

Related modules
: <project:#mqr.doe>

Detailed examples
: <https://github.com/nklsxn/mqrpy-guide>


## Introduction

The goal of experimental design is to efficiently collect enough data to make
conclusions about a particular set of questions.
In the case of ANOVA, the question is whether subsets of a sample have different means.
In the case of regression, the question is whether or not the terms in an equation
have non-zero coefficients.

While it is possible to come up with bespoke null-hypotheses and then derive the
distributions of related statistics,
many problems can be answered with a set of standard designs and their corresponding null-hypotheses.
MQR focuses on the standard designs that can be analysed with linear regression.
So, while ANOVA and regression are data analysis tools,
_design of experiments_ is about data collection.
Standard factorial, fractional factorial and central composite designs are demonstrated below.
Amongst design and process parameters, possibly many or them, these designs identify
* **main effects**,
  which are linear terms like $x$ in $z = a x + b y + c$,<br>
* **interactions**
  which are non-linear terms involving variables that change one another's effect on response like
  $x y$ in $z = a x + b y + c x y + d$, and<br>
* **quadratic effects**
  which are second-order non-linearities within a single variable (or curvature along an axis) like
  $y^2$ in $z = a x^2 + b y^2 + c x y + d$.
Additionally, MQR provides tools to randomise runs, partly randomise runs (as in split-plot designs),
and label point types to allow screening for curvature.

The main features of this module are:
* easy concatenation of test runs, allowing screening to be expanded to more detailed tests,
* labelling with point types and block labels, for easy management in analysis,
* transforming from coded variables to physical values or category labels, to assist experimental technique,
* easy randomisation, and
* creation of from standard designs (factorial, etc.) using `pyDOE3` functions.

For further information about `pyDOE3`, see <https://pydoe3.readthedocs.io>.


## Experimental workflow

The module is designed to support a workflow similar to:

1. (Analyse the process with process maps and FMEA, resulting in experimental variables.)
1. Design the experiment using the tools in `mqr.doe` (or `pyDOE3` directly).
1. Randomise the runs, possibly in blocks.
1. Save the design to a _design file_, and append the experimental observations
   as new columns to make an _experiment file_.
1. (optional) Instead of creating a new file, get the dataframe version of a design
   and enter data directly into the notebook as an extra column on the dataframe:
   `design['Observation'] = np.array([...])`
1. Load the _experiment file_ (if you created one) ready for analysis with ANOVA
   and regression tools.


## Experimental designs

The main type is <project:#mqr.doe.Design>.
The `Design` type is a class that looks a lot like a `pandas.DataFrame`.
The difference is that most operations in `mqr.doe` treat the various columns
("Runs", "Blocks", etc.) in particular ways, and to manage that treatment,
`Design` was not implemented as a single `DataFrame`.
To get the final `DataFrame`, call the <project:#mqr.doe.Design.to_df> method.
Also, when displayed in a notebook,`Design`s are displayed like the `DataFrame`
that would result from calling `Design.to_df()`.
The examples below show how to create basic experimental designs.

Note that the convenience functions `from_*` reflect the interface of `pyDOE3`:
<project:#mqr.doe.Design.from_ccdesign> calls `pyDOE3.ccdesign`.

### Full factorial designs

Full factorical designs test every level of every variable and all interactions.

```{code-cell} ipython3

names = ['x1', 'x2', 'x3']
levels = [2, 3, 2]

mqr.doe.Design.from_fullfact(names, levels)
```

### Fractional factorial designs

Fractional factorial designs use Yates labels to express desired aliasing.
For example, this design has every combination of two levels for `x1` and `x2`,
but counfounds `x3` with the `x1 * x2` interaction.

```{code-cell} ipython3

names = ['x1', 'x2', 'x3']
generator = 'a b ab'

mqr.doe.Design.from_fracfact(names, generator)
```

### A screening design that checks for curvature

This design adds three centre points to a $2^2$ full-factorial design to check for curvature.
The plus operator concatenates runs.
The mean of the centre points in this design can be compared to the mean of the corner points
using an appropriate hypothesis test (like <project:#mqr.inference.mean.test_2sample>).

Note that corner points have type 1 and centre points have type 0.

```{code-cell} ipython3

names = ['x1', 'x2']
levels = [2, 2]

fullfact = mqr.doe.Design.from_fullfact(names, levels)
centres = mqr.doe.Design.from_centrepoints(names, 3)

fullfact + centres
```

### Central composite design

Assuming curvature was detected or is known to exist from physical reasoning,
this central composite design characterises all main effects, interactions and
quadratic curvature along both axes.
The axial points (points lying on an axis) are designed to quantify quadratic effects.
Axial points are always labelled with point type 2.

The $\sqrt{2}$ magnitude of the axial points is deliberate
and creates desirable properties in the variance of the predicted response.
See [^1] and [^2] for more details.

The `centres` argument adds centre points, which are useful for estimating error,
and therefore increase the power of the ANOVA statistical tests.

```{code-cell} ipython3

names = ['x1', 'x2']
centres = (3, 3)

mqr.doe.Design.from_ccdesign(names, centres)
```

### Axial points

Central composite designs can be constructed manually using
factorial designs, centre points and axial points.
This example constructs the central composite design from the previous example.
Here, though, the blocking tools create orthogonal blocks from the corner and axial points.

```{code-cell} ipython3

names = ['x1', 'x2']
levels = [2, 2]

fullfact = mqr.doe.Design.from_fullfact(names, levels)
axial = mqr.doe.Design.from_axial(names, magnitude=np.sqrt(2))
centres = mqr.doe.Design.from_centrepoints(names, 3)

fullfact.as_block(1) + centres.as_block(1) + axial.as_block(2) + centres.as_block(2)
```

### Custom designs

MQR does not expose the latin hypercube or Box-Behnken designs from `pyDOE3`,
but they can be constructed easily.
This example calls `pyDOE3` directly, resulting in an `np.array` of levels.
The design is a Box-Behnken design with no centre points for three factors.

```{code-cell} ipython3
import pyDOE3

names = ['x1', 'x2', 'x3']
levels = pyDOE3.bbdesign(len(names), 0)

mqr.doe.Design.from_levels(names, levels)
```

## Practicalities

These features are to help with the practicalities of running experiments.

### Replication

Runs can be replicated.
The optional argument `label` adds a column that labels replicates.

```{code-cell} ipython3

names = ['x1', 'x2']
levels = [2, 2]

design = mqr.doe.Design.from_fullfact(names, levels)
design.replicate(3)
```

If necessary, the replicates can be labelled.

```{code-cell} ipython3

names = ['x1', 'x2']
levels = [2, 2]

design = mqr.doe.Design.from_fullfact(names, levels)
design.replicate(3, 'Rep')
```

### Randomisation

Rearrange the rows of a `pd.DataFrame` by calling <project:#mqr.doe.Design.randomise_runs>.
Randomisation uses the `numpy` random number generator, so seeding that generator will seed the MQR randomisation.
Blocks or factor levels can be kept in order by including the name of the block or factor in the argument `order`.

This example is the blocked central composite design from above.
The runs within each block are randomised, or put another way, the blocks are ordered.

```{code-cell} ipython3

names = ['x1', 'x2']
levels = [2, 2]

fullfact = mqr.doe.Design.from_fullfact(names, levels)
axial = mqr.doe.Design.from_axial(names, magnitude=np.sqrt(2))
centres = mqr.doe.Design.from_centrepoints(names, 3)
design = fullfact.as_block(1) + centres.as_block(1) + axial.as_block(2) + centres.as_block(2)

# Seeding with 0 for replicatable docs;
# in practise, use something like the date-time or a randomly generated seed.
np.random.seed(0)
design.randomise_runs('Block')
```

### Transforms
Writing down exactly which values correspond to each level is convenient for careful experimental technique.
These values can be read off a screen or printed while conducting an actual experiment.

First, define a transform that maps the levels that correspond to each label
(when `mqr.doe` constructs a transform from labels like below, it assumes the transform is affine).
The transforms can be callable objects like a lambda, function or `mqr.doe.Transorm`.
Or they can be dicts that give a mapped value for every coded level.

This example transforms
`x1` with an affine transform,
`x2` with an affine transform that is inversely proportional to the coded variable, and
`x4` with a categorical value.
The factor `x3` is left in coded units.

```{code-cell} ipython3

from mqr.doe import Transform

names = ['x1', 'x2', 'x3', 'x4']
levels = [2, 2, 2, 2]
design = mqr.doe.Design.from_fullfact(names, levels)

transforms = {
    'x1': Transform.from_map({-1:100, 1:110}),
    'x2': lambda x: -x + 5,
    'x4': {-1: 'low', 1: 'high'},
}
design.transform(**transforms)
```

### Saving to a file

There are a few options for saving designs to files.
If the design is going to a database, use python's pickle library, or some similar serialisation.

Below is an example that saves the design to a CSV or Excel file.
The index_label argument in `DataFrame.to_csv(...)` tells Pandas to include the index column with the given name.

```{code-cell} ipython3

names = ['x1', 'x2']
levels = [2, 2]
design = mqr.doe.Design.from_fullfact(names, levels)

# np.random.randint(0, 2**32-1)
np.random.seed(1294194915) # Randomly generated seed (above)
frozen_design = design.randomise_runs().to_df()

# Run these to create the actual files
# frozen_design.to_csv(
#     'doe-section6-1294194915.csv',
#     index_label='run')
# frozen_design.to_excel(
#     'doe-section6-1294194915.xlsx',
#     index_label='run')
```

## References

[^1]:   Montgomery, D. C. (2017).
        Design and analysis of experiments.
        John wiley & sons.
[^2]:   Box, G. E., & Hunter, J. S. (1957).
        Multi-factor experimental designs for exploring response surfaces.
        The Annals of Mathematical Statistics, 195-241.
