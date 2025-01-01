"""
=======================
Plots (:mod:`mqr.plot`)
=======================

.. currentmodule:: mqr.plot

User guide
    :doc:`/user_guide/plots`

Detailed examples
    https://github.com/nklsxn/mqrpy-guide


.. rubric:: Functions
.. autosummary::
    :toctree: generated/

    Figure
    ishikawa
    confint

.. rubric:: Modules
.. autosummary::

    process
    correlation
    regression
    anova
    msa
    spc
    tools

.. rubric:: Modules
.. autosummary::

    tools

.. toctree::
    :maxdepth: 1
    :hidden:
    :titlesonly:

    plot/mqr.plot.process
    plot/mqr.plot.correlation
    plot/mqr.plot.regression
    plot/mqr.plot.anova
    plot/mqr.plot.msa
    plot/mqr.plot.spc
    plot/mqr.plot.tools
"""
from mqr.plot.figure import Figure

from mqr.plot.confint import confint
from mqr.plot.ishikawa import ishikawa
from mqr.plot.lib.util import grouped_df

import mqr.plot.process as process
import mqr.plot.correlation as correlation
import mqr.plot.regression as regression
import mqr.plot.anova as anova
import mqr.plot.msa as msa
import mqr.plot.spc as spc
import mqr.plot.tools as tools

from mqr.plot.lib.util import grouped_df
