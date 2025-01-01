"""
=======================
Plots (:mod:`mqr.plot`)
=======================

.. currentmodule:: mqr.plot

.. rubric:: Functions
.. autosummary::
    :toctree: generated/

    Figure
    confint
    ishikawa

.. rubric:: Modules
.. autosummary::

    anova
    msa
    process
    correlation
    regression
    spc

.. rubric:: Modules
.. autosummary::

    tools

.. toctree::
    :maxdepth: 1
    :hidden:
    :titlesonly:

    plot/mqr.plot.anova
    plot/mqr.plot.msa
    plot/mqr.plot.process
    plot/mqr.plot.correlation
    plot/mqr.plot.regression
    plot/mqr.plot.spc
    plot/mqr.plot.tools
"""
from mqr.plot.figure import Figure

from mqr.plot.confint import confint
from mqr.plot.ishikawa import ishikawa
from mqr.plot.lib.util import grouped_df

import mqr.plot.anova as anova
import mqr.plot.correlation as correlation
import mqr.plot.msa as msa
import mqr.plot.process as process
import mqr.plot.regression as regression
import mqr.plot.spc as spc
import mqr.plot.tools as tools

from mqr.plot.lib.util import grouped_df
