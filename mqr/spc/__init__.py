"""
============================================
Statistical Process Control (:mod:`mqr.spc`)
============================================

.. :currentmodule: mqr.spc

User guide
    :doc:`/user_guide/statistical-process-control`

Detailed examples
    https://github.com/nklsxn/mqrpy-guide


This module provides classes for statistical process control. It is designed for
use with :mod:`mqr.plot.spc`, which provides plotting routines for the types
defined in this module. The module can also be used independently of control
charts, for example in scripts that detect alarm conditions and take some action.

.. rubric:: Classes

These are superclasses that define the behaviour of chart parameters and statistics.

.. autosummary::
    :toctree: generated/

    ControlParams
    ShewhartParams
    ControlStatistic

These are subclasses of :class:`ControlParams`. Users can define new control
parameters by subclassing :class:`ControlParams` or :class:`ShewhartParams`.

.. autosummary::
    :toctree: generated/

    XBarParams
    RParams
    SParams
    EwmaParams
    MewmaParams

.. rubric:: Submodules

These submodules define alarm rules and other utilities, including unbiasing constants.

.. autosummary::

    rules
    util

"""

from mqr.spc.lib.control import ControlParams, ControlStatistic, ShewhartParams
from mqr.spc.lib.control import XBarParams, RParams, SParams, EwmaParams, MewmaParams
from mqr.spc.util import solve_arl, solve_h4

import mqr.spc.rules
import mqr.spc.util
