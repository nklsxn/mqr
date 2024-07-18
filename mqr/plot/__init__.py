"""
Routines for common plots.

Includes:
* fishbone diagrams (`ishikawa`),
* descriptive statistics (`summary`),
* correlation (`correlation`),
* confidence intervals (`confint`),
* process capability (`process`).
* residuals and interactions from regression (`regression`),
* measurement system analysis (`msa`).
* statistical process control charts (`control`), and
* probability plots (`probplot`).
"""
from mqr.plot.figure import Figure

from mqr.plot.lib.confint import confint
from mqr.plot.lib.ishikawa import ishikawa
from mqr.plot.lib.summary import summary
from mqr.plot.lib.util import grouped_df

import mqr.plot.lib.control as control
import mqr.plot.lib.correlation as correlation
import mqr.plot.lib.msa as msa
import mqr.plot.lib.probplot as probplot
import mqr.plot.lib.process as process
import mqr.plot.lib.regression as regression
