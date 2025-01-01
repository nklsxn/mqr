import IPython.core.formatters

import mqr
from mqr.notebook.defaults import Defaults
from mqr.notebook import formatters

try:
    IPython.get_ipython().display_formatter.formatters["text/html"] = formatters.HTMLFormatter()
except:
    pass
