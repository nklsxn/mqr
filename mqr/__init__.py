import importlib.metadata
__version__ = importlib.metadata.version('mqrpy')

import mqr.anova
import mqr.doe
import mqr.inference
import mqr.msa
import mqr.nbtools
import mqr.plot
import mqr.process
import mqr.spc
import mqr.transforms

# Sample data
def sample_data(name):
    import importlib
    return importlib.resources.files('mqr.data')/name

# Notebook formatters
import mqr.notebook
