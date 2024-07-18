import mqr.anova
import mqr.constants
import mqr.control
import mqr.doe
import mqr.inference
import mqr.msa
import mqr.nbtools
import mqr.plot
import mqr.process
import mqr.summary

# Sample data
def sample_data(name):
    import importlib
    return importlib.resources.files('mqr.data')/name
