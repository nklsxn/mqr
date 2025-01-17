# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import importlib

project = 'mqr'
copyright = '2024-2025 Nikolas Crossan and Kevin Otto'
author = 'Nikolas Crossan'
version = importlib.metadata.version('mqrpy')
release = version


# -- App setup ---------------------------------------------------------------

def setup(app):
    app.add_css_file('_static/mqr.css')


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.duration',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme',
    'numpydoc',
    'matplotlib.sphinxext.plot_directive',

    'sphinx.ext.coverage',
    'sphinx.ext.ifconfig',

    # 'myst_parser',
    'myst_nb',
]

autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
add_function_parentheses = False


# -- Options for myst-parser and myst-nb -------------------------------------
# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html

source_suffix = {
    '.rst': 'restructuredtext',
    '.ipynb': 'myst-nb',
    '.myst': 'myst-nb',
}

myst_enable_extensions = [
    "amsmath", #
    "attrs_block", #
    "attrs_inline", #
    "colon_fence", #
    "deflist", #
    "dollarmath", #
    "fieldlist", #
    "html_admonition", #
    # "html_image",
    # "linkify",
    "replacements", #
    "smartquotes", #
    # "strikethrough",
    # "substitution",
    "tasklist", #
]
myst_footnote_transition = False
myst_heading_anchors = 3
myst_links_external_new_tab = True
myst_url_schemes = {
    "doi": "https://doi.org/{{path}}",
    "http": None,
    "https": None,
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_sidebars = {
    "index": ["search-button-field"],
    "**": ["search-button-field", "sidebar-nav-bs"]
}
html_theme_options = {
    "github_url": "https://github.com/nklsxn/mqr",
    "header_links_before_dropdown": 6,
    "icon_links": [],
    "logo": {
        "text": "MQR",
    },
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    "navbar_persistent": [],
    "show_version_warning_banner": True,
    "secondary_sidebar_items": ["page-toc"],
    "switcher": {
        "version_match": version,
        "json_url": "https://raw.githubusercontent.com/nklsxn/mqr/refs/heads/master/docs/switcher.json"
    }
}
html_context = {
    "default_mode": "light",
}

# html_title = f"{project} v{version} Manual"
# html_last_updated_fmt = "%b %d, %Y"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_css_files = [
    'mqr.css',
]
html_static_path = ['_static']

# Output file base name for HTML help builder.
htmlhelp_basename = "project-templatedoc"

# mathjax_path = "scipy-mathjax/MathJax.js?config=scipy-mathjax"


# -- Intersphinx configuration -------------------------------------------------

intersphinx_mapping = {
    'matplotlib': ('https://matplotlib.org/stable', None),
    'seaborn': ('https://seaborn.pydata.org/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
    'statsmodels': ('https://www.statsmodels.org/stable', None),
    'reliability': ('https://reliability.readthedocs.io/en/latest', None),
    'pyDOE3': ('https://pydoe3.readthedocs.io/en/latest', None),
}


# -- matplotlib configuration --------------------------------------------------

import matplotlib as mpl
import mqr

plot_include_source = True
plot_pre_code = '''
import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt
import mqr
'''
plot_formats = ['png', 'pdf']
plot_rcparams = mqr.plot.defaults.Defaults.rc_params
plot_apply_rcparams = True
