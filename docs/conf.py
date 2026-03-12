# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'pyna'
copyright = '2024, Wenyin Wei'
author = 'Wenyin Wei'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'myst_parser',
    'nbsphinx',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'navigation_depth': 4,
}

# MyST parser settings (for .md files)
myst_enable_extensions = [
    'amsmath',
    'colon_fence',
    'dollarmath',
]

# Napoleon settings (Google/NumPy docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# nbsphinx settings
nbsphinx_execute = 'never'  # don't re-execute notebooks during build

intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

autodoc_mock_imports = ['cupy', 'dolfinx', 'ufl', 'petsc4py', 'mpi4py']
