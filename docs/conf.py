# Configuration file for the Sphinx documentation builder.
import os
import sys
from importlib import metadata
sys.path.insert(0, os.path.abspath('..'))

project = 'pyna'
copyright = '2024-2026, Wenyin Wei'
author = 'Wenyin Wei'
try:
    release = metadata.version('pyna-chaos')
except metadata.PackageNotFoundError:
    release = '0.8.21'
version = '.'.join(release.split('.')[:2])

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'sphinx_design',
    'myst_parser',
    'nbsphinx',
    'autoapi.extension',
]

templates_path = ['_templates']
exclude_patterns = [
    '_build', 'Thumbs.db', '.DS_Store',
    # Keep scratch examples and research notebooks out of the public docs build.
    # They are listed from the tutorial index without being executed by Sphinx.
    'notebooks/examples/*',
    'notebooks/research/*',
]

# ? Theme ?
html_theme = 'furo'
html_static_path = ['_static']

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary":    "#0d6efd",   # blue
        "color-brand-content":    "#0d6efd",
        "color-admonition-background": "#e8f4f8",
        "color-sidebar-background": "#f0f8ff",
        "font-stack": "Inter, sans-serif",
        "font-stack--monospace": "JetBrains Mono, Fira Code, monospace",
    },
    "dark_css_variables": {
        "color-brand-primary":    "#4fc3f7",   # teal-blue for dark mode
        "color-brand-content":    "#4fc3f7",
        "color-sidebar-background": "#1a1d23",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

# ? MyST parser settings ?
myst_enable_extensions = [
    'amsmath',
    'colon_fence',
    'dollarmath',
    'html_admonition',
    'html_image',
]

# ? Napoleon settings (Google/NumPy docstrings) ?
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_admonition_for_notes = True
napoleon_use_rtype = False

# nbsphinx settings
nbsphinx_execute = 'auto'
nbsphinx_timeout = 300  # seconds per notebook (per-notebook override via metadata "nbsphinx": {"timeout": N})
nbsphinx_allow_errors = False

# ? intersphinx ?
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy':  ('https://numpy.org/doc/stable/', None),
    'scipy':  ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# ? autodoc ?
autodoc_mock_imports = ['cupy', 'dolfinx', 'ufl', 'petsc4py', 'mpi4py', 'deprecated', 'joblib', 'plotly']
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': True,
}

# AutoAPI provides a complete source-parsed reference for GitHub Pages without
# importing optional runtime backends at documentation build time.
autoapi_type = 'python'
autoapi_dirs = ['../pyna']
autoapi_root = 'en/api/generated'
autoapi_add_toctree_entry = False
autoapi_keep_files = False
autoapi_options = [
    'members',
    'undoc-members',
    'show-inheritance',
    'show-module-summary',
]
autoapi_ignore = [
    '*/__pycache__/*',
    '*/build/*',
]

# ? autosectionlabel ?
autosectionlabel_prefix_document = True

# ? copybutton ?
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# Suppress duplicate-object warnings from existing source docstrings
suppress_warnings = [
    'autosectionlabel.*',
    'ref.duplicate',
    'ref.python',
    'docutils',
    'toc.not_included',
    'toc.no_title',
    'autoapi.python_import_resolution',
    'sphinx_autodoc_typehints.forward_reference',
]

nitpicky = False
