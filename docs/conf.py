# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'pyna'
copyright = '2024-2026, Wenyin Wei'
author = 'Wenyin Wei'
release = '0.3.0'
version = '0.3.0'

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
]

templates_path = ['_templates']
exclude_patterns = [
    '_build', 'Thumbs.db', '.DS_Store',
    # Only include tutorial notebooks; skip examples and research
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

# ? nbsphinx settings ?
nbsphinx_execute = 'auto'
nbsphinx_timeout = 300  # seconds per notebook

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

# ? autosectionlabel ?
autosectionlabel_prefix_document = True

# ? copybutton ?
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# Suppress duplicate-object warnings from existing source docstrings
suppress_warnings = [
    'autosectionlabel.*',
    'ref.duplicate',
]

nitpicky = False
