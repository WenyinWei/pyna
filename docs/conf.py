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
    release = '0.8.22'
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
    'notebooks/tutorials/general_dynamics_geometry_patterns.ipynb',
    'notebooks/tutorials/analytic_stellarator_geometry_workflow.ipynb',
    'notebooks/tutorials/RMP_resonance_exec.ipynb',
    'notebooks/validate_chaos.ipynb',
]

# HTML theme
html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_js_files = ['language-switcher.js']
html_copy_source = False
html_show_sourcelink = False
html_title = 'pyna dynamics toolkit'

html_theme_options = {
    "logo": {
        "text": "pyna",
    },
    "navbar_align": "content",
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-language", "theme-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "header_links_before_dropdown": 6,
    "show_nav_level": 1,
    "show_toc_level": 2,
    "navigation_depth": 4,
    "collapse_navigation": True,
    "back_to_top_button": True,
    "navigation_with_keys": True,
    "secondary_sidebar_items": ["page-toc"],
    "pygments_light_style": "a11y-high-contrast-light",
    "pygments_dark_style": "a11y-high-contrast-dark",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/WenyinWei/pyna",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        },
    ],
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
# Tutorial notebooks are executed locally and committed with their outputs.
# The GitHub Pages workflow only renders those saved outputs, avoiding slow or
# non-reproducible CI reruns of Monte Carlo and field-line tracing tutorials.
nbsphinx_execute = 'never'
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


def _disable_nbsphinx_notebook_copy(app):
    """Render notebooks as HTML without publishing raw .ipynb sources."""
    if app.builder.format == 'html' and hasattr(app.env, 'nbsphinx_notebooks'):
        app.env.nbsphinx_notebooks = {}
    return []


def _remove_html_source_maps(app, exception):
    """Keep published pages compact and free of third-party source-map payloads."""
    if exception is not None or app.builder.format != 'html':
        return
    for root, _dirs, files in os.walk(app.builder.outdir):
        for filename in files:
            path = os.path.join(root, filename)
            if filename.endswith(('.css', '.js')):
                try:
                    with open(path, encoding='utf-8') as stream:
                        lines = stream.readlines()
                except UnicodeDecodeError:
                    lines = []
                filtered = [line for line in lines if 'sourceMappingURL=' not in line]
                if filtered != lines:
                    with open(path, 'w', encoding='utf-8') as stream:
                        stream.writelines(filtered)
            if filename.endswith('.map'):
                os.remove(path)


def setup(app):
    # nbsphinx copies executed notebooks during html-collect-pages.  The pages
    # already contain rendered outputs, and publishing raw JSON can expose large
    # base64 payloads, so keep GitHub Pages to HTML/assets only.
    app.connect('html-collect-pages', _disable_nbsphinx_notebook_copy, priority=400)
    app.connect('build-finished', _remove_html_source_maps)
