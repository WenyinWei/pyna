# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# import os
# import sys
# sys.path.insert(0, os.path.abspath('../'))

project = 'pyna'
copyright = '2022, Wenyin Wei'
author = 'Wenyin Wei'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
#     'sphinxcontrib.apidoc',
    'sphinx.ext.autodoc', 
    'sphinx.ext.doctest', 
    'sphinx.ext.todo', 
    'sphinx.ext.mathjax', 
    'sphinx.ext.ifconfig', 
    'sphinx.ext.viewcode', 
    'sphinx.ext.githubpages', 
    'sphinx.ext.napoleon', # to let sphinx understand Google/Numpy style comment
    ]
# apidoc_module_dir = '../pyna'
# apidoc_output_dir = './docs/api'
# apidoc_excluded_paths = [ ]
# apidoc_separate_modules = True

autosummary_generate = True

autodoc_default_options = {
    'members': True,
    # The ones below should be optional but work nicely together with
    # example_package/autodoctest/doc/source/_templates/autosummary/class.rst
    # and other defaults in sphinx-autodoc.
    'show-inheritance': True,
    'inherited-members': True,
    'no-special-members': True,
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
