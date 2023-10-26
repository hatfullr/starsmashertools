# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
#import os
#import sys
#path = os.path.abspath('../..')
#sys.path.insert(0, path)

from starsmashertools import __version__ as version
release = version

# -- Project information -----------------------------------------------------

project = "starsmashertools"
author = "Roger Hatfull"


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'numpydoc', # Needs to be loaded *after* autodoc.
    'sphinxext.github',
]

# Copied from from https://github.com/pypr/pysph/blob/master/docs/source/conf.py
autodoc_default_flags = ['show-inheritance']
autoclass_content = 'both'
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

# This probably isn't right
intersphinx_mapping = {
    'cyarray': ('https://cyarray.readthedocs.io/en/latest', None),
}

# original
#intersphinx_mapping = {
#    "rtd": ("https://docs.readthedocs.io/en/stable/", None),
#    "python": ("https://docs.python.org/3/", None),
#    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
#}
#intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'






# -- Options for EPUB output
epub_show_urls = "footnote"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []#"_build", "Thumbs.db", ".DS_Store"]

pygments_style = 'sphinx'



# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

