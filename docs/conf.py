# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from starsmashertools import __version__ as version
import inspect
#import starsmashertools
import sys
import os
import subprocess

curdir = os.getcwd()
os.chdir(os.path.abspath('..'))
print(os.getcwd())
p = subprocess.Popen(['./install'])
p.wait()
os.chdir(curdir)

sys.path.insert(0, os.path.abspath('..'))

release = version

project = 'starsmashertools'
copyright = '2024, Roger Hatfull'
author = 'Roger Hatfull'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

# Mappings for sphinx.ext.intersphinx. Projects have to have Sphinx-generated doc! (.inv file)
intersphinx_mapping = {
    'python' : ('https://docs.python.org/3/', None),
    'numpy' : ('https://numpy.org/doc/stable/', None),
    'matplotlib' : ('https://matplotlib.org/stable/', None),
}

autosummary_generate = True  # Turn on sphinx.ext.autosummary
#autodoc_inherit_docstrings = False  # Prevent errors with Matplotlib inherited docs
add_module_names = False # Remove namespaces from class/method signatures

autoclass_content = 'both' # Add __init__ doc (ie. params) to class summaries
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

#autodoc_typehints = 'none'

templates_path = ['_templates']
exclude_patterns = [] #['build', 'Thumbs.db', '.DS_Store']

# -- Options for EPUB output
epub_show_urls = "footnote"

pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
#html_theme = 'alabaster'

html_static_path = ['_static']

def remove_python_comments(content):
    import re
    comment_checks = [
        # Remove # comments
        re.compile("(?<!['\"])#.*", flags = re.M),
    ]
    for check in comment_checks:
        for match in check.findall(content):
            content = content.replace(match, '')
    return content

def get_decorators(obj):
    code = remove_python_comments(inspect.getsource(obj))
    
    for line in code.split('\n'):
        line = line.strip()
        if not line: continue
        if '(' in line: line = line[:line.index('(')]
        yield line.replace('@', '')

def autodoc_skip_member(app, what, name, obj, skip, options):
    try:
        mod = obj.__module__

        if 'starsmashertools' not in mod: return True
        
        # Only include API-decorated objects
        for decorator in get_decorators(obj):
            if decorator == 'api': return False
    except: pass
    return True

def autodoc_process_signature(app, what, name, obj, options, signature, return_annotation):
    # Prevent the 'cli' keyword from being seen in the docs.
    import re
    for match in re.findall(r'(?=cli)[^),]*[, ]*', signature, flags=re.M):
        signature = signature.replace(match, '')
    return signature, return_annotation

def setup(app):
    app.connect('autodoc-skip-member', autodoc_skip_member)
    app.connect('autodoc-process-signature', autodoc_process_signature)
