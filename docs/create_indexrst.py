"""
We have to automatically generate the index.rst file because Sphinx isn't very
good or smart.

This file must ALWAYS be exactly 1 directory down from the starsmashertools 
source directory.
"""
import os

src = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))


basic = """
starsmashertools
================

.. toctree::
   :maxdepth: 2

   Home <self>
   gettingstarted
   cliprograms

*****************
API documentation
*****************

.. autosummary::
   :toctree: _autosummary
   
   *modules*

"""


def get_py_files(directory):
    with os.scandir(directory) as it:
        for entry in it:
            if entry.is_dir(follow_symlinks=False):
                yield from get_py_files(entry.path)
            elif entry.is_file(follow_symlinks=False):
                if not entry.path.endswith('.py'): continue
                yield entry.path

def remove_python_comments(content):
    import re
    comment_checks = [
        # Remove # comments first
        re.compile("(?<!['\"])#.*", flags = re.M),
        # Then remove block comments (which can be commented out by #)
        re.compile('(?<!\')(?<!\\\\)""".*?"""', flags = re.M | re.S),
        re.compile("(?<!\")(?<!\\\\)'''.*?'''", flags = re.M | re.S),
    ]
    for check in comment_checks:
        for match in check.findall(content):
            content = content.replace(match, '')
    return content

def get_modules():
    # All the python files essentially are their own module
    for _file in get_py_files(os.path.join(src, 'starsmashertools')):
        if os.path.basename(_file) == '__init__.py': continue

        # Check if an @api decorator is in the file somewhere
        with open(_file, 'r') as f:
            content = remove_python_comments(f.read())
        if '@api' not in content: continue

        module_path = os.path.relpath(_file, start = src).replace('.py', '')
        yield module_path.replace(os.sep, '.')

modules = sorted(list(get_modules()))

indent = 0
for line in basic.split('\n'):
    if '*modules*' not in line: continue
    indent = line.index('*modules*')
    break

with open('index.rst', 'w') as f:
    f.write(basic.replace('*modules*', ('\n' + ' '*indent).join(modules)))
