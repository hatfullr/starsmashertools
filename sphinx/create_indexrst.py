"""
We have to automatically generate the index.rst file because Sphinx isn't very
good or smart.

This file must ALWAYS be exactly 1 directory down from the starsmashertools 
source directory.
"""
import os



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
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    for _file in get_py_files(os.path.join(src, 'starsmashertools')):
        if os.path.basename(_file) == '__init__.py': continue

        # Check if an @api decorator is in the file somewhere
        with open(_file, 'r') as f:
            content = remove_python_comments(f.read())
        if '@api' not in content: continue

        module_path = os.path.relpath(_file, start = src).replace('.py', '')
        yield module_path.replace(os.sep, '.')


def create_API_module_rst_files(header_style = '=', subheader_style = '-'):
    import numpy as np
    modules = list(get_modules())
    names = [module.split('.')[1] for module in modules]
    names = np.unique(names)

    directory = os.path.dirname(__file__)

    files = []
    for name in names:
        line = header_style*len(name)
        header = line + '\n' + name + '\n' + line
        filename = 'starsmashertools.' + name + '.rst'
        mods = [module for module in modules if module.split('.')[1] == name]
        
        with open(filename, 'w') as f:
            f.write(header + '\n')
            f.write('\n')
            
            f.write('   .. autosummary::\n')
            f.write('      :toctree: ' + name + '\n')
            f.write('      \n')
            toadd = []
            for module in modules:
                if module.split('.')[1] != name: continue
                toadd += [module]
            for module in sorted(toadd):
                f.write('      ' + module + '\n')

        files += [filename]
    return files

if __name__ == '__main__':
    files = create_API_module_rst_files()

    with open('index.rst', 'w') as f:
        f.write("""
.. include:: gettingstarted.rst

.. toctree::
   :maxdepth: 1
   :hidden:

   Getting Started <self>
   cliprograms

""")
        f.write('\n')
        f.write('.. toctree::\n')
        f.write('   :caption: API Documentation\n')
        f.write('   :hidden:\n')
        f.write('   \n')
        for _file in files:
            f.write('   '+_file.replace('.rst', '')+'\n')
        f.write('\n')

