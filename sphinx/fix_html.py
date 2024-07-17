"""
autosummary from Sphinx creates awful names. Here I try to fix that.
"""
import os
import starsmashertools
import copy
import create_indexrst

def get_html_files():
    def get(d):
        with os.scandir(d) as it:
            for entry in it:
                if entry.is_dir():
                    yield from get(entry.path)
                    continue
                if not entry.path.endswith('.html'): continue
                yield entry.path
    yield from get(os.path.join(
        starsmashertools.SOURCE_DIRECTORY,
        'docs',
    ))
    yield from get(os.path.join(
        starsmashertools.SOURCE_DIRECTORY,
        'sphinx',
    ))

def scrub_names_in_toc():
    modules = list(create_indexrst.get_modules())

    searches = [
        'toctree-l2',
    ]
    
    for path in get_html_files():
        with open(path, 'r') as f:
            contents = f.read()

        lines = contents.split('\n')
        for i, line in enumerate(lines):
            for search in searches:
                if search not in line: continue
                for module in modules:
                    line = line.replace('>' + module + '<', '>' + module.split('.')[-1] + '<')
                    lines[i] = line

        with open(path, 'w') as f:
            f.write('\n'.join(lines))

if __name__ == '__main__':
    scrub_names_in_toc()
