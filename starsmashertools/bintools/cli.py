# A CLI is a Command Line Interface that manages inputs and Page navigation.
import sys
import os
import argparse
import starsmashertools.bintools.page
import starsmashertools.bintools.inputmanager
import collections


class CLI(object):
    instance = None
    
    def __init__(self, name, description):
        if CLI.instance is not None:
            raise Exception("Only one CLI instance is permitted per process")
        
        self.name = name
        self.description = description
        super(CLI, self).__init__()

        CLI.instance = self
        
        self.inputmanager = starsmashertools.bintools.inputmanager.InputManager()
        
        self.parser = argparse.ArgumentParser(
            prog = self.name,
            description = self.description,
        )

        self.parser.add_argument(
            '-o',
            '--output',
            nargs='?',
            required=False,
            metavar = 'filename',
            type=str,
            help = "Outputs that are written to the terminal will also be written/appended to the given file name."
        )
        
        self._args = {}
        self.pages = collections.OrderedDict()
        self._mainmenu = None

    @property
    def args(self):
        if not self._args:
            args = self.parser.parse_args()
            for item in dir(args):
                if not item.startswith('_'):
                    self._args[item] = getattr(args, item)
        return self._args

    def _prepare_page(self, page):
        if page.identifier is None: page.identifier = len(self.pages)
        if page.identifier in self.pages.keys():
            raise KeyError("Page with identifier '%s' is already added to this CLI" % str(page.identifier))
        if page.inputmanager is None: page.inputmanager = self.inputmanager
        return page

    def _add_page(self, page):
        page = self._prepare_page(page)
        self.pages[page.identifier] = page
        return page

    @staticmethod
    def write(*args, **kwargs):
        ret = print(*args, **kwargs)
        filename = CLI.instance.args['output']
        if filename is not None:
            mode = 'w'
            if os.path.isfile(filename): mode = 'a'
            content = ' '.join(args)
            content = starsmashertools.bintools.Style.clean(content)
            with open(filename, mode) as f:
                f.write(content)
        return ret
    
    def set_mainmenu(self, page):
        self._mainmenu = page

    def reset(self):
        if self._mainmenu is not None:
            self.navigate(self._mainmenu.identifier)
        else: quit()
        
    def add_page(self, inputtypes, contents, **kwargs):
        return self._add_page(starsmashertools.bintools.page.Page(self, inputtypes, contents, **kwargs))

    def add_list(self, inputtypes, **kwargs):
        return self._add_page(starsmashertools.bintools.page.List(self, inputtypes, **kwargs))

    def add_table(self, inputtypes, columns, **kwargs):
        return self._add_page(starsmashertools.bintools.page.Table(self, inputtypes, columns, **kwargs))

    # Go to the page specified by the identifier
    def navigate(self, identifier):
        if identifier not in self.pages.keys():
            raise Exception("No page with identifier '%s' exists" % str(identifier))
        self.pages[identifier].show()

    def run(self):
        self.navigate(self.pages[0].identifier)


