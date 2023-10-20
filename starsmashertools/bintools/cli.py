# A CLI is a Command Line Interface that manages inputs and Page navigation.
import sys
import os
import argparse
import starsmashertools.bintools.page
import starsmashertools.bintools.inputmanager
import starsmashertools
import starsmashertools.lib.simulation
import starsmashertools.helpers.argumentenforcer
import starsmashertools.helpers.path
import starsmashertools.helpers.file
import collections


class CLI(object):
    instance = None
    
    def __init__(
            self,
            name : str,
            description : str,
            require_valid_directory : bool = True,
    ):
        if CLI.instance is not None:
            raise Exception("Only one CLI instance is permitted per process")
        
        self.name = name
        self.description = description
        
        # Store the current working directory for working with simulations
        self.directory = os.getcwd()

        # If we are in a simulation directory, record as such
        self.simulations = []
        try:
            self.add_simulation(self.directory)
        except starsmashertools.lib.simulation.Simulation.InvalidDirectoryError:
            if require_valid_directory: raise
            pass

        self.page = None
        
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
            if starsmashertools.helpers.path.isfile(filename): mode = 'a'
            content = ' '.join(args)
            content = starsmashertools.bintools.Style.clean(content)
            with starsmashertools.helpers.file.open(filename, mode) as f:
                f.write(content)
        return ret
    
    def set_mainmenu(self, page):
        self._mainmenu = page

    def reset(self):
        if self._mainmenu is not None:
            self.navigate(self._mainmenu.identifier)
        else: quit()

    # str to use as identifier
    def remove_page(
            self,
            page : str | starsmashertools.bintools.page.Page,
    ):
        if not isinstance(page, str): page = page.identifier
        if page not in self.pages.keys(): raise KeyError("No page with identifier '%s' found" % str(page))
        removed = self.pages.pop(page)

        # Fully disconnect the page
        for idx, p in self.pages.items():
            if removed in p.connections.keys():
                p.disconnect(removed)
        
        # We also need to remove all this page's connections recursively
        keys = list(removed.connections.keys())
        for connection in keys:
            self.remove_page(connection)
        
    def add_page(self, inputtypes, contents, **kwargs):
        return self._add_page(starsmashertools.bintools.page.Page(self, inputtypes, contents, **kwargs))

    def add_confirmation_page(self, **kwargs):
        return self._add_page(starsmashertools.bintools.page.ConfirmationPage(self, **kwargs))
    
    def add_list(self, inputtypes, **kwargs):
        return self._add_page(starsmashertools.bintools.page.List(self, inputtypes, **kwargs))

    def add_table(self, inputtypes, columns, **kwargs):
        return self._add_page(starsmashertools.bintools.page.Table(self, inputtypes, columns, **kwargs))

    def navigate(self, identifier : str | starsmashertools.bintools.page.Page):
        """
        Go to the given page
        """
        if (isinstance(identifier, starsmashertools.bintools.page.Page) or
            issubclass(identifier.__class__, starsmashertools.bintools.page.Page)):
            identifier = identifier.identifier
        
        if identifier not in self.pages.keys():
            raise Exception("No page with identifier '%s' exists" % str(identifier))
        self.pages[identifier].show()
        
    def run(self):
        self.navigate(self.pages[0].identifier)

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def add_simulation(
            self,
            simulation : str | starsmashertools.lib.simulation.Simulation,
    ):
        if isinstance(simulation, str):
            simulation = starsmashertools.helpers.path.realpath(simulation)
            simulation = starsmashertools.get_simulation(simulation)
        self.simulations += [simulation]

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def remove_simulation(
            self,
            simulation : int | str | starsmashertools.lib.simulation.Simulation,
    ):
        if isinstance(simulation, str):
            simulation = starsmashertools.get_simulation(simulation)
        elif isinstance(simulation, int):
            simulation = self.simulations[simulation]
        self.simulations.remove(simulation)
            

        
