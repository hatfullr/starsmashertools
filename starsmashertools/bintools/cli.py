# A CLI is a Command Line Interface that manages inputs and Page navigation.
import sys
import os
import argparse
import starsmashertools.bintools.page
import starsmashertools.lib.simulation
import starsmashertools.helpers.argumentenforcer
import collections
import curses
import copy
import textwrap
import signal

class CLI(object):
    instance = None
    stdscr = None
    pad = None
    position = [None, None]
    overlay = None
    
    def __init__(
            self,
            name : str,
            description : str,
            require_valid_directory : bool = True,
    ):
        import starsmashertools.bintools.inputmanager
        
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
    def get_width():
        return CLI.get_height_and_width()[1]

    @staticmethod
    def get_height():
        return CLI.get_height_and_width()[0]

    @staticmethod
    def get_height_and_width():
        return CLI.stdscr.getmaxyx()

    @staticmethod
    def write(*args, xy=None, move=None, move_relative=False, end='\n', flush=True):
        import starsmashertools.bintools
        import starsmashertools.helpers.file
        import starsmashertools.helpers.path
        
        string = ' '.join(args) + end

        # Current cursor position
        y, x = CLI.pad.getyx()

        if xy is not None:
            newx, newy = xy
            maxy, maxx = CLI.get_height_and_width()
            while newy > maxy: newy -= maxy
            while newx > maxx: newx -= maxx
            while newy < 0: newy += maxy
            while newx < 0: newx += maxx
            CLI.move(newy, newx)

        string = CLI.prepare_string(string)
        ret = CLI.writestr(string)
        
        if move is None:
            y += string.count('\n')
            x = 0
            if '\n' in string or end != '\n':
                x = len(string.split('\n')[-1])
        else:
            if move[0] is not None:
                if move_relative: x += move[0]
                else: x = move[0]
            if move[1] is not None:
                if move_relative: y += move[1]
                else: y = move[1]
        
        height, width = CLI.get_height_and_width()
        CLI.move(y, x)
        if flush: CLI.refresh()
        
        filename = CLI.instance.args['output']
        if filename is not None:
            mode = 'w'
            if starsmashertools.helpers.path.isfile(filename): mode = 'a'
            content = ' '.join(args)
            content = starsmashertools.bintools.Style.clean(content)
            with starsmashertools.helpers.file.open(filename, mode) as f:
                f.write(content)
        return ret

    @staticmethod
    def prepare_string(string):
        """
        Before calling writestr, the string should be prepared first by wrapping
        text for printing in the terminal.
        """
        import starsmashertools.bintools

        width = CLI.get_width()
        codes = starsmashertools.bintools.ANSI.get_all()
        head = 0
        i = 0
        while i < len(string):
            # Skip ANSI codes
            for code in codes:
                if string[i:i + len(code)] == code:
                    i += len(code)
                    break
            else: # No codes found
                # On a newline, reset the head
                if string[i] == "\n":
                    head = 0
                else: # Not a newline
                    head += 1 # Advance the head
                    if head >= width: # It's time to wrap
                        string = string[:i] + "\n" + string[i:]
                        head = 0 # Reset the head
                i += 1 # Advance the counter
        return string
        
        

    @staticmethod
    def writestr(text, _codes=[]):
        """
        We parse out the styles for proper writing
        """
        import time
        import curses
        import starsmashertools.bintools

        ANSI = starsmashertools.bintools.ANSI
        mapping = {
            ANSI.BOLD : [curses.A_BOLD],
            ANSI.GREENFG : [curses.color_pair(1)],
            ANSI.REDFG : [curses.color_pair(2)],
            ANSI.WHITEBG : [curses.color_pair(3)],
            ANSI.LIGHTGRAYBG : [curses.color_pair(4)],
        }

        codes = ANSI.get_all()
        
        # print stuff that doesn't have any codes
        lowest_idx = len(text)
        lowest_code = None
        for code in codes + [ANSI.NORMAL]:
            if code not in text: continue
            idx = text.index(code)
            if idx < lowest_idx:
                lowest_idx = idx
                lowest_code = code

        # If no codes
        if lowest_code is None:
            CLI.pad.addstr(text)
            for code in _codes:
                CLI.pad.attroff(code)
            return

        # Print the first stuff that has no codes
        CLI.pad.addstr(text[:lowest_idx], *_codes)

        # Turn on the code
        if lowest_code != ANSI.NORMAL:
            for code in mapping[lowest_code]:
                if code not in _codes:
                    _codes += [code]
                    CLI.pad.attron(code)
        else:
            # Turn off all codes
            for code in _codes:
                CLI.pad.attroff(code)
            _codes = []

        # Chop the text
        text = copy.deepcopy(text)[lowest_idx + len(lowest_code):]

        # Repeat the process
        CLI.writestr(text, _codes=_codes)
        
    
    def set_mainmenu(self, page):
        self._mainmenu = page

    def reset(self):
        if self._mainmenu is not None:
            self.navigate(self._mainmenu.identifier)
        else: self._quit()

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
        
    def add_page(
            self,
            inputtypes,
            contents,
            kind : type = starsmashertools.bintools.page.Page,
            **kwargs
    ):
        return self._add_page(kind(self, inputtypes, contents, **kwargs))

    def add_function_page(self, function, **kwargs):
        return self._add_page(starsmashertools.bintools.page.FunctionPage(self, function, **kwargs))

    def add_argument_page(self, *args, **kwargs):
        return self._add_page(starsmashertools.bintools.page.ArgumentPage(self, *args, **kwargs))
    
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
        self.clear()
        self.pages[identifier].show()

    def clear(self):
        CLI.pad.clear()
        CLI.refresh()
        
    def run(self):
        def main(stdscr):
            # This puts us in shell mode, which the rest of our code expects
            curses.use_default_colors()
            
            curses.init_pair(1, curses.COLOR_GREEN, -1) # green
            curses.init_pair(2, curses.COLOR_RED, -1) # red
            curses.init_pair(3, -1, curses.COLOR_WHITE)
            curses.init_pair(4, -1, curses.COLOR_YELLOW) # eh
            
            stdscr.refresh()

            CLI.stdscr = stdscr

            height, width = stdscr.getmaxyx()
            
            CLI.pad = curses.newpad(1000, 1000)
            CLI.position = [0, 0]
            CLI.refresh()
            
            self.navigate(self.pages[0].identifier)

        def on_resize(*args, **kwargs):
            CLI.stdscr.redrawwin()
            y, x = CLI.pad.getyx()
            columns, lines = os.get_terminal_size()
            height, width = CLI.get_height_and_width()
            CLI.stdscr.resize(lines, columns)
            
            if lines != height and (y >= lines - 1 or CLI.position[1] < 0):
                CLI.scroll((0, lines - height))
            
        signal.signal(signal.SIGWINCH, on_resize)
        try:
            curses.wrapper(main)
        except KeyboardInterrupt:
            self._quit()

    @staticmethod
    def scroll(offset, refresh=True):
        """
        Scroll the content by the offset amount. Positive goes up, negative goes
        down.
        """
        CLI.position[1] += offset[1]
        CLI.position[0] += offset[0]
        
        if refresh: CLI.refresh()

    @staticmethod
    def move(y, x, *args, **kwargs):
        height, width = CLI.get_height_and_width()
        if y > height:
            
            #CLI.scroll(y - height)
            #CLI.stdscr.resize(100, width)
            #CLI.refresh()
            CLI.scroll((0, y - height), refresh=False)
            
            #y -= height
        CLI.pad.move(y, x, *args, **kwargs)
        
    @staticmethod
    def refresh():
        if CLI.pad is None:
            raise Exception("Cannot refresh the screen because the screen has not yet been created")

        height, width = CLI.get_height_and_width()
        CLI.stdscr.redrawwin()
        CLI.pad.refresh(
            CLI.position[1],
            CLI.position[0],
            0, 0, height - 1, width - 1,
        )

    @staticmethod
    def _quit():
        try:
            import matplotlib.pyplot as plt
            # Destroy the window of each opened figure manually
            for fignum in plt.get_fignums():
                plt.figure(fignum)
                try:
                    plt.gcf().canvas.get_tk_widget().destroy()
                except: pass
        except: pass
        quit()
    

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def add_simulation(
            self,
            simulation : str | starsmashertools.lib.simulation.Simulation,
    ):
        import starsmashertools.helpers.path
        import starsmashertools
        
        if isinstance(simulation, str):
            simulation = starsmashertools.helpers.path.realpath(simulation)
            simulation = starsmashertools.get_simulation(simulation)
        self.simulations += [simulation]

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def remove_simulation(
            self,
            simulation : int | str | starsmashertools.lib.simulation.Simulation,
    ):
        import starsmashertools
        
        if isinstance(simulation, str):
            simulation = starsmashertools.get_simulation(simulation)
        elif isinstance(simulation, int):
            simulation = self.simulations[simulation]
        self.simulations.remove(simulation)










class HookedCLI(CLI, object):
    """
    This is a CLI which uses the @cli decorator to expose functions to the user.
    """

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def __init__(self, *args, **kwargs):
        import starsmashertools.helpers.clidecorator

        super(HookedCLI, self).__init__(*args, **kwargs)

        self._object = None

        self.exposed_programs = starsmashertools.helpers.clidecorator.get_exposed_programs()
        
        self.mainmenu = self.add_list(
            [int, str],
            bullet = '%5d)',
        )
        
        self.set_mainmenu(self.mainmenu)

    def run(self, *args, **kwargs):
        import starsmashertools.bintools
        
        available_functions = self.get_available_functions()
        names = self.get_function_names(available_functions)

        newline = starsmashertools.bintools.Style.get('characters', 'newline')

        for name in names:
            self.mainmenu.add(name)
        
        for i, (name, function) in enumerate(zip(names, available_functions)):
            page = self.add_function_page(
                function,
                bullet = '%5d)',
                header = ("Select an argument to modify for '%s'. When you are finished, press enter to execute" + newline) % name,
                footer = newline,
                back = self.mainmenu,
                _quit = True,
            )
            self.mainmenu.connect(page, [i, name])

        return super(HookedCLI, self).run(*args, **kwargs)

    def set_object(self, _object):
        self._object = _object
        
    def set_mainmenu_properties(self, **properties):
        for key, val in properties.items():
            setattr(self.mainmenu, key, val)

    def get_function_class_name(self, function):
        if '.' in function.__qualname__:
            return ".".join(function.__qualname__.split(".")[:-1])
        return None
    
    def get_available_functions(self):
        available_functions = []
        for key, val in self.exposed_programs.items():
            if val['program'] == self.name:
                available_functions += [key]
        basenames = [a.__name__ for a in self._object.__class__.__bases__]
        keep = []
        for function in available_functions:
            cls = self.get_function_class_name(function)
            if cls is None: continue
            if cls == self._object.__class__.__name__ or cls in basenames:
                keep += [function]
        return keep

    def get_function_names(self, functions):
        names = []
        for function in functions:
            if '.' in function.__qualname__:
                names += [function.__qualname__.split('.')[-1]]
            else: names += [function.__qualname__]
        return names

