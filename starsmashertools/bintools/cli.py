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
import starsmashertools.helpers.string
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
        
    def add_page(
            self,
            inputtypes,
            contents,
            kind : type = starsmashertools.bintools.page.Page,
            **kwargs
    ):
        return self._add_page(kind(self, inputtypes, contents, **kwargs))

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
            quit()

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
        CLI.stdscr.redrawwin()
        height, width = CLI.get_height_and_width()
        CLI.pad.refresh(
            CLI.position[1],
            CLI.position[0],
            0, 0, height - 1, width - 1,
        )

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


