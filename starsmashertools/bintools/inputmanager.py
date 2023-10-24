# This class handles the user input and parsing it.
import starsmashertools.bintools
import starsmashertools.bintools.cli
import os
import curses
import readline # Allows for fancy input editing
import sys
import textwrap

class InputManager(object):

    class InvalidInputError(Exception):
        __module__ = Exception.__module__
    
    def __init__(self, prompt=': '):
        super(InputManager, self).__init__()
        self.prompt = prompt
        self.input = None

    def parse(self, string, types):
        string = string.strip()
        if not isinstance(types, (list, tuple)):
            types = [types]
        
        for _type in types:
            if _type is bool:
                if string in ['true', 'false']:
                    return True if string == 'true' else False
            else:
                try:
                    return _type(string)
                except:
                    pass
        
        if len(types) == 1: type_string = "'%s'" % types[0].__name__
        elif len(types) == 2: type_string = "'%s' or '%s'" % tuple([t.__name__ for t in types])
        elif len(types) > 2:
            type_string = ", ".join(["'%s'" % t.__name__ for t in types[:-2]])
            type_string += ", '%s' or '%s'" % (types[-2].__name__, types[-1].__name__)
        else:
            raise Exception("The list of input types must have a length greater than 0")
        raise InputManager.InvalidInputError("Input must be of type " + type_string)

    def reset(self):
        starsmashertools.bintools.cli.CLI.stdscr.refresh()
        width = starsmashertools.bintools.cli.CLI.get_width()
        
        # This carefully returns the Python cursor to where it started before
        # the call to input(). The curses cursor remains unmoved.
        wrapped = textwrap.wrap(
            self.input,
            width=width,
            drop_whitespace=False,
            replace_whitespace=False,
        )

        if self.input: 
            nlines = len(wrapped)
        else: # Empty input means Enter button was pressed only. Enter button
              # creates a new line
            nlines = 1
        
        print("\033[F"*nlines, end='')
        print(" "*(width * (nlines+1)), end='') # Clear the text
        print("\033[F"*nlines, end='')
        sys.stdout.flush()

        starsmashertools.bintools.cli.CLI.stdscr.clrtobot()

    
    def get(self, _type, prompt=None, halt=False, **kwargs):
        if prompt is None: prompt = self.prompt
        
        starsmashertools.bintools.cli.CLI.stdscr.refresh()
        
        while True:
            curses.reset_shell_mode()
            self.input = input(prompt)
            sys.stdout.flush()
            curses.reset_prog_mode()
        
            # Write to output file if needed
            filename = starsmashertools.bintools.cli.CLI.instance.args['output']
            if filename is not None:
                mode = 'w'
                if os.path.isfile(filename): mode = 'a'
                newline = starsmashertools.bintools.Style.get('characters', 'newline')
                content = newline + prompt + self.input + newline
                content = starsmashertools.bintools.Style.clean(content)
                with open(filename, mode) as f:
                    f.write(content)
        
            try:
                if len(self.input) == 0:
                    raise InputManager.InvalidInputError(repr(self.input))
                return self.parse(self.input, _type)
            except InputManager.InvalidInputError as error:
                self.reset()
                starsmashertools.bintools.print_error(halt=halt)

        
class CursesInput(object):
    def __init__(self, window):
        self.window = window
        
