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

    def parse(self, string, _types):
        import starsmashertools.lib.units
        import types, typing
        import starsmashertools.helpers.string
        import starsmashertools.preferences
        
        string = string.strip()
        if not isinstance(_types, (list, tuple)):
            _types = [_types]

        # Separate union types
        new_types = []
        for i, t in enumerate(_types):
            if not (isinstance(t, types.UnionType) or
                    typing.get_origin(t) is typing.Union):
                new_types += [t]
                continue
            for a in t.__args__:
                if a not in new_types:
                    new_types += [a]
        _types = new_types
        
        for _type in _types:
            if _type is types.NoneType:
                if string == 'None': return None
            if _type is bool:
                if string in ['true', 'false']:
                    return True if string == 'true' else False
            else:
                try:
                    return _type(string)
                except:
                    if _type is starsmashertools.lib.units.Unit:
                        all_labels = starsmashertools.lib.units.get_all_labels()
                        label_str = starsmashertools.helpers.string.list_to_string(
                            all_labels,
                            join = 'or',
                        )
                        raise InputManager.InvalidInputError("Input of type 'Unit' must have syntax 'Unit(value, label)', where 'label' is %s. To add more units, edit '%s'" % (label_str, starsmashertools.preferences.__file__))

        if None in _types: # Accept any generic input
            return starsmashertools.helpers.string.parse(string)

        if len(_types) <= 0:
            raise Exception("The list of input types must have a length greater than 0")
        type_string = starsmashertools.helpers.string.list_to_string(
            [t.__name__ for t in _types],
            join = 'or',
        )
        raise InputManager.InvalidInputError("Input must be of type " + type_string)

    def reset(self):
        starsmashertools.bintools.cli.CLI.refresh()
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

    
    def get(
            self,
            _type,
            prompt = None,
            halt = False,
            error_on_empty = True,
            **kwargs
    ):
        if prompt is None: prompt = self.prompt
        
        starsmashertools.bintools.cli.CLI.refresh()
        
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
                    if error_on_empty:
                        raise InputManager.InvalidInputError(repr(self.input))
                    else: return None
                return self.parse(self.input, _type)
            except InputManager.InvalidInputError as error:
                self.reset()
                starsmashertools.bintools.print_error(halt=halt)

        
class CursesInput(object):
    def __init__(self, window):
        self.window = window
        
