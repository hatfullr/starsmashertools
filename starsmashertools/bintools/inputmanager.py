# This class handles the user input and parsing it.
import os
import curses
import readline # Allows for fancy input editing
import sys
import textwrap
import typing
import types

union_types = (
    types.UnionType,
    # This is true whenever we have a keyword which can be some type but also
    # NoneType.
    typing._UnionGenericAlias,
)

def _get_types(_types):
    # Given a list of Python types from a function annotation, parse out each
    # individual type.
    new_types = []
    for i, t in enumerate(_types):
        if not (isinstance(t, union_types) or
                typing.get_origin(t) is typing.Union):
            new_types += [t]
            continue
        for a in t.__args__:
            if a not in new_types:
                new_types += [a]
    # Sort the types a bit for smoother parsing attempts
    my_types = []
    for t in [bool, int, float, str]:
        if t in new_types: my_types += [t]
    for t in new_types:
        if t not in my_types: my_types += [t]
    return my_types
    

class InputManager(object):
    class InvalidInputError(Exception):
        __module__ = Exception.__module__

    class NullValue(object): pass
    
    def __init__(self, prompt=': '):
        super(InputManager, self).__init__()
        self.prompt = prompt
        self.input = None

        try: # Load the previous session
            self.session = Session.load()
        except FileNotFoundError: # Start a new session
            self.session = Session()

    def parse(self, string, _types):
        import starsmashertools.lib.units
        import starsmashertools.helpers.string
        import starsmashertools.preferences
        import numpy as np
        
        string = string.strip()
        if not isinstance(_types, (list, tuple)):
            _types = [_types]

        for _type in _get_types(_types):
            if _type is types.NoneType or _type is type(None):
                if string == 'None': return None
            elif _type is bool:
                if string.lower() in ['true', 'false']:
                    return True if string.lower() == 'true' else False
            elif _type in [list, tuple, np.ndarray]:
                # Easiest to just try and eval the string
                val = None
                try:
                    val = eval(string)
                except SyntaxError as e:
                    raise InputManager.InvalidInputError(str(e))
                if isinstance(val, _type): return val
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

        names = []
        for t in _types:
            if isinstance(t, union_types): names += [_t.__name__ for _t in t.__args__]
            elif t is None: names += ['None']
            else: names += [t.__name__]
        type_string = starsmashertools.helpers.string.list_to_string(
            names,
            join = 'or',
        )
        raise InputManager.InvalidInputError("Input must be of type " + type_string)

    def reset(self):
        import starsmashertools.bintools.cli
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
            prompt : str | type(None) = None,
            halt : bool = False,
            error_on_empty : bool = True,
            refresh : bool = True,
            **kwargs
    ):
        import starsmashertools.bintools.cli
        import starsmashertools.bintools
        
        if prompt is None: prompt = self.prompt
        
        if refresh:
            starsmashertools.bintools.cli.CLI.refresh()
            #print("refreshed")
            #import time
            #time.sleep(1)
        
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
                    else: return InputManager.NullValue()
                return self.parse(self.input, _type)
            except InputManager.InvalidInputError as error:
                self.reset()
                starsmashertools.bintools.print_error(halt=halt)

        
class CursesInput(object):
    def __init__(self, window):
        self.window = window
        


class Session(object):
    """ Store information about the CLI session. Whenever an InputManager has 
    method :py:func:`~.InputManager.get` called, we store the user's input
    whenever we ask them for a keyword (has a default value). The next time we
    ask them for that value, the new default will be their previous response. 
    """

    # These are function arguments we don't ever want the user to edit.
    ignore_arguments = [
        'self',
        'cli',
        'kwargs',
    ]

    @staticmethod
    def get_function_path(function : typing.Callable):
        if function.__module__ == 'builtins':
            return function.__qualname__
        return function.__module__ + '.' + function.__qualname__

    class Value(object):
        """ These are stored in a Session. """
        def __init__(self, function : typing.Callable | type(None) = None):
            import inspect
            import os

            self.function_path = None
            self.positional = None
            self.keyword = None
            self._class = None
            self.module = None
            self.name = None
            
            if function is None: return

            self.module = function.__module__
            self.function_path = Session.get_function_path(function)
            self.name = function.__name__
            if '.'+self.name in function.__qualname__:
                self._class = function.__qualname__.replace('.'+self.name, '')
            
            try:
                src = inspect.getsourcefile(function)
            except TypeError as e: # Fails if it's a builtin function (?)
                raise Exception("Cannot save default parameters for a builtin function: '%s'" % self.function_path) from e
            if src is None:
                raise FileNotFoundError("Failed to find the source file from where the given function was defined: '%s'" % self.function_path)
            
            self.positional, self.keyword = Session.Value.get_arguments(function)

        @staticmethod
        def get_arguments(function : typing.Callable):
            import inspect
            type_hints = typing.get_type_hints(function)
            positional = {}
            keyword = {}
            for name, parameter in inspect.signature(function).parameters.items():
                if name in Session.ignore_arguments: continue
                param_type_hints = type_hints.get(name, None)

                if isinstance(param_type_hints, type):
                    param_type_hints = param_type_hints.__name__

                elif param_type_hints is None: param_type_hints = 'None'
                else:
                    if isinstance(param_type_hints, union_types):
                        param_type_hints = [t.__name__ for t in param_type_hints.__args__]
                    else:
                        for i, t in enumerate(param_type_hints):
                            if t is None:
                                param_type_hints[i] = 'None'
                            else:
                                param_type_hints[i] = t.__name__
                
                if parameter.default == inspect._empty: # Positional argument
                    positional[name] = {
                        'type hints' : param_type_hints,
                    }
                else: # Keyword argument
                    keyword[name] = {
                        'type hints' : param_type_hints,
                        'default' : parameter.default,
                    }
            return positional, keyword


        def to_json(self):
            return {
                'function path' : self.function_path,
                'class' : self._class,
                'module' : self.module,
                'name' : self.name,
                'positional' : self.positional,
                'keyword' : self.keyword,
            }
        
        @staticmethod
        def from_json(obj):
            ret = Session.Value()
            ret.function_path = obj['function path']
            ret.positional = obj['positional']
            ret.keyword = obj['keyword']
            ret._class = obj['class']
            ret.module = obj['module']
            ret.name = obj['name']
            return ret

        def __eq__(self, other):
            if not isinstance(other, Session.Value):
                raise TypeError("Cannot compare type 'Session.Value' to '%s'. Can only compare 'Session.Value' to 'Session.Value'" % type(other).__name__)
            if not self.function_path == other.function_path: return False
            for c1, c2 in zip([self.positional, self.keyword], [other.positional, other.keyword]):
                for key, val in c1.items():
                    if key not in c2.keys(): return False
                    if val != c2[key]: return False
                for key, val in c2.items():
                    if key not in c1.keys(): return False
                    if val != c1[key]: return False
            return True

    def __init__(self, values = {}):
        self.values = values

    def __contains__(self, value):
        if not isinstance(value, Session.Value): return False
        for key, val in self.values.items():
            if val == value: return True
        return False
    
    @staticmethod
    def get_filename():
        # This will save session files in the current working directory
        return '.sstools_session.json'

    @staticmethod
    def update_to_current(session):
        """ Remove non-exiting keys and functions. """
        import importlib
        new_session = Session(session.values)
        for function_name, val in session.values.items():
            found = importlib.util.find_spec(val.module)
            if found is None: continue # module not found
            module = importlib.import_module(val.module)
            if val._class is not None:
                if not hasattr(module, val._class): continue # class not found
                parent = getattr(module, val._class)
                if not hasattr(parent, val.name): continue # function name not found
            else:
                if not hasattr(module, val.name): continue # module method not found
                parent = module

            method = getattr(parent, val.name)
            new_val = Session.Value(function = method)
            for key, v in val.positional.items():
                if key not in new_val.positional.keys(): continue
                new_val.positional[key] = v
            for key, v in val.keyword.items():
                if key not in new_val.keyword.keys(): continue
                new_val.keyword[key] = v
            del module

            new_session.values[function_name] = new_val
        return new_session

    @staticmethod
    def load():
        import starsmashertools.helpers.jsonfile
        import importlib
        
        obj = starsmashertools.helpers.jsonfile.load(Session.get_filename())
        for key, val in obj.items():
            obj[key] = Session.Value.from_json(val)

        session = Session.update_to_current(Session(obj))
        return session

    def save(self):
        import starsmashertools.helpers.jsonfile
        
        obj = {}
        for key, val in self.values.items():
            obj[key] = val.to_json()
        
        starsmashertools.helpers.jsonfile.save(Session.get_filename(), obj)

    def get(self, function : typing.Callable, name : str):
        import inspect
        function_path = Session.get_function_path(function)
        default = inspect.signature(function).parameters[name].default
        if function_path in self.values.keys():
            return self.values[function_path].positional.get(
                name,
                self.values[function_path].keyword.get(
                    name,
                    {},
                ).get('default', default)
            )
        return default
    
    def set(self, function : typing.Callable, name : str, value):
        function_path = Session.get_function_path(function)
        
        if function_path in self.values.keys():
            if name in self.values[function_path].keyword.keys():
                # Set the default value
                self.values[function_path].keyword[name]['default'] = value
            else:
                # This name is not yet a known keyword. This can happen when,
                # e.g., a user switches their starsmashertools version.

                # Obtain the default values
                positional, keyword = Session.Value.get_arguments(function)

                if name not in keyword.keys():
                    # Must be in positionals
                    positional[name] = value
                    #raise KeyError("The given name '%s' is not a valid keyword name for function '%s'" % (name, function_path))
                    self.values[function_path].positional = positional
                else:
                    # Set the default value
                    keyword[name]['default'] = value
                    self.values[function_path].keyword = keyword
        else: # This is the first time seeing this function
            # Create a set of default values for this function
            self.values[function_path] = Session.Value(function = function)
            # Retry now that we know about the function
            self.set(function, name, value)



