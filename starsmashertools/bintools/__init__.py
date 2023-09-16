import os
import traceback
import sys

# If 'error' is None, uses the last-raised error.
def print_error(error=None, halt=False):
    import starsmashertools.bintools.cli
    if error is not None:
        if halt: raise error
        else:
            try:
                raise error
            except Exception as e:
                string = traceback.format_exc().split('\n')[-2]
                starsmashertools.bintools.cli.CLI.write(string)
    else:
        if halt: raise
        string = traceback.format_exc().split('\n')[-2]
        starsmashertools.bintools.cli.CLI.write(string)


class Style(object):
    # 'True' values are for terminal output
    # 'False' values are for redirected output
    # If not a dict, then the value is used no matter what
    # Styles
    text = {
        'normal'    : {
            True  : '\033[0m',
            False : '',
        },
        'bold'      : {
            True  : '\033[1m',
            False : '',
        },
    }

    formatting = {
        'header' : '',
        'footer' : '',
    }

    colors = {
        'green'     : {
            True  : '\033[92m',
            False : '',
        },
        'red'       : {
            True  : '\033[91m',
            False : '',
        },
    }

    backgrounds = {
        'white'     : {
            True  : '\033[107m',
            False : '',
        },
        'light gray' : {
            True  : '\033[47m',
            False : '',
        },
    }

    characters = {
        # Special characters
        'newline' : os.linesep,
        'checkmark' : u'\N{check mark}',
        'table vertical' : "│",
        'table horizontal' : "─",
        'table top left' : "┌",
        'table top right' : "┐",
        'table bottom left' : "└",
        'table bottom right' : "┘",
        'table top separator' : "┬",
        'table bottom separator' : "┴",
        'table left separator' : "├",
        'table right separator' : "┤",
        'table full separator' : "┼",
    }

    @staticmethod
    def get_category(category):
        if not hasattr(Style, category) or category.startswith("_"):
            raise ValueError("No category named '%s'" % str(category))
        return getattr(Style, category)
    
    @staticmethod
    def get(category, name):
        cat = Style.get_category(category)
        if name not in cat.keys():
            raise KeyError("No name '%s' in category '%s'" % (str(name), str(category)))
        if isinstance(cat[name], dict):
            return cat[name][sys.stdout.isatty()]
        return cat[name]

    @staticmethod
    def set(category, name, value):
        cat = Style.get_category(category)
        cat[name] = value
        setattr(Style, category, cat)

    @staticmethod
    def bold(string):
        return Style.get('text', 'bold') + string + Style.get('text', 'normal')

    @staticmethod
    def color(string, _color):
        return Style.get('colors', _color) + string + Style.get('text', 'normal')

    @staticmethod
    def background(string, _color):
        return Style.get('backgrounds', _color) + string + Style.get('text', 'normal')

    @staticmethod
    def clean(string):
        values = []
        for key in Style.text.keys():
            value = Style.get('text', key)
            if value not in values: values += [value]

        for key in Style.colors.keys():
            value = Style.get('colors', key)
            if value not in values: values += [value]

        for key in Style.backgrounds.keys():
            value = Style.get('backgrounds', key)
            if value not in values: values += [value]

        for value in values:
            string = string.replace(value, '')
        return string

    # Given a string littered with terminal magic, return a formatter that does
    # things correctly.
    @staticmethod
    def get_formatter(string, format):
        clean_string = Style.clean(string)
        idx = string.index(clean_string)
        formatter = string[:idx] + format + string[idx + len(clean_string):]
        return formatter, clean_string
