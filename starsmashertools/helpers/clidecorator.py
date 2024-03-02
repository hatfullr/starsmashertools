import functools
import starsmashertools
import starsmashertools.helpers.argumentenforcer

_exposed_programs = {}

def cli(program, *args, **kwargs):
    # Add this decorator to functions that you wish to tag as callable by CLI
    # programs in starsmashertools/bin. Pass as an argument to the decorator the
    # name of the program you wish to expose the function to, or a list of
    # programs.

    # Indicate to the function that it is being called from the CLI rather than
    # regular execution
    kwargs['cli'] = True

    # This structure allows decorator arguments
    def decorator(f):
        # Wrapper behavior. We need to use functools here or else we get into
        # trouble with other wrappers
        _exposed_programs[f] = {
            'program' : program,
            'args' : args,
            'kwargs' : kwargs,
        }
        @functools.wraps(f)
        def wrapper(*_args, **_kwargs):
            return f(*_args, **_kwargs)
        return wrapper
    return decorator

def clioptions(**kwargs):
    def decorator(f):
        # Functions can have arbitrary attributes
        f.clioptions = kwargs
        @functools.wraps(f)
        def wrapper(*_args, **_kwargs):
            return f(*_args, **_kwargs)
        return wrapper
    return decorator


def get_exposed_programs():
    import importlib
    
    decorators = starsmashertools._get_decorators()
    function = cli
    func_name = '%s.%s' % (function.__module__, function.__qualname__)
    
    # Fill _exposed_programs by importing
    for obj in decorators.get(func_name, []):
        module_name = obj['module']
        full_name = obj['full name']
        module = importlib.import_module(module_name)
        try:
            if obj['class'] is not None:
                f = None
                for _class in obj['class'].split('.'):
                    if f is None: f = getattr(module, _class)
                    else: f = getattr(f, _class)
                if f is None:
                    f = getattr(getattr(module, obj['class']), obj['short name'])
                else: f = getattr(f, obj['short name'])
            else:
                f = getattr(module, obj['short name'])
        except AttributeError:
            # This can happen if, for example, certain module members are only
            # defined inside a conditional statement.
            pass
        del module
    return _exposed_programs
    
def get_clioptions(function):
    options = getattr(function, 'clioptions', {})
    return validate_options(**options)

@starsmashertools.helpers.argumentenforcer.enforcetypes
def validate_options(
        confirm : str | type(None) = None,
):
    return locals()
