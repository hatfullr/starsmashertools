import inspect
import os
import functools

information = {}

def api(f):
    name = "%s.%s" % (f.__module__, f.__name__)
    information[name] = []
    
    for parameter in inspect.signature(f).parameters.values():
        if parameter.kind in [inspect.Parameter.POSITIONAL_OR_KEYWORD]:
            raise Exception("Argument '%s' of API function '%s' has ambiguous type '%s'. Please use the special arguments '/' and/or '*' to specify arguments as either '%s' or '%s'" % (parameter.name, name, str(inspect.Parameter.POSITIONAL_OR_KEYWORD), str(inspect.Parameter.POSITIONAL_ONLY), str(inspect.Parameter.KEYWORD_ONLY)))
        information[name] += [parameter]

    # Wrapper behavior. We need to use functools here or else we get into
    # trouble with other wrappers
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper
