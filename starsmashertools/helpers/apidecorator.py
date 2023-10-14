import inspect
import functools

#information = {}

def api(f):
    #name = "%s.%s" % (f.__module__, f.__qualname__)
    #information[name] = []
    
    #for parameter in inspect.signature(f).parameters.values():
    #    information[name] += [parameter]

    # Wrapper behavior. We need to use functools here or else we get into
    # trouble with other wrappers
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper
