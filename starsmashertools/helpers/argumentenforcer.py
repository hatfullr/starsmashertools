import typing
import types
import inspect

def enforcetypes(f):
    def type_checker(*args, **kwargs):
        hints = typing.get_type_hints(f)
        all_args = kwargs.copy()
        all_args.update(dict(zip(f.__code__.co_varnames, args)))
        for key in all_args:
            if key in hints:
                if isinstance(hints[key], types.UnionType):
                    if (not isinstance(all_args[key], hints[key].__args__) and
                        not any([issubclass(all_args[key].__class__, h) for h in hints[key].__args__])):
                        typelist = []
                        for a in hints[key].__args__:
                            module = a.__module__
                            if module == 'builtins':
                                typelist += ["'"+a.__qualname__+"'"]
                            else:
                                typelist += ["'"+module+"."+a.__qualname__+"'"]
                        _types = ", ".join(typelist[:-1])
                        _types += " or %s" % typelist[-1]
                        raise TypeError("Argument '{}' must be one of types {}, not '{}'".format(key, _types, type(all_args[key]).__name__))
                else:
                    if (type(all_args[key]) != hints[key] and
                        not issubclass(all_args[key].__class__, hints[key])):
                        raise TypeError("Argument '{}' must be type '{}', not '{}'".format(key, hints[key].__name__, type(all_args[key]).__name__))
        return f(*args, **kwargs)
    return type_checker


def enforcevalues(obj):
    _locals = inspect.currentframe().f_back.f_locals
    for var_name, var_val in obj.items():
        if var_name not in _locals.keys():
            raise KeyError("There is no key '%s' in this function, and so its argument value cannot be enforced." % var_name)
        if isinstance(var_val, str) or not hasattr(var_val, '__iter__'):
            raise TypeError("You must specify an iterable object to check variable values against. Received type '%s'" % type(var_val).__name__)
        
        if _locals[var_name] not in var_val:
            accepted_vals = ["'%s'" % str(v) for v in var_val]
            accepted_str = ", ".join(accepted_vals[:-1])
            accepted_str += " or "+accepted_vals[-1]
            raise ValueError("Argument '%s' must be one of %s, not '%s'" % (var_name, accepted_str, str(_locals[var_name])))



