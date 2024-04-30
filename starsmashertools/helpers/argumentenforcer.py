import typing
import types
import inspect
import functools
import numpy as np

def enforcetypes(f):
    """
    If this function is called, the input must be of type :py:class:`dict`.
    Otherwise, this function is to be used as a wrapper.
    """
    if isinstance(f, dict): return _enforcetypes(f)

    # Wrapper behavior. We need to use functools here or else we get into
    # trouble with other wrappers.
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        _check_types(f, args, kwargs)
        return f(*args, **kwargs)
    return wrapper


@enforcetypes
def _enforcetypes(obj : dict):
    variables, frame = _get_variables_in_context_from_dict(obj.keys())
    for name, types in obj.items():
        value = variables[name]
        if not isinstance(value, tuple(types)):
            raise ArgumentTypeError(
                given_name     = name,
                given_type     = type(value),
                expected_types = types,
                frame          = frame,
            )

def _get_types_from_annotation(annotation):
    if hasattr(annotation, '__args__'): # types.UnionType or typing.UnionType
        return annotation.__args__
    return [annotation]

def _check_types(f, args, kwargs):
    signature = inspect.signature(f)
    mismatched_types = []

    parameters = signature.parameters
    positional = [p for _,p in zip(args, parameters.values())]
    
    for i, parameter in enumerate(parameters.values()):
        annotation = parameter.annotation

        # Don't check empty annotations
        if annotation is inspect._empty: continue

        if parameter in positional: arg = args[i]
        else:
            if parameter.name not in kwargs.keys(): continue
            arg = kwargs[parameter.name]

        types = _get_types_from_annotation(annotation)

        # If the given type is a NumPy generic, convert it to a Python type
        if isinstance(arg, np.generic):
            if type(arg) in types: continue
            compare = arg.item()
        else: compare = arg
        
        if type(compare) in types: continue

        
        # Check for subclasses only if compare is not a builtin type. The reason
        # is because for some god-forsaken reason, bool is a subclass of int in
        # Python...
        if compare.__class__.__module__ not in ['builtins', '__builtins__']:
            # Check for subclasses
            try:
                if issubclass(compare.__class__, tuple(types)): continue
            except Exception as e:
                print(types)
                raise(e)

        if typing.Callable in types:
            if callable(arg): continue

        raise ArgumentTypeError(
            given_name = parameter.name,
            given_type = type(compare),
            expected_types = types,
        )



def enforcevalues(obj):
    # Make sure the inputs are correct
    for key, var_val in obj.items():
        # if is non-str iterable
        if not hasattr(var_val, '__iter__') or isinstance(var_val, str):
            raise TypeError("You must specify a non-str iterable object to check variable values against. Received '%s'" % str(var_val))
    
    variables, frame = _get_variables_in_context_from_dict(obj.keys())
    for var_name, var_val in obj.items():
        value = variables[var_name]
        if value not in var_val:
            raise ArgumentValueError(
                given_name      = var_name,
                given_value     = str(value),
                expected_values = var_val,
                frame           = frame,
            )


# Returns a dictionary with keys equal to the variable names and values equal to
# the variable values, where the variables match the keys from the input list
def _get_variables_in_context_from_dict(name_list):
    found = []
    frame = inspect.currentframe()
    while frame is not None:
        # Don't search any frames in this file
        if frame.f_code.co_filename == __file__:
            frame = frame.f_back
            continue
        
        found += list(frame.f_locals.keys())
        for name in name_list:
            # If a name isn't in this list, then it isn't the frame we are
            # looking for
            if name not in frame.f_locals.keys(): break
        # Found the right frame that has all the given variables (probably)
        else: break
        
        frame = frame.f_back
    else:
        import starsmashertools.helpers.string
        list_str = starsmashertools.helpers.string.list_to_string(
            sorted(list(name_list)),
        )
        found_str = starsmashertools.helpers.string.list_to_string(
            sorted(list(set(found))),
        )
        raise Exception("Failed to find the context that contains all the given variable names: %s.\nFound: %s" % (list_str, found_str))

    _locals = frame.f_locals
    keys = _locals.keys()
    ret = {}
    for var_name in name_list:
        if var_name not in keys:
            raise KeyError("'%s'" % var_name)
        val = _locals[var_name]
        ret[var_name] = val
    return ret, frame




class ArgumentTypeError(TypeError, object):
    def __init__(
            self,
            *args,
            given_name = None,
            given_type = None,
            expected_types = None,
            frame = None,
            **kwargs
    ):
        check = False
        for item in [given_name, given_type, expected_types]:
            if item is None:
                check = True
                break
        
        if not args and not check:
            try:
                _types = [ArgumentTypeError._get_type_string(a) for a in expected_types]
            except TypeError as e:
                if 'object is not iterable' not in str(e): raise(e)
                expected_types = [expected_types]
            _types = [ArgumentTypeError._get_type_string(a) for a in expected_types]
            if len(_types) == 0:
                raise Exception("_types has len 0. This should never happen.")
            elif len(_types) == 1: _types = "type %s" % _types[0]
            else:
                last_type = _types[-1]
                _types = "one of types " + ", ".join([t for t in _types[:-1]])
                _types += " or %s" % last_type

            given_type = ArgumentTypeError._get_type_string(given_type)
            
            if frame is None:
                fmt = "Argument '{name}' must be {types}, not {given_type}"
                args = (fmt.format(
                    name = given_name,
                    types = _types,
                    given_type = given_type,
                ),)
            else:
                fmt = "Argument '{name}' in '{function}' must be {types}, not {given_type}"
                
                function = inspect.getframeinfo(frame).function
                _self = frame.f_locals.get('self', None)
                if _self is not None:
                    function = _self.__class__.__name__ + '.' + function
                
                args = (fmt.format(
                    name = given_name,
                    types = _types,
                    given_type = given_type,
                    function = function,
                ),)
            
                    
        super(ArgumentTypeError, self).__init__(*args, **kwargs)

    @staticmethod
    def _get_type_string(obj):
        if hasattr(obj, "__module__"):
            module = obj.__module__
            qualname = obj.__qualname__
        elif isinstance(obj, np.dtype):
            module = type(obj).__module__
            qualname = type(obj).__qualname__
        else:
            raise Exception("Invalid annotation '%s'" % str(obj))
        if module == 'builtins':
            return "'%s'" % qualname
        else:
            return "'%s.%s'" % (module, qualname)


class ArgumentValueError(ValueError, object):
    def __init__(
            self,
            *args,
            given_name = None,
            given_value = None,
            expected_values = None,
            **kwargs
    ):
        if not args and None not in [given_name, given_value, expected_values]:
            if not hasattr(expected_values, '__iter__') or isinstance(expected_values, str):
                expected_values = [expected_values]
            
            if len(expected_values) == 0:
                raise Exception("expected_values has len 0. This should never happen.")
            elif len(expected_values) == 1: values = "value '%s'" % expected_values[0]
            elif len(expected_values) >= 2:
                values = "one of values " + ", ".join(["'%s'" % str(v) for v in expected_values[:-1]])
                values += " or '%s'" % expected_values[-1]
            
            fmt = "Argument '{name}' must have {values}, not '{value}'"
            args = (fmt.format(
                name = given_name,
                values = values,
                value = str(given_value),
            ),)
        super(ArgumentValueError, self).__init__(*args, **kwargs)
