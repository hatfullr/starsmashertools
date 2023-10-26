import typing
import types
import inspect
import functools
import numpy as np

# Use these functions as wrappers like so:
#
# @starsmashertools.helpers.argumentenforcer.enforcetypes
# def myfunc(stuff : str):
#     return stuff
#
# If the argument 'stuff' is detected as not of type 'str' then an error is
# raised. Arguments that are an instance of a subclass of the specified type
# pass the check.
# 
# If you want to constrain input arguments to specific values then you can add
# the following to the beginning of your function:
#
# @starsmashertools.helpers.argumentenforcer.enforcetypes
# def myfunc(stuff : str):
#     starsmashertools.helpers.argumentenforcer.enforcevalues({
#         'stuff' : ['things', 'nelly'],
#     })
#     return stuff
#
# An error will be raised if 'stuff' has any value other than 'things' or
# 'nelly'.

# Works as a wrapper by default, unless this function is called otherwise with a
# dictionary object passed as input
def enforcetypes(f):
    # Dict behavior
    if isinstance(f, dict):
        return _enforcetypes(f)
    
    # Wrapper behavior. We need to use functools here or else we get into
    # trouble with other wrappers.
    @functools.wraps(f)
    def type_checker(*args, **kwargs):
        hints = typing.get_type_hints(f)
        all_args = kwargs.copy()
        if hasattr(f, "__wrapped__"):
            varnames = f.__wrapped__.__code__.co_varnames
        else: varnames = f.__code__.co_varnames
        all_args.update(dict(zip(varnames, args)))
        for key in all_args.keys():
            if key in hints.keys():
                expected_types = hints[key]
                arg = all_args[key]
                types_for_error = _check_type(arg, expected_types)
                if types_for_error is not None:
                    raise ArgumentTypeError(
                        given_name     = key,
                        given_type     = type(arg) if not isinstance(arg, np.ndarray) else arg.dtype.type,
                        expected_types = types_for_error,
                    )
        return f(*args, **kwargs)
    
    return type_checker

@enforcetypes
def _enforcetypes(obj : dict):
    variables = _get_variables_in_context_from_dict(obj.keys())

    for var_name, var_val in obj.items():
        value = variables[var_name]
        types_for_error = _check_type(value, var_val)
        if types_for_error is not None:
            raise ArgumentTypeError(
                given_name     = var_name,
                given_type     = type(value),
                expected_types = var_val,
            )


numpy_conversions = {
    np.complex64 : complex,
    np.complex128 : complex,
    np.float16 : float,
    np.float32 : float,
    np.float64 : float,
    np.int8 : int,
    np.int16 : int,
    np.int32 : int,
    np.int64 : int,
    np.bool_ : bool,
    np.str_ : str,
    np.uint8 : int,
    np.uint16 : int,
    np.uint32 : int,
    np.uint64 : int,
}

def _check_type(obj, _types):
    # This is the only way to check for Union types...
    if isinstance(_types, types.UnionType) or hasattr(_types, "__args__"):
        if isinstance(obj, np.ndarray) and np.ndarray in _types.__args__:
            others = [a for a in _types.__args__ if a is not np.ndarray]
            if not others: return _types.__args__
            # Check if this type is included in any of the Union types
            for key, val in numpy_conversions.items():
                if key in _types.__args__ or val in _types.__args__:
                    break
            else:
                return None
            new_type = numpy_conversions.get(obj.dtype.type, None)
            if new_type is None: return None
            if new_type not in _types.__args__: return others
            return None
        
        if (not isinstance(obj, _types.__args__) and
            not any([issubclass(obj.__class__, t) for t in _types.__args__])):
            return _types.__args__
    else:
        if hasattr(_types, '__iter__') and not _types in [str, dict]:
            try:
                _types = tuple(_types)
            except TypeError as e:
                if 'object is not iterable' not in str(e):
                    raise
        
        if (type(obj) != _types and
            not issubclass(obj.__class__, _types)):
            return _types
    return None


    
    

def enforcevalues(obj):
    # Make sure the inputs are correct
    for key, var_val in obj.items():
        # if is non-str iterable
        if not hasattr(var_val, '__iter__') or isinstance(var_val, str):
            raise TypeError("You must specify a non-str iterable object to check variable values against. Received '%s'" % str(var_val))
        
    variables = _get_variables_in_context_from_dict(obj.keys())
    for var_name, var_val in obj.items():
        value = variables[var_name]
        if value not in var_val:
            raise ArgumentValueError(
                given_name      = var_name,
                given_value     = str(value),
                expected_values = var_val,
            )

@enforcetypes
def enforce_numpy_dtype(
        array : np.ndarray,
        expected_dtypes : list | tuple,
):
    dtype = array.dtype
    for _type in expected_dtypes:
        if np.issubdtype(dtype, _type): break
    else:
        varname = None
        # Start with the caller of this function
        frame = inspect.currentframe().f_back.f_back
        while frame is not None:
            varnames = frame.f_code.co_varnames
            for name, value in frame.f_locals.items():
                if value is array:
                    varname = name
                    break
            if varname is not None: break
            frame = frame.f_back
        if varname is None: raise Exception("Failed to find the original name of argument 'array'")
        raise ArgumentTypeError(
            given_name = varname,
            given_type = dtype,
            expected_types = expected_dtypes,
        )


# Returns a dictionary with keys equal to the variable names and values equal to
# the variable values, where the variables match the keys from the input
# list
def _get_variables_in_context_from_dict(name_list):
    frame = inspect.currentframe()
    while frame is not None:
        # Don't search any frames in this file
        if frame.f_code.co_filename != __file__:
            for name in name_list:
                if name not in frame.f_locals.keys():
                    break
            else:
                # Found the right frame that has all the given variables
                break
        frame = frame.f_back
    else:
        raise Exception("Failed to find the context that contains all the given variable names")
    
    _locals = frame.f_locals
    keys = _locals.keys()
    ret = {}
    for var_name in name_list:
        if var_name not in keys:
            raise KeyError("'%s'" % var_name)
        val = _locals[var_name]
        ret[var_name] = val
    return ret

class ArgumentTypeError(TypeError, object):
    __module__ = TypeError.__module__
    def __init__(self, *args, given_name=None, given_type=None, expected_types=None, **kwargs):

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
            fmt = "Argument '{name}' must be {types}, not {given_type}"
            given_type = ArgumentTypeError._get_type_string(given_type)
            args = (fmt.format(
                name=given_name,
                types=_types,
                given_type=given_type,
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
        
        if module == 'builtins':
            return "'%s'" % qualname
        else:
            return "'%s.%s'" % (module, qualname)




        
class ArgumentValueError(ValueError, object):
    __module__ = ValueError.__module__
    def __init__(self, *args, given_name=None, given_value=None, expected_values=None, **kwargs):
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

    def __getstate__(self, *args, **kwargs):
        return self.__dict__
    #def __setstate__(self, *args, **kwargs):
    #    return super(ArgumentValueError, self).__setstate__(*args, **kwargs)
    #def __getnewargs__(self, *args, **kwargs):
    #    return super(ArgumentValueError, self).__getnewargs__(*args, **kwargs)
    #def __reduce__(self, *args, **kwargs):
    #    return super(ArgumentValueError, self).__reduce__(*args, **kwargs)
