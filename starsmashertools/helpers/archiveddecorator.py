import functools

archive_key = 'functions'

class FunctionNotFoundError(Exception, object): pass

class NullResult(object): pass

def archived(f):
    # Add this decorator to Simulation functions which take a long time
    @functools.wraps(f)
    def wrapper(obj, *args, **kwargs):
        import starsmashertools.lib.simulation
        import starsmashertools.helpers.path

        # "obj" here is "self"
        if not isinstance(obj, starsmashertools.lib.simulation.Simulation):
            raise TypeError("Decorator '@archived' can only be used on methods in Simulation objects.")
        
        fed_kwargs = get_default_kwargs(f)
        fed_kwargs.update(kwargs)

        try:
            return get_function_result_from_archive(obj, f, args, fed_kwargs)
        except FunctionNotFoundError: pass
        
        result = f(obj, *args, **kwargs)
        save_function_to_archive(obj, f, result, args, fed_kwargs)
        return result
    
    return wrapper

def get_default_kwargs(function):
    import inspect
    signature = inspect.signature(function)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

def compare(args1, kwargs1, args2, kwargs2):
    if len(args1) != len(args2): return False
    if set(kwargs1.keys()) != set(kwargs2.keys()): return False
    for v1,v2 in zip(args1, args2):
        if v1 != v2: return False
    for key, val in kwargs1.items():
        if val != kwargs2[key]: return False
    return True

def get_function_result_from_archive(simulation, function, args, kwargs):
    archive = simulation.archive
    key = function.__qualname__
    current_state = simulation.get_state()
    
    if archive_key in archive.keys():
        stored = archive[archive_key].value
        if current_state != stored['state']:
            # Signal to overwrite all the stored values
            raise FunctionNotFoundError(key)
        
        for key, val in stored['calls'].items():
            # stored['calls'][key] = [ArchivedFunction, ArchivedFunction, ...]
            for obj in val[::-1]:
                # Compare from newest-to-oldest (probably faster on average)
                if not compare(args, kwargs, obj['args'], obj['kwargs']):
                    continue
                # Found a match
                return obj['result']
    
    # We get here if we did not find the function call in the archive
    raise FunctionNotFoundError(key)

def save_function_to_archive(simulation, function, result, args, kwargs):
    import starsmashertools

    max_saved_func_args = starsmashertools.preferences.get(
        'Simulation', 'max saved func args', throw_error = False,
    )
    if max_saved_func_args is None: max_saved_func_args = 100

    current_state = simulation.get_state()
    archive = simulation.archive
    key = function.__qualname__

    value = {
        'args' : args,
        'kwargs' : kwargs,
        'result' : result,
    }
    obj = {
        'state' : current_state,
        'calls' : {key:[value]},
    }

    
    if archive_key in archive.keys():
        archived_obj = archive[archive_key].value
        if archived_obj['state'] == current_state:
            obj = archived_obj
            
            values = obj['calls'].get(key, [])
            if len(values) >= max_saved_func_args:
                values = values[len(values) - max_saved_func_args + 1:]
            
            values += [value]
            obj['calls'][key] = values
    
    # Overwrite all the stored values
    archive.add(
        archive_key,
        obj,
        origin = simulation.directory,
    )
