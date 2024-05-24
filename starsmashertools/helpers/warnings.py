# A custom replacement for Python's crappy warning system

def warn(message, category=UserWarning):
    import inspect
    import starsmashertools
    import os
    import warnings
    
    try:
        src = os.path.join(
            os.path.realpath(starsmashertools.SOURCE_DIRECTORY),
            'starsmashertools',
        )
        frame = inspect.currentframe()
        while frame is not None:
            finfo = inspect.getframeinfo(frame)
            filename = os.path.realpath(finfo.filename)
            if not filename.startswith(src): break
            frame = frame.f_back

        # If the caller was outside the starsmashertools source directory, call a
        # proper-looking warning
        if frame is None:
            return warnings.warn(message, category=category)

        # If the caller was in the starsmashertools source directory, then behave
        # normally.
        return warnings.warn_explicit(
            message,
            category,
            finfo.filename,
            finfo.lineno,
        )
    except: # If anything goes wrong, just raise a normal warning
        return warnings.warn(message, category)


    
def filterwarnings(*args, **kwargs):
    import warnings
    return warnings.filterwarnings(*args, **kwargs)

def resetwarnings(*args, **kwargs):
    import warnings
    return warnings.resetwarnings(*args, **kwargs)
