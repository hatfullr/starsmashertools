# A custom replacement for Python's crappy warning system
#from builtins import warnings as python_warnings
import inspect
import starsmashertools
import os
import warnings

def warn(message, category=UserWarning):
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

    # If the caller was in the starsmashertools source directory, then behave
    # normally.
    if frame is None:
        return warnings.warn(message, category=category)

    # If the caller was outside the starsmashertools source directory, call a
    # proper-looking warning
    thewarning = warnings.warn_explicit(
        message,
        category,
        finfo.filename,
        finfo.lineno,
    )
    return thewarning

    
def filterwarnings(*args, **kwargs):
    return warnings.filterwarnings(*args, **kwargs)

def resetwarnings(*args, **kwargs):
    return warnings.resetwarnings(*args, **kwargs)
