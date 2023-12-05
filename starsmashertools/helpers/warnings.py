# A custom replacement for Python's crappy warning system
from builtins import warnings as python_warnings
import inspect
import starsmashertools
import os

def warn(message, category=UserWarning):
    src = os.path.join(
        os.path.realpath(starsmashertools.SOURCE_DIRECTORY),
        'starsmashertools',
    )
    frame = inspect.currentframe()
    while frame is not None:
        filename = os.path.realpath(frame.filename)
        if not filename.startswith(src): break
        frame = frame.f_back

    # If the caller was in the starsmashertools source directory, then behave
    # normally.
    if frame is None:
        return python_warnings.warn(message, category=category)

    # If the caller was outside the starsmashertools source directory, call a
    # proper-looking warning
    thewarning = python_warnings.warn_explicit(
        message,
        category,
        frame.filename,
        frame.lineno,
    )
    return thewarning
    
