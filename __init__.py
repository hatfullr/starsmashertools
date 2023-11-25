# This file will only be loaded if starsmashertools is imported from the
# directory above this one.

# Erase this file's identity
del __builtins__, __cached__, __doc__, __file__, __loader__,  __package__, __spec__

# import the __ private members that are native to Python
from .starsmashertools import __spec__, __builtins__, __cached__, __doc__, __loader__, __file__
import os

del __path__
__path__ = [os.path.dirname(__file__)]
del os

__name__ = 'starsmashertools'
__package__ = 'starsmashertools'

# This correctly imports all functions, including private ones.
with open(__file__, 'r') as f:
    exec(f.read())

try: del __warningregistry__
except: pass


