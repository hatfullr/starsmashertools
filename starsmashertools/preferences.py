import os
import types
import numpy as np

defaults = {
    'Simulation' : {
        'search directory' : '~/data/',
        # Used to identify the directories which contain the StarSmasher source code
        'src identifiers' : [
            'starsmasher.h',
            'init.f',
            'output.f',
        ],
        'children file' : os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            'data',
            'children.json.gz',
        ),
        'output files' : 'out*.sph',
        # Edit this object to add/overwrite the simulation units. You can
        # define either a number or a string, where the string is code which
        # can be run by eval, where the available variables are the attributes
        # of the Units class (see starsmashertools/lib/simulation.py).
        'units' : {
            'xyz' : 'length',
            'r' : 'length',
            'r2' : 'length * length',
            'v' : 'velocity',
            'v2' : 'velocity * velocity',
            'vdot' : 'acceleration',
            'ekin' : 'specificenergy',
            'etot' : 'specificenergy',
        },
    },
    'Dynamical' : {
        'original initial restartrad' : 'restartrad.sph.orig',
    },
    'Input' : {
        # We choose whichever name we find first in the directory
        'input filenames' : [
            'sph.input',
        ],
        'src init filename' : 'init.f',
    },
    'LogFile' : {
        'file pattern' : 'log*.sph',
    },
    'OutputIterator' : {
        'max buffer size' : 100,
    },
    'Output' : {
        # These are cached quantities that the Output objects will be
        # aware of at initialization. Each entry must be a function
        # that accepts only an Output object as argument.
        # To specify your own units edit the 'units' object in
        # 'Simulation' above.
        'cache' : {
            'ID' : lambda obj: np.arange(obj['ntot']),
            'xyz': lambda obj: np.column_stack((obj['x'], obj['y'], obj['z'])),
            'r2' : lambda obj: np.sum(obj['xyz']**2, axis=-1),
            'r' : lambda obj: np.sqrt(obj['r2']),
            'v2' : lambda obj: np.sum(np.column_stack((obj['vx'], obj['vy'], obj['vz']))**2, axis=-1),
            'vdot' : lambda obj: np.sum(np.column_stack((obj['vxdot'], obj['vydot'], obj['vzdot']))**2, axis=-1),
            'v' : lambda obj: np.sqrt(obj['v2']),
            'ekin' : lambda obj: 0.5 * obj['v2'],
            'etot' : lambda obj: obj['ekin'] + obj['grpot'] + obj['u'],
            'unbound' : lambda obj: obj['etot'] >= 0,
            'angular momentum' : lambda obj: obj['am'] * np.sqrt(obj['r2'] * obj['v2']),
        },
    },
}

#constants = {
#    'gravitational' : 6.6739e-8 # Comes from starsmasher.h in all source files
#}




def get_default(name, default_name, throw_error=False):
    if not isinstance(name, str): name = type(name).__name__
    if name in defaults.keys():
        if default_name in defaults[name].keys():
            return defaults[name][default_name]
        
    if throw_error:
        raise Exception("Missing field '%s' in '%s' in preferences.defaults" % (default_name, name))
    
