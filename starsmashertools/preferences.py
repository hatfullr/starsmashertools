import os
import numpy as np

defaults = {
    'Simulation' : {
        'search directory' : '~/data/',
        # Used to identify the directories which contain the StarSmasher source
        # code
        'src identifiers' : [
            'starsmasher.h',
            'init.f',
            'output.f',
        ],
        # You can create a file with this name in your simulation directories to
        # instruct starsmashertools on where to find the child simulations. Each
        # file should be a text file with one directory per line.
        'children hint filename' : 'children.sstools',
        'output files' : 'out*.sph',
        
        # Files which are defined in the StarSmasher code and present in the
        # simulation directory will be compressed, such as 'sph.eos' when neos=2
        # for example. Others need to be added manually here.
        'compress include' : [
            'out*.sph',
            'log*.sph',
            'energy*.sph',
            'col*.sph',
            'restartrad.sph*',
        ],
        # Files you want to exclude from compression
        'compress exclude' : [
        ],
    },
    
    # 
    # Note that currently the setting of values 'gram', 'sec', 'cm', and
    # 'kelvin' in src/starsmasher.h is not supported. We expect all these values
    # to equal 1.d0 for now.
    'Units' : {
        # Edit this object to add/overwrite the simulation units. You can define
        # either a number or a string, where the string is code which can be run
        # by eval, where the available variables are the attributes of the Units
        # class (see starsmashertools/lib/units.py). Some special strings are
        # 'length' and 'mass' which correspond with 'runit' and 'munit' in the
        # sph.input file or in the src/init.f file.
        'extra' : {
            'xyz' : 'length',
            'r' : 'length',
            'r2' : 'length * length',
            'v' : 'velocity',
            'v2' : 'velocity * velocity',
            'vdot' : 'acceleration',
            'ekin' : 'specificenergy',
            'etot' : 'specificenergy',
        },
        # Items in this dict decide how units are converted from one to another.
        'unit conversions' : [
            {
                'base' : 'cm',
                'conversions' : [
                    ['m' , 1e2],
                    ['km', 1e5],
                ],
            },
            {
                'base' : 's',
                'conversions' : [
                    ['min', 60],
                    ['hr' , 3600],
                    ['day', 3600*24],
                    ['yr' , 3600*24*365.25],
                ],
            },
        ],
        # Add items to this list to create additional abbreviations for unit
        # labels. The conversions are queried in-order from top to bottom. See
        # the Unit.Label class in starsmashertools/lib/units.py for details.
        'label conversions' : [
            ['erg', 'cm*g*g/s*s'], # Don't remove this
            ['erg/g', 'cm*g/s*s'], # Don't remove this
        ],
    },
    'Dynamical' : {
        'original initial restartrad' : 'restartrad.sph.orig',
    },
    'Input' : {
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
            'mejecta' : lambda obj: np.sum(obj['am'][obj['unbound']]) if np.any(obj['unbound']) else 0.,

            # Add your own functions here
        },
    },
    'PDCFile' : {
        'filename' : 'pdc.json.gz',
    },
    'GPUJob' : {
        'threadsperblock' : 512,
    },
}



def get_default(name, default_name, throw_error=False):
    if not isinstance(name, str): name = type(name).__name__
    if name in defaults.keys():
        if default_name in defaults[name].keys():
            return defaults[name][default_name]
        
    if throw_error:
        raise Exception("Missing field '%s' in '%s' in preferences.defaults" % (default_name, name))
    
