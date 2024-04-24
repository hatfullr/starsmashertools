"""
This file can be overwritten whenever starsmashertools is updated. You can 
create a copy of this file in the user directory to make your own preferences 
that will not be overwritten. It is your responsibility to keep your own 
preferences.py file up-to-date. Values that are missing will be skipped and the
corresponding values from the defaults will be used.

To completely exclude default values, you can do:

    prefs = {...}
    exclude = [
        'lib.units.Unit.conversions.Msun
    ]

For example. Each '.' is a delimiter for the keys to follow in the dict.
"""

import os
import numpy as np
from starsmashertools.lib.archive import REPLACE_OLD, REPLACE_NEQ


prefs = {
    
    ############################################################################
    
    'helpers' : {
        'string' : {
        }, # 'string'
        
        'file' : {
            'Lock' : {
                'timeout' : float('inf'), # seconds
            }, # 'Lock'
        }, # 'file'

        'formatter' : {
            'FormatSheet' : {
                # Directory paths to additional format sheets
                'directories' : [],
            }, # 'FormatSheet'
        }, # 'formatter'

        'gpujob' : {
            'GPUJob' : {
                'threadsperblock' : 512,
            },
        }, # 'gpujob'
    }, # 'helpers'
    
    ############################################################################
    
    'lib' : {
        'archive' : {
            'Archive' : {
                'replacement flags' : (REPLACE_OLD, REPLACE_NEQ,),
                # Setting this to False will prevent starsmashertools from
                # overwriting Archives which have old formatting, but you will
                # need to update the Archive formatting yourself in order to
                # read it.
                'auto update format' : True,
                # The maximum allowed size in bytes that the buffer can have
                # when auto save is disabled. When the buffer exceeds this value
                # the archive is saved. default = 1e5 (0.1 MB)
                'max buffer size' : int(1e5),
            }, # 'Archive'
            'SimulationArchive' : {
                # This name will be used to create an archive of computed data.
                # See the SimulationArchive class for details.
                # CAUTION: Editing this value will cause any old archives in
                #          simulations to be unrecognized by starsmashertools.
                #          You will need to change their names before you can
                #          use them again.
                'filename' : 'sstools.archive',
            }, # 'SimulationArchive'
        }, #'archive'

        'binary' : {
            'Binary' : {
                'get_RLOF CLI output window' : 6,
            }, # 'Binary'
        }, # 'binary'

        # Settings for obtaining the radiative flux of output files which
        # originate from simulations that use radiative cooling in Hatfull et
        # al. (in prep).
        'flux' : {
            'get' : {
                'options' : {
                    # Image
                    'resolution' : [400, 400],
                    'theta' : 0, # Viewing angle (polar)
                    'phi' : 0,   # Viewing angle (azimuthal)
                    'fluffy' : False,
                    'rays' : True,

                    # Limits
                    # Flux contributions are not considered beyond this optical
                    # depth.
                    'tau_s' : 20.,
                    # Particles with tau < tau_skip are ignored entirely.
                    'tau_skip' : 1.e-5,
                    # Particles with Teff <= teff_cut are excluded in surf_br_v.
                    'teff_cut' : 3500.,

                    # Dust
                    # Give a value in cm^2/g, or None to turn off artificial
                    # dust.
                    'dust_opacity' : 1,
                    # The temperature range in which to use artificial dust.
                    # Give None for either value to turn off the lower/upper
                    # limits.
                    'dust_Trange' : [100, 1000],

                    # Spectrum
                    'spectrum_size' : 1000,
                    'lmax' : 3000, # nm, stand for 1000K, no resolving below
                    'dl_il' : 3., # nm resolution of boxes
                    'dl' : 10,
                    'nbox' : 10,
                    'nboxh' : 5,
                    'l_range' : [10, 3000],
                    'factor1' : 1.191045e-05, # for Plank function integrations
                    'factor2' : 1.438728e+00,
                    'filters' : {
                        # Use None to turn off lower/upper bounds
                        'V' : [500, 600], # nm
                        'R' : [590, 730], # nm
                        'I' : [720, 880], # nm
                    },
                }, # 'options'
            }, # 'get'
        }, # 'flux'
        
        # StarSmasher source code input parameters
        'input' : {
            'Input' : {
                'src init filename' : 'init.f',
            }, # 'Input'
        }, # 'input'

        'logfile' : {
            'find' : {
                'file pattern' : 'log*.sph',
            }, # 'find'
        }, # 'logfile'

        'output' : {
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
                    'angularmomentum' : lambda obj: obj['am'] * np.sqrt(obj['r2'] * obj['v2']),
                    'mejecta' : lambda obj: np.sum(obj['am'][obj['unbound']]) if np.any(obj['unbound']) else 0.,
                    # Add your own functions here. The keys need to be written
                    # as though they were variable names, so that
                    # Output.condense works properly. You can define a function
                    # somewhere above outside this dict if it's needed.
                },
                
                # The maximum number of results from Output.condense that is
                # allowed to be stored in the simulation archives per output.
                'max stored condense results' : 10000,
                
                'string format sheet' : 'cli.format',
            }, # 'Output'

            'OutputIterator' : {
                'max buffer size' : 100,
            }, # 'OutputIterator'
        }, # 'output'

        'report' : {
            'Report' : {
                'columns' : [
                    {
                        'args' : (
                            lambda sim: sim.directory,
                        ),
                        'kwargs' : {
                            'header' : 'name',
                            'formatter' : '{:20}',
                            'shorten' : {
                                'args' : (20,),
                                'kwargs' : {
                                    'where' : 'left',
                                },
                            },
                        },
                    },
                    {
                        'args' : (
                            lambda sim: sim.get_start_time(),
                        ),
                        'kwargs' : {
                            'header' : 'start',
                            'formatter' : '{:7g}',
                        },
                    },
                    {
                        'args' : (
                            lambda sim: sim.get_current_time(),
                        ),
                        'kwargs' : {
                            'header' : 'current',
                            'formatter' : '{:7g}',
                        },
                    },
                    {
                        'args' : (
                            lambda sim: sim.get_stop_time(),
                        ),
                        'kwargs' : {
                            'header' : 'stop',
                            'formatter' : '{:7g}',
                        },
                    },
                ], # 'columns'
            }, # 'Report'
        }, # 'report'
        
        'simulation' : {
            'Simulation' : {
                # Whenever a function in a Simulation is called, if it has a
                # decorator '@archived' then before running that function we
                # check the SimulationArchive to see if that function has been
                # called with the same arguments. If the Simulation has not
                # changed since when we last called that function with those
                # arguments, we return the result that was stored in the
                # SimulationArchive. Otherwise, or if the function isn't in the
                # SimulationArchive, then we call the function and store the
                # arguments and result in the SimulationArchive. This is the
                # maximum number of function calls we can store in the
                # SimulationArchive this way, per function. That is, the
                # SimulationArchive will contain up to 'max saved func args'
                # calls for each function called. When the maximum is exceeded,
                # the oldest (first) call is replaced.
                'max saved func args' : 100,

                'output files' : 'out*.sph',

                'search directory' : '/',

                # Used to identify the directories which contain the StarSmasher
                # source code
                'src identifiers' : [
                    'starsmasher.h',
                    'init.f',
                    'output.f',
                ],

                # The name of the file which is used to restart a simulation
                # from the output of some other simulation. This file should
                # always be identical to an output file from some other
                # simulation, if used at all.
                'start file' : 'restartrad.sph.orig',

                # Files which are created by the StarSmasher code. We check
                # these files for changes when we track the State of a
                # Simulation.
                'state files' : [
                    'out*.sph',
                    'log*.sph',
                    'energy*.sph',
                    'col*.sph',
                    'restartrad.sph*',
                ],
            }, # 'Simulation'
        }, # 'simulation'

        'units' : {
            'Unit' : {
                # Items in this dict decide how units are converted from one to
                # another.
                'conversions' : {
                    'm' : '1e2 cm',
                    'km' : '1e5 cm',
                    'Rsun' : '6.9599e10 cm',
                    'Msun' : '1.9891e33 g',
                    'min' : '60 s',
                    'hr' : '3600 s',
                    'day' : '86400 s',
                    'yr' : '31557600 s',
                    # Comes from 'constants' in lib/units.py
                    'Lsun' : '3.828e33 cm*cm*g/s*s*s',
                }, # 'conversions'
                'Label' : {
                    # Add items to this list to create additional abbreviations
                    # for unit labels. The conversions are queried in-order from
                    # top to bottom. See the Unit.Label class in
                    # starsmashertools/lib/units.py for details.
                    'conversions' : {
                        'cm*cm*g/s*s' : 'erg', # Don't remove this
                        'cm*cm/s*s' : 'erg/g', # Don't remove this
                        #['erg', 'cm*cm*g/s*s'], # Don't remove this
                        #['erg/g', 'cm*cm/s*s'], # Don't remove this
                    },
                }, # 'Label'
            }, # 'Unit'
            # Note that currently the setting of values 'gram', 'sec', 'cm', and
            # 'kelvin' in src/starsmasher.h is not supported. We expect all
            # these values to equal 1.d0 for now.
            'Units' : {
                # Edit this object to add/overwrite the simulation units. You
                # can define either a number or a string, where the string is
                # code which can be run by eval, where the available variables
                # are the attributes of the Units class (see
                # starsmashertools/lib/units.py). Some special strings are
                # 'length' and 'mass' which correspond with 'runit' and 'munit'
                # in the sph.input file or in the src/init.f file.
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
            }, # 'Units'
        }, # 'units'
    }, # 'lib'
    
    ############################################################################

    'mpl' : {
        'artists' : {
            'PlottingPreferences' : {
                # The given kwargs below override the settings here
                'defaults' : {
                    'color'      :  'k', # Color of particles
                    's'          :    1, # Size of particles
                    'marker'     :  '.', # Particle markers
                    'linewidth'  :    0, # Width of marker outlines
                    'rasterized' : True, # Make a non-vectorized plot, is faster
                }, # 'defaults'
        
                # This determines the keywords to use in the plots. They are
                # accessed in-order while plotting. For example, the first item
                # determines the kwargs for the primary star and the second item
                # for the secondary star. If additional information is plotted
                # after the list has run out of items, it uses the default
                # values only thereafter.
                'kwargs' : [
                    # Star 1
                    {
                        'label'  : 'Primary',
                    }, # Use defaults
                    # Star 2
                    {
                        'color'  : 'r',
                        'label'  : 'Secondary',
                    },
                ], # 'kwargs'
                
                # For core particles only. These override all other settings
                # above. As in, these settings are applied "last".
                'core kwargs' : [
                    { # Star 1
                        's'         :  50,
                        'marker'    : 'o',
                        'color'     : 'none',
                        'edgecolor' : 'k',
                        'linewidth' : 1,
                        'label'     : 'Primary Cores',
                        'zorder'    : float('inf'), # Put on top of everything
                    },
                    { # Star 2
                        's'         : 50,
                        'marker'    : 'o',
                        'color'     : 'none',
                        'edgecolor' : 'r',
                        'linewidth' : 1,
                        'label'     : 'Secondary Cores',
                        'zorder'    : float('inf'), # Put on top of everything
                    },
                ], # 'core kwargs'
        
                'legend' : {
                    'loc' : 'lower left',
                    'bbox_to_anchor' : (0, 1.0, 1, 0.2),
                    'borderaxespad' : 0,
                    'handletextpad' : 0,
                    'ncol' : 5,
                    'borderpad' : 0.1,
                    'frameon' : False,
                    'labelspacing' : 0,
                    
                    # This is a special option that doesn't get used by
                    # Matplotlib. Instead, we manually adjust the legend handles
                    # to give them a fixed size so that the markers are always
                    # clearly visible.
                    'markersize' : 50,
                }, # 'legend'
            }, # 'PlottingPreferences'

            'FluxPlot' : {
                'images' : {
                    'interpolation' : 'none',
                    'cmap' : 'inferno',
                }, # 'image'
                'particle highlight' : {
                    'interpolation' : 'none',
                    'color' : 'c',
                }, # 'particle highlight'
                'particle outline' : {
                    'facecolors' : 'none',
                    'edgecolors' : 'c',
                    'linewidths' : 1.,
                }, # 'particle outline'
            }, # 'FluxPlot'
        }, # 'artists'

        'axes' : {
        }, # 'axes'
        
        'figure' : {
            'Figure' : {
                'stylesheet directories' : [
                    # The starsmashertools default directory
                    '{SOURCE_DIRECTORY}/starsmashertools/mpl/stylesheets',
            
                    # Add paths to directories below which contain *.mplstyle
                    # files for use with Matplotlib.
                ],
            }, # 'Figure'
            'FluxFigure' : {
                # These can be strings which refer to keys in the 'image' dict
                # of a FluxResult, or they can be functions which accept a
                # FluxResult as the input and return a 2D NumPy array.
                'values' : [
                    'rho_cell',
                    'T_cell',
                    'kappa_cell',
                    'Teff_cell',
                    'flux',
                ],
            }, # 'FluxFigure'
        }, # 'figure'
    }, # 'mpl'

    ############################################################################
}
