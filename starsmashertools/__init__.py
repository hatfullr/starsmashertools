import starsmashertools.lib.relaxation
import starsmashertools.lib.binary
import starsmashertools.lib.dynamical
import starsmashertools.lib.simulation
import starsmashertools.lib.output
import starsmashertools.helpers.argumentenforcer


# Return the type of simulation that a directory is
@starsmashertools.helpers.argumentenforcer.enforcetypes
def get_simulation(directory : str):
    simulation = starsmashertools.lib.simulation.Simulation(directory)
    nrelax = simulation['nrelax']
    if nrelax == 0: return starsmashertools.lib.dynamical.Dynamical(directory)
    if nrelax == 1: return starsmashertools.lib.relaxation.Relaxation(directory)
    if nrelax == 2: return starsmashertools.lib.binary.Binary(directory)
    raise NotImplementedError("nrelax = %d" % nrelax)

@starsmashertools.helpers.argumentenforcer.enforcetypes
def get_type(directory : str):
    try:
        return type(get_simulation(directory))
    except NotImplementedError as e:
        if "nrelax =" in str(e):
            return None

def relaxation(*args, **kwargs):
    return starsmashertools.lib.relaxation.Relaxation(*args, **kwargs)

def binary(*args, **kwargs):
    return starsmashertools.lib.binary.Binary(*args, **kwargs)

def dynamical(*args, **kwargs):
    return starsmashertools.lib.dynamical.Dynamical(*args, **kwargs)

def output(*args, **kwargs):
    return starsmashertools.lib.output.Output(*args, **kwargs)

def iterator(*args, **kwargs):
    return starsmashertools.lib.output.OutputIterator(*args, **kwargs)

