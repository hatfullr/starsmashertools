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

# Returns an Output object which contains only the particles specified
# by the given mask. The mask can be either a numpy boolean array
@starsmashertools.helpers.argumentenforcer.enforcetypes
def get_particles(
        mask : np.ndarray | list | tuple,
        output : starsmashertools.lib.output.Output,
):
        
    if isinstance(output, starsmashertools.lib.output.Output):
        output.mask(mask)
        return output

    if not hasattr(output, '__iter__'):
        output = [output]

    ret = []
    for o in output:
        o.mask(mask)
        ret += [convert(o)]
    
    if len(ret) == 1: return ret[0]
    return ret
