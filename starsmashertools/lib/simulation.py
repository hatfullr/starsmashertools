import starsmashertools.preferences as preferences
import starsmashertools.helpers.path as path
import starsmashertools.helpers.jsonfile as jsonfile
import starsmashertools.lib.input
import starsmashertools.lib.output
import starsmashertools.lib.logfile
import starsmashertools.helpers.stacktrace
from starsmashertools.lib.teos import TEOS
import numpy as np
from glob import glob

class Simulation(object):
    # A list of attribute names to exclude in the JSON object
    json_exclude = [
        '__dict__',
        '_teos',
        'teos',
        'units',
        '_units',
        '_logfiles',
    ]
    
    def __init__(self, directory):
        directory = path.realpath(directory)
        
        if not Simulation.valid_directory(directory):
            raise Exception("Invalid directory '%s'" % directory)

        super(Simulation, self).__init__()

        self.directory = directory

        self.input = starsmashertools.lib.input.Input(self.directory)
        self._children = None
        self._children_object = None
        self._units = None
        self._teos = None
        self._logfiles = None

        self.reader = starsmashertools.lib.output.Reader(self)

    def __hash__(self):
        return hash(self.directory)
        
    def __eq__(self, other):
        # Check if the basenames are the same, so that e.g. pdc.json
        # files can work on different file systems
        if not isinstance(other, Simulation): return False
        return path.basename(self.directory) == path.basename(other.directory)

    def __getitem__(self, key): return self.input[key]

    def __contains__(self, item):
        if isinstance(item, str):
            if starsmashertools.helpers.path.isfile(item):
                return item in self.get_output_iterator()
        elif isinstance(item, starsmashertools.lib.output.Output):
            return item in self.get_output_iterator()
        elif isinstance(item, starsmashertools.lib.output.OutputIterator):
            # All the files in the iterator must be members of this simulation
            full_iterator = self.get_output_iterator()
            for filename in item.filenames:
                if filename not in full_iterator: return False
            return True
        return False

    def keys(self, *args, **kwargs): return self.input.keys(*args, **kwargs)
    def values(self, *args, **kwargs): return self.input.values(*args, **kwargs)
    def items(self, *args, **kwargs): return self.input.items(*args, **kwargs)

    @staticmethod
    def valid_directory(directory):
        return path.get_src(directory) is not None
    
    @property
    def units(self):
        if self._units is None: self._units = Units(self)
        return self._units

    @property
    def teos(self):
        if self._teos is None and self['neos'] == 2:
            self._teos = TEOS(path.realpath(path.join(self.directory, self['eosfile'])))
        return self._teos

    def get_logfiles(self, pattern=None):
        if self._logfiles is None:
            self._logfiles = []
            for path in starsmashertools.lib.logfile.find(self.directory, pattern=pattern):
                self._logfiles += [starsmashertools.lib.logfile.LogFile(path)]
        return self._logfiles

    # Override this in children. Must return a list of children
    def _get_children(self, *args, **kwargs):
        raise NotImplementedError

    # Return a list of simulations used to create this simulation.
    # For example, if this Simulation is a dynamical simulation (nrelax = 0),
    # then it has one child, which is the binary scan that it originated from.
    #@profile
    def get_children(self, *args, **kwargs):
        
        verbose = kwargs.get('verbose', False)
        if self._children is None:
            filename = preferences.get_default('Simulation', 'children file', throw_error=True)

            # Try to load the children file
            if self._children_object is None:
                if path.isfile(filename):
                    if verbose: print("Loading children from file")
                    self._children_object = jsonfile.load(filename)

            # If we didn't find and read the children file, make our own
            if self._children_object is None: self._children_object = {}
            
            if self.directory in self._children_object.keys():
                # Load the children from the file
                if verbose: print("Loading children from file")
                self._children_object = jsonfile.load(filename)
                children = self._children_object[self.directory]
                if children is not None:
                    self._children = [None if child is None else Simulation.from_json(child) for child in children]
            else:
                print("Getting children for the first time for '%s'" % self.directory)
                children = self._get_children(*args, **kwargs)
                children_to_save = None
                if children is not None:
                    children_to_save = [child if not hasattr(child, 'to_json') else child.to_json() for child in children]
                self._children_object[self.directory] = children_to_save
                if verbose: print("Saving children to file")
                jsonfile.save(filename, self._children_object)
                self._children = children
        
        return self._children

    def to_json(self):
        ret = {'type' : self.__module__+"."+type(self).__name__ }
        for attr in dir(self):
            if hasattr(self, 'json_exclude'):
                if attr in self.json_exclude: continue
            val = getattr(self, attr)
            if isinstance(val, (str, int, float, bool, list, dict, np.ndarray)):
                ret[attr] = val
        return ret

    @staticmethod
    def from_json(obj):
        instance = eval((obj['type']+"('%s')") % obj['directory'])
        for key, val in obj.items():
            if hasattr(instance, key):
                setattr(instance, key, val)
        return instance


    def get_outputfiles(self, pattern=None):
        if pattern is None:
            pattern = preferences.get_default('Simulation', 'output files', throw_error=True)
        matches = glob(path.join(self.directory, pattern))
        if matches:
            matches = sorted(matches)
        return matches

    # The initial file is always the one that was written first.
    def get_initialfile(self, pattern=None):
        return self.get_outputfiles(pattern=pattern)[0]

    # Given string arguments, return data from each of this simulation's
    # output files
    def get_output_iterator(self, start=0, stop=None, step=1, **kwargs):
        filenames = self.get_outputfiles()
        filenames = np.asarray(filenames, dtype=object)

        if stop is None: stop = len(filenames)
        
        indices = np.arange(start, stop + 1, step)
        filenames = filenames[indices].tolist()

        # Now that we have all the file names, we can create an output iterator
        # from them
        return starsmashertools.lib.output.OutputIterator(filenames, self, **kwargs)

    # If indices is None, returns all the Output objects in this simulation as
    # an OutputIterator. If indices is an integer, returns a single Output
    # object. If indices is an iterable, returns an OutputIterator.
    def get_output(self, indices=None):
        filenames = self.get_outputfiles()
        if indices is None:
            return self.get_output_iterator()
        else:
            if isinstance(indices, int):
                try:
                    return starsmashertools.lib.output.Output(filenames[indices], self)
                except IndexError as e:
                    raise IndexError("Invalid index '%d' in simulation '%s'" % (indices, self.directory)) from e
            elif not isinstance(indices, str) and hasattr(indices, '__iter__'):
                filenames = np.array(filenames, dtype=str)
                try:
                    return starsmashertools.lib.output.OutputIterator(filenames[indices].tolist(), self)
                except IndexError as e:
                    raise IndexError("Invalid indices '%s' in simulation '%s'" % (str(indices), self.directory)) from e
        raise ValueError("Invalid value for argument 'indices'. Must be 'None', of type 'int', or be an iterable object, not '%s'" % type(indices).__name__)


    def get_file(self, filename):
        filepath = path.join(self.directory, filename)
        if not path.isfile(filepath):
            raise FileNotFoundError("No file '%s' found in simulation '%s'" % (filename, self.directory))
        return filepath
        
    


    class OutputNotInSimulationError(Exception):
        def __init__(self, simulation, output, message=None):
            if message is None:
                message = "Output is not a member of simulation: '%s' for simulation '%s'" % (output.path, simulation.directory)
            super(Simulation.OutputNotInSimulationError, self).__init__(message)


            

    



# This class is used to convert the raw StarSmasher outputs to cgs units
class Units(dict, object):
    def __init__(self, simulation):
        self._initializing = True
        
        self.length = simulation['runit']
        self.mass = simulation['munit']

        obj = {
            # Header units
            'hco' : self.length,
            'hfloor' : self.length,
            'sep0' : self.length,
            'tf' : self.time,
            'dtout' : self.time,
            't' : self.time,
            'tjumpahead' : self.time,
            'trelax' : self.time,
            'dt' : self.time,
            'omega2' : self.frequency * self.frequency,
            'erad' : self.specificenergy,
            'displacex' : self.length,
            'displacey' : self.length,
            'displacez' : self.length,

            

            # Output file units
            'x' : self.length,
            'y' : self.length,
            'z' : self.length,
            'am' : self.mass,
            'hp' : self.length,
            'rho' : self.density,
            'vx' : self.velocity,
            'vy' : self.velocity,
            'vz' : self.velocity,
            'vxdot' : self.acceleration,
            'vydot' : self.acceleration,
            'vzdot' : self.acceleration,
            'u' : self.specificenergy,
            'udot' : self.specificluminosity,
            'grpot' : self.specificenergy,
            'meanmolecular' : 1.,
            'cc' : 1,
            'divv' : self.velocity / self.length, # If this is divergence of velocity
            'ueq' : self.specificenergy,
            'tthermal' : self.time,


            
            # Extra units. You can add your own here if you want more units, but
            # it's probably best to use the preferences.py file instead.
            'popacity' : self.opacity,
            'uraddot' : self.specificluminosity,
            'temperature' : 1,
            'tau' : 1,
            'dEemergdt' : self.luminosity,
            'dEdiffdt' : self.luminosity,
            'dEmaxdiffdt' : self.luminosity,
            'uraddotcool' : self.specificluminosity,
            'uraddotheat' : self.specificluminosity,
        }

        _locals = {}
        for attr in dir(self):
            _locals[attr] = getattr(self, attr)
        
        for key, val in preferences.get_default("Simulation", "units").items():
            if isinstance(val, (float, int)):
                obj[key] = val
            elif isinstance(val, str):
                obj[key] = eval(val, {}, _locals)
            else:
                raise TypeError("All values declared in preferences.py in defaults['Simulation']['units'] must be type 'float', 'int', or 'str', not '%s'" % type(val).__name__)
            
        
        super(Units, self).__init__(obj)

        
        del self._initializing
        
    def __setitem__(self, item):
        if not hasattr(self, "_initializing"):
            raise Exception("Changing Unit values after initialization is not supported")

    @property
    def time(self): return np.sqrt(self.length**3 / (self.gravconst * self.mass))

    @property
    def frequency(self): return 1. / self.time
        
    @property
    def gravconst(self): return 6.67390e-08 # This comes from src/starsmasher.h

    @property
    def area(self): return self.length * self.length

    @property
    def volume(self): return self.area * self.length
    
    @property
    def energy(self): return self.gravconst * self.mass * self.mass / self.length

    @property
    def velocity(self): return self.length / self.time
    
    @property
    def acceleration(self): return self.velocity / self.time

    @property
    def density(self): return self.mass / self.volume

    @property
    def opacity(self): return self.length**2 / self.mass

    @property
    def luminosity(self): return self.energy / self.time

    @property
    def flux(self): return self.luminosity / self.area


    

    @property
    def specificenergy(self): return self.energy / self.mass

    @property
    def specificluminosity(self): return self.luminosity / self.mass