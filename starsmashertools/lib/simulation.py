import starsmashertools.preferences as preferences
import starsmashertools.helpers.path as path
import starsmashertools.helpers.jsonfile as jsonfile
import starsmashertools
import starsmashertools.lib.input
import starsmashertools.lib.output
import starsmashertools.lib.logfile
import starsmashertools.lib.units
import starsmashertools.helpers.stacktrace
import starsmashertools.helpers.midpoint
from starsmashertools.lib.teos import TEOS
import numpy as np
from glob import glob

class Simulation(object):
    def __init__(self, directory):
        directory = path.realpath(directory)
        
        if not Simulation.valid_directory(directory):
            raise Exception("Invalid directory '%s'" % directory)

        super(Simulation, self).__init__()

        self.directory = directory

        self.input = starsmashertools.lib.input.Input(self.directory)
        self._children = None
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
        if self._units is None: self._units = starsmashertools.lib.units.Units(self)
        return self._units

    @property
    def teos(self):
        if self._teos is None and self['neos'] == 2:
            self._teos = TEOS(path.realpath(path.join(self.directory, self['eosfile'])))
        return self._teos

    # Keywords are passed to logfile.find() method
    def get_logfiles(self, **kwargs):
        if self._logfiles is None:
            self._logfiles = []
            for path in starsmashertools.lib.logfile.find(self.directory, **kwargs):
                self._logfiles += [starsmashertools.lib.logfile.LogFile(path, self)]
        return self._logfiles

    # Override this in children. Must return a list of Simulation objects
    def _get_children(self, *args, **kwargs):
        raise NotImplementedError

    def _get_children_from_hint_files(self):
        children = None
        hint_filenames = preferences.get_default('Simulation', 'children hint filenames', throw_error=False)
        if hint_filenames is not None:
            for name in hint_filenames:
                fname = path.join(self.directory, name)
                if path.isfile(fname):
                    with open(fname, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line: continue

                            if line.lower() == 'point mass':
                                if children is None: children = []
                                children += ['point mass']
                                continue
                            
                            simulation = None
                            try:
                                simulation = starsmashertools.get_simulation(line)
                            except FileNotFoundError as e:
                                if 'Directory does not exist' in str(e):
                                    warnings.warn("Failed to find directory in children hint file '%s': '%s'" % (fname, line))
                                else: raise

                            if simulation is not None:
                                if children is None: children = []
                                children += [simulation]
        return children

    # Load the children from the file saved in data/
    def _load_children(self, verbose=False):
        filename = preferences.get_default('Simulation', 'children file', throw_error=True)
        if verbose: print("Loading children from file")
        children = None
        if not path.isfile(filename):
            raise FileNotFoundError(filename)
        
        children_object = jsonfile.load(filename)
        for directory, _children in children_object.items():
            simulation = starsmashertools.get_simulation(directory)
            if simulation == self:
                if children is None: children = []
                # Sometimes children can be 'point mass'
                for child in _children:
                    if isinstance(child, str) and child.lower() != 'point mass':
                        child = starsmashertools.get_simulation(child)
                    children += [child]
        
        return children

    def _save_children(self, children_object=None, verbose=False):
        if verbose: print("Saving children to file")
        if not hasattr(self._children, '__iter__') or isinstance(self._children, str):
            raise TypeError("Property Simulation._children must be a non-str iterable")

        filename = preferences.get_default('Simulation', 'children file', throw_error=True)

        children_object = {}
        if path.isfile(filename):
            children_object = jsonfile.load(filename)
            
            if not isinstance(children_object, dict):
                raise TypeError("The object saved in '%s' must be a dictionary. Try deleting or renaming the file and running your code again." % str(filename))

        # Children can sometimes be 'None'
        to_save = []
        for child in self._children:
            if isinstance(child, Simulation):
                child = child.directory
            to_save += [child]
        children_object[self.directory] = to_save
        jsonfile.save(filename, children_object)

    # Return a list of Simulation objects used to create this simulation.
    # For example, if this Simulation is a dynamical simulation (nrelax = 0),
    # then it has one child, which is the binary scan that it originated from.
    #@profile
    def get_children(self, *args, **kwargs):
        verbose = kwargs.get('verbose', False)
        if self._children is None:
            # First see if the children are given to us in the data/ directory.
            # Load in the object for the first time if needed.
            try:
                self._children = self._load_children(verbose=verbose)
            except FileNotFoundError:
                pass
            
            # If loading the children didn't work
            if self._children is None:
                if verbose: print("Searching for child simulations for the first time for '%s'" % self.directory)
                
                # If there wasn't a file to load in the data/ directory, locate
                # the children manually and save them to the data/ directory.
                self._children = self._get_children_from_hint_files()
                if self._children is None:
                    # If we didn't get the children from the hint files,
                    # search for the children using the overidden method
                    
                    self._children = self._get_children(*args, **kwargs)
                    
                    if self._children is None:
                        raise Exception("Children found was 'None'. If you want to specify that a Simulation has no children, then its get_children() method should return empty list '[]', not 'None'.")
                
                # Now save the children to the data/ directory
                self._save_children(verbose=verbose)
        
        return self._children

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
        
        indices = np.arange(start, stop, step)
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
        


    # Return the Output file which corresponds with the beginning of an envelope
    # disruption event for a star consisting of the given IDs. An envelope is
    # considered 'disrupted' when some fraction of the particles are outside
    # some radius (measured from either the center of mass of the IDs or the
    # position of a core particle if there is one in the IDs).
    #
    # frac: The fraction of particles outside 'radius' compared to particles
    #       inside 'radius' for which the envelope is considered disrupted
    def get_envelope_disruption(
            self,
            IDs,
            radius,
            frac,
            omit_large=True,
            boundonly=True,
            # Give times in simulation units here
            search_window=(0, None),
    ):
        inv_nparticles = 1. / len(IDs)
        radiusSqr = radius * radius
        def get_frac(output):
            idx = np.full(len(output['ID']), False)
            idx[IDs] = True
            if boundonly:  idx[output['unbound']] = False
            if omit_large: idx[2*output['hp'] > radius] = False

            with starsmashertools.mask(output, idx) as masked:
                xyz = np.column_stack((masked['x'],masked['y'],masked['z']))
                cores = masked['u'] == 0
                ncores = sum(cores)
                if ncores == 1:
                    xc = masked['x'][cores][0]
                    yc = masked['y'][cores][0]
                    zc = masked['z'][cores][0]
                else:
                    xc, yc, zc = starsmashertools.math.center_of_mass(
                        masked['am'],
                        masked['x'],
                        masked['y'],
                        masked['z'],
                    )
                center = np.array([xc, yc, zc])
                r2 = np.sum((xyz - center)**2, axis=-1)
                frac = np.sum(masked['am'][r2 >= radiusSqr]) / np.sum(masked['am'])
            return frac
                
        filenames = self.get_outputfiles()
        outputs = [starsmashertools.lib.output.Output(filename, self) for filename in filenames]
        m = starsmashertools.helpers.midpoint.Midpoint(outputs)
        
        if search_window[0] > 0:
            m.set_criteria(
                lambda output: output['t'] < search_window[0],
                lambda output: output['t'] == search_window[0],
                lambda output: output['t'] > search_window[0],
            )
            m.objects = m.objects[m.objects.index(m.get()):]
        if search_window[1] is not None:
            m.set_criteria(
                lambda output: output['t'] < search_window[1],
                lambda output: output['t'] == search_window[1],
                lambda output: output['t'] > search_window[1],
            )
            m.objects = m.objects[:m.objects.index(m.get())]
        
        m.set_criteria(
            lambda output: get_frac(output) < frac,
            lambda output: get_frac(output) == frac,
            lambda output: get_frac(output) > frac,
        )
        return m.get()
    

    class OutputNotInSimulationError(Exception):
        def __init__(self, simulation, output, message=None):
            if message is None:
                message = "Output is not a member of simulation: '%s' for simulation '%s'" % (output.path, simulation.directory)
            super(Simulation.OutputNotInSimulationError, self).__init__(message)


            

    



