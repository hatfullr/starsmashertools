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
import starsmashertools.helpers.string
import starsmashertools.helpers.argumentenforcer
import starsmashertools.helpers.file
import starsmashertools.helpers.compressiontask
from starsmashertools.helpers.apidecorator import api
from starsmashertools.helpers.clidecorator import cli
from starsmashertools.lib.teos import TEOS
import numpy as np
import glob
import tarfile

class Simulation(object):
    ############################################################################
    # private attributes

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(
            self,
            directory : str,
    ):
        directory = path.realpath(directory.strip())
        
        if not Simulation.valid_directory(directory):
            raise Simulation.InvalidDirectoryError(directory)

        super(Simulation, self).__init__()

        self.directory = directory

        self.input = starsmashertools.lib.input.Input(self.directory)
        self._children = None
        self._units = None
        self._teos = None
        self._logfiles = None
        self._compression_task = None

        self.reader = starsmashertools.lib.output.Reader(self)

    @property
    def compressed(self):
        filename = self._get_compression_filename()
        if not starsmashertools.helpers.path.isfile(filename): return False
        return starsmashertools.helpers.compressiontask.CompressionTask.isCompressedFile(filename)
            
    def __hash__(self):
        return hash(self.directory)

    @api
    def __eq__(self, other):
        # Check if the basenames are the same, so that e.g. pdc.json
        # files can work on different file systems
        if not isinstance(other, Simulation): return False
        return path.basename(self.directory) == path.basename(other.directory)

    @api
    def __getitem__(self, key): return self.input[key]

    @api
    def __contains__(self, item):
        if isinstance(item, str):
            if path.isfile(item):
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

    # Override this in children. Must return a list of Simulation objects
    def _get_children(self):
        raise NotImplementedError

    def _get_children_from_hint_files(self):
        children = None
        hint_filenames = preferences.get_default('Simulation', 'children hint filenames', throw_error=False)
        if hint_filenames is not None:
            for name in hint_filenames:
                fname = path.join(self.directory, name)
                if path.isfile(fname):
                    with starsmashertools.helpers.file.open(fname, 'r') as f:
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
        print_warning = False
        if 'version' in children_object.keys():
            version = children_object.pop('version')
            print_warning = version.split('.')[0] < starsmashertools.__version__.split('.')[0]
        else: print_warning = True
        if print_warning:  
            warnings.warn("The children data stored in '%s' is from a different version of starsmashertools. If you encounter an error, try deleting the children data")
        
        
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
        children_object['version'] = starsmashertools.__version__
        jsonfile.save(filename, children_object)



    def _get_compression_filename(self):
        dirname = path.basename(self.directory)
        return path.join(self.directory, dirname+".zip")
        









        
    ############################################################################
    # public attributes

    @staticmethod
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def valid_directory(directory : str):
        return path.get_src(directory) is not None

    @api
    def keys(self, *args, **kwargs): return self.input.keys(*args, **kwargs)
    @api
    def values(self, *args, **kwargs): return self.input.values(*args, **kwargs)
    @api
    def items(self, *args, **kwargs): return self.input.items(*args, **kwargs)
    
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
    @api
    def get_logfiles(self,**kwargs):
        if self._logfiles is None:
            self._logfiles = []
            for _path in starsmashertools.lib.logfile.find(self.directory, **kwargs):
                if path.getsize(_path) > 0:
                    self._logfiles += [starsmashertools.lib.logfile.LogFile(_path, self)]
        return self._logfiles
    
    @api
    @cli('starsmashertools')
    def get_children(self, verbose : bool = False, cli : bool = False):
        """
        Return a list of `starsmashertools.lib.simulation.Simulation` objects
        that were used to create this simulation. For example, if this
        simulation is a dynamical simulation (nrelax = 0), then it has one
        child, which is the binary scan that it originated from. Similarly, a
        binary scan has two children, which are each the stellar relaxations
        that make up the two stars in the binary.

        This function acts as a wrapper for `~._get_children`, which is
        overridden in subclasses of Simulation. This allows the results to be
        stored on the hard drive in starsmashertools/data/ for quick access.
        
        Parameters
        ----------
        verbose : bool, default = False
            If `True`, debug messages will be printed to the console.

        Returns
        -------
        list
            A list of the child simulations.
        """
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
                    
                    self._children = self._get_children()
                    
                    if self._children is None:
                        raise Exception("Children found was 'None'. If you want to specify that a Simulation has no children, then its get_children() method should return empty list '[]', not 'None'.")
                
                # Now save the children to the data/ directory
                self._save_children(verbose=verbose)

        if cli:
            string = []
            for i, child in enumerate(self._children):
                if issubclass(child.__class__, Simulation):
                    string += ["Star %d: %s" % (i+1,child.directory)]
                else: string += ["Star %d: %s" % (i+1, str(child))]
            return "\n".join(string)
        return self._children

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_file(
            self,
            filename_or_pattern : str,
            recursive : bool = True,
    ):
        """
        Search the simulation directory for a file or files that match a
        pattern.

        Parameters
        ----------
        filename_or_pattern : str
            A file name in the simulation directory or a pattern (e.g., 
            ``out*.sph``)
        recursive : bool, default = True
            If `True`, searches all sub directories in the simulation. If
            `False`, searches just the simulation directory.

        Returns
        -------
        `list`

        See Also
        --------
        ``glob``
        """
        if recursive: _path = path.join(self.directory, '**', filename_or_pattern)
        else: _path = path.join(self.directory, filename_or_pattern)
        return glob.glob(_path, recursive=recursive)
        
    @api
    def get_outputfiles(self, pattern : str | type(None) = None):
        if pattern is None:
            pattern = preferences.get_default('Simulation', 'output files', throw_error=True)
        matches = self.get_file(pattern)
        if matches:
            matches = sorted(matches)
        return matches

    # The initial file is always the one that was written first.
    @api
    def get_initialfile(self, pattern : str | type(None) = None):
        return self.get_outputfiles(pattern=pattern)[0]

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_output_iterator(
            self,
            start : int | type(None) = None,
            stop : int | type(None) = None,
            step : int | type(None) = None,
            **kwargs
    ):
        """
        Return a `starsmashertools.lib.output.OutputIterator` containing all the
        `starsmashertools.lib.output.Output` objects present in this simulation.
        
        Parameters
        ----------
        start : int, None, default = None
            Passed directly to `~.get_output`.

        stop : int, None, default = None
            Passed directly to `~.get_output`.

        step : int, None, default = None
            Passed directly to `~.get_output`.

        Other Parameters
        ----------------
        **kwargs
            Other keyword arguments are passed directly to 
            `starsmashertools.lib.output.OutputIterator`
        
        Returns
        -------
        iterator
            The created `starsmashertools.lib.output.OutputIterator` instance.
        """
        # Now that we have all the file names, we can create an output iterator
        # from them
        outputs = self.get_output(start=start, stop=stop, step=step)
        if not isinstance(outputs, list): outputs = [outputs]
        filenames = [output.path for output in outputs]
        return starsmashertools.lib.output.OutputIterator(filenames, self, **kwargs)

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_output_at_time(
            self,
            time : int | float | starsmashertools.lib.units.Unit,
    ):
        """
        Return the `starsmashertools.lib.output.Output` object of this
        simulation whose time best matches the given time. Raises a ValueError
        if the given time is out of bounds.

        Parameters
        ----------
        time : int, float, `starsmashertools.lib.units.Unit`
            The simulation time to locate. If an `int` or `float` are given then
            this value represents time in the simulation units. If a 
            `starsmashertools.lib.units.Unit` is given, it will be converted to
            this simulation's time unit.

        Returns
        -------
        `starsmashertools.lib.output.Output`
            The Output object closest to the given time.
        """
        if isinstance(time, starsmashertools.lib.units.Unit):
            time = float(time.convert(self.units['t']))

        outputs = self.get_output()
        if not outputs: raise ValueError("Cannot find output at time %f because there are no output files in simulation '%s'" % (time, str(self.directory)))
            
        if time < 0 or time > outputs[-1]['t']:
            raise ValueError("Time %f is out of bounds [0, %f]" % (time, outputs[-1]['t']))
        
        m = starsmashertools.helpers.midpoint.Midpoint(outputs)
        m.set_criteria(
            lambda output: output['t'] < time,
            lambda output: output['t'] == time,
            lambda output: output['t'] > time,
        )
        return m.get()

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_output(
            self,
            start : int | type(None) = None,
            stop : int | type(None) = None,
            step : int | type(None) = None,
            times : int | float | starsmashertools.lib.units.Unit | list | tuple | np.ndarray | type(None) = None,
            indices : list | tuple | np.ndarray | type(None) = None,
    ):
        """
        Obtain a list of `starsmashertools.lib.output.Output` objects associated
        with this Simulation. Returns all outputs if no arguments are specified.
        
        Parameters
        ----------
        start : int, None, default = None
            The starting index of a slice of the list of all output files.

        stop : int, None, default = None
            The ending index in the slice of the list of all output files.

        step : int, None, default = None
            How many output files to skip in the slice.

        time : int, float, `starsmashertools.lib.units.Unit`, list, tuple, np.ndarray, default = None
            If given, returns the output files that are closest to the given
            simulation time or collection of times. This can possibly include
            duplicate items.

        indices : list, tuple, np.ndarray, default = None
            If given, returns the output files at each index from the result of
            `~.get_outputfiles()`.
        
        Returns
        -------
        list, `starsmashertools.lib.output.Output`
            A list of `starsmashertools.lib.output.Output` objects. If the list
            contains only a single item, that item is returned instead.
        """
        filenames = np.asarray(self.get_outputfiles(), dtype=object)
        if times is not None:
            starsmashertools.helpers.argumentenforcer.enforcevalues({
                'start' : [None],
                'stop' : [None],
                'step' : [None],
                'indices' : [None],
            })
            if hasattr(times, '__iter__'):
                ret = [self.get_output_at_time(time) for time in times]
                if len(ret) == 1: return ret[0]
                return ret
            else:
                return self.get_output_at_time(times)
        elif indices is not None:
            starsmashertools.helpers.argumentenforcer.enforcevalues({
                'start' : [None],
                'stop' : [None],
                'step' : [None],
                'times' : [None],
            })
            indices = np.asarray(indices, dtype=int)
            filenames = filenames[indices]
        else:
            if start is not None and stop is None and step is None:
                # User is intending to just get a single index
                if start != -1:
                    stop = start + 1
            s = slice(start, stop, step)
            filenames = filenames.tolist()[s]
        
        ret = [starsmashertools.lib.output.Output(filename, self) for filename in filenames]
        if len(ret) == 1: return ret[0]
        return ret

    @api
    def get_compressed_properties(self):
        """
        Get a dictionary of properties on the files contained in the compressed
        archive.

        Returns
        -------
        dict
            A dictionary whose keys are the names of the files in the compressed
            archive that they would have if the archive were decompressed. Each
            value is a dictionary holding various values corresponding to each
            file in the archive.
        """
        if not self.compressed: return {}
        filename = self._get_compression_filename()
        return starsmashertools.helpers.compressiontask.CompressionTask.get_compressed_properties(filename)

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def compress(
            self,
            filename : str | type(None) = None,
            include_patterns : list | type(None) = None,
            exclude_patterns : list | type(None) = None,
            recursive : bool = True,
            delete : bool = True,
            delete_after : bool = True,
            **kwargs,
    ):
        """
        Create a compressed version of the Simulation. File creation times are
        preserved.

        Parameters
        ----------
        filename : str, None, default = None
            The name of the resulting compressed file. If `None` then the name
            of the file will be the name of the simulation directory with
            `.tar.{method}` on the end, where `{method}` is replaced with the
            compression method shorthand, e.g. `gz`, `bz2`, etc.
        
        include_patterns : list, None, default = None
            File name patterns to include in the compression. If `None`, uses 
            the `compress include` value in `~.preferences`.

        exclude_patterns : list, None, default = None
            File name patterns to exclude in the compression. If `None`, uses
            the `compress exclude` value from `~.preferences`.

        recursive : bool, default = True
            If `True`, subdirectories are also searched for files matching the
            given patterns. If `False`, only searches the main simulation
            directory.

        delete : bool, default = True
            If `True`, the files which are compressed are deleted.

        delete_after : bool, default = True
            If `True`, compressed files are deleted only after all files have
            been compressed. If `False`, each file is deleted after it has been
            compressed. If `delete` is `False` this option is ignored.

        Other Parameters
        ----------------
        **kwargs
            Remaining keyword arguments are passed directly to 
            `~helpers.compressiontask.CompressionTask.compress`.

        See Also
        --------
        `~.decompress`
        `~.helpers.compressiontask.CompressionTask.compress`
        `~.helpers.compressiontask.CompressionTask.get_methods`
        `~.preferences`
        """

        # Obtain the file names to be compressed.
        if include_patterns is None:
            include_patterns = preferences.get_default('Simulation', 'compress include')
        if exclude_patterns is None:
            exclude_patterns = preferences.get_default('Simulation', 'compress exclude')

        exclude_files = []
        for pattern in exclude_patterns:
            fs = self.get_file(pattern, recursive=recursive)
            exclude_files += fs
        
        files = []
        for pattern in include_patterns:
            fs = self.get_file(pattern, recursive=recursive)
            files += [f for f in fs if f not in exclude_files]
            
        for key, val in self.items():
            if not isinstance(val, str): continue
            _path = path.join(self.directory, val)
            if not path.isfile(_path): continue
            files += [_path]

        filename = self._get_compression_filename()
        self._compression_task = starsmashertools.helpers.compressiontask.CompressionTask()
        self._compression_task.compress(files, filename, delete=delete, delete_after=delete_after, **kwargs)
        self._compression_task = None

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def decompress(
            self,
            filename : str | type(None) = None,
            delete : bool = True,
            **kwargs
    ):
        """
        Decompress the simulation from the given filename.

        Parameters
        ----------
        filename : str, None default = None
            The filename to decompress. If `None`, the simulation directory is
            searched for a compressed file whose path ends with `.tar.{method}`
            where `{method}` is one of the available compression methods from
            `~helpers.compressiontask.CompressionTask.get_methods`.
            The file chosen is one which has a `compression.sstools` file
            included in it, which is an empty file that is created when using
            `~helpers.compressiontask.CompressionTask.compress`.

        delete : bool, default = True
            If `True`, the compressed file is deleted after decompression.
        
        Other Parameters
        ----------------
        **kwargs
            Remaining keyword arguments are passed directly to
            `~helpers.compressiontask.CompressionTask.decompress`.

        See Also
        --------
        `~.compress`
        `~.helpers.compresisontask.CompressionTask.decompress`
        `~.helpers.compressiontask.CompressionTask.get_methods`
        """

        if not self.compressed:
            raise Exception("Cannot decompress a simulation that is not compressed")
        
        self._compression_task = starsmashertools.helpers.compressiontask.CompressionTask()
        
        if filename is None:
            filename = self._get_compression_filename()
            self._compression_task.decompress(filename, delete=delete, **kwargs)
            self._compression_task = None
            return
        raise FileNotFoundError("Failed to find any valid compressed files in directory '%s'" % self.directory)


    @api
    def get_size(self):
        """Returns the size of the simulation in bytes.

        Returns
        -------
        int
        """
        return path.get_directory_size(self.directory)


    @api
    def get_output_headers(self, **kwargs):
        """
        Read all the headers of the output files in this simulation and return
        them as a dictionary.

        Parameters
        ----------
        keys : list
            The keys to query the output file headers for.

        Returns
        -------
        `dict`
            Each key is a `starsmashertools.lib.output.Output` object and each
            value is the entire header.

        Other Parameters
        ----------------
        kwargs : dict
            Extra keyword arguments passed to `.get_output_iterator`. Note that
            keyword `return_headers` is always `True` and `return_data` is
            always `False`. Keyword `asynchronous` is set to `False` by default
            because reading headers is often faster that way.
        """
        kwargs['return_headers'] = True
        kwargs['return_data'] = False
        kwargs['asynchronous'] = kwargs.get('asynchronous', False)
        iterator = self.get_output_iterator(**kwargs)
        ret = {}
        for output in iterator:
            ret[output] = output.header
        return ret

    @api
    @cli('starsmashertools', -1)
    def get_ejected_mass(self, *args, cli : bool = False, **kwargs):
        """
        Return the total ejected mass for the given output file indices.

        Parameters
        ----------

        Returns
        -------
        """
        
        outputs = self.get_output(*args, **kwargs)
        if not isinstance(outputs, list): outputs = [outputs]
        ret = np.zeros(len(outputs))
        for i, output in enumerate(outputs):
            if 'mejecta' in output.keys():
                ret[i] = output['mejecta']
            elif np.any(output['unbound']):
                ret[i] = np.sum(output['am'][output['unbound']])
        if len(ret) == 1: ret = ret[0]

        if cli: return str(ret)
        return ret








    class InvalidDirectoryError(Exception): pass
    

    class OutputNotInSimulationError(Exception):
        def __init__(self, simulation, output, message=None):
            if message is None:
                message = "Output is not a member of simulation: '%s' for simulation '%s'" % (output.path, simulation.directory)
            super(Simulation.OutputNotInSimulationError, self).__init__(message)


            

    



