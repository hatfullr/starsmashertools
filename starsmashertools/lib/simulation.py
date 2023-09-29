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
from starsmashertools.lib.teos import TEOS
import numpy as np
import glob
import tarfile

class Simulation(object):
    ############################################################################
    # private attributes

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def __init__(
            self,
            directory : str,
    ):
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


    def _get_compression_method(self, filename):
        methods = self.get_compression_methods()
        for method in methods:
            if filename.endswith(method): return method
        return None

    def _get_compression_filename(self, method):
        dirname = starsmashertools.helpers.path.basename(self.directory)
        filename = dirname + ".tar." + method
        return starsmashertools.helpers.path.join(self.directory, filename)

    def _get_compressed_name(self):
        filenames = []
        for method in self.get_compression_methods():
            filename = self._get_compression_filename(method)
            
            if starsmashertools.helpers.path.isfile(filename):
                with tarfile.open(filename, 'r:'+method) as tar:
                    try:
                        tar.getmember('compression.sstools')
                    except KeyError: continue
                filenames += [filename]

        if not filenames: return None

        # Sort the compress files by their modification times so that we get the
        # most recent file
        filenames = starsmashertools.helpers.file.sort_by_mtimes(filenames)
        return filenames[0]

    def _get_compression_method(self, filename):
        for method in self.get_compression_methods():
            if filename.endswith(method): return method

    def _get_files_for_compression(
            self,
            include_patterns = None,
            exclude_patterns = None,
            recursive = True,
    ):
        """Obtain the file names to be compressed."""
        if include_patterns is None:
            include_patterns = preferences.get_default('Simulation', 'compress include')
        if exclude_patterns is None:
            exclude_patterns = preferences.get_default('Simulation', 'compress exclude')

        exclude_files = []
        for pattern in exclude_patterns:
            files = self.get_file(pattern, recursive=recursive)
            exclude_files += files
        
        filenames = []
        for pattern in include_patterns:
            files = self.get_file(pattern, recursive=recursive)
            filenames += [f for f in files if f not in exclude_files]
            
        for key, val in self.items():
            if not isinstance(val, str): continue
            path = starsmashertools.helpers.path.join(self.directory, val)
            if not starsmashertools.helpers.path.isfile(path): continue
            filenames += [path]
        return filenames




        









        
    ############################################################################
    # public attributes

    @staticmethod
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def valid_directory(directory : str):
        return path.get_src(directory) is not None

    def keys(self, *args, **kwargs): return self.input.keys(*args, **kwargs)
    def values(self, *args, **kwargs): return self.input.values(*args, **kwargs)
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

    @property
    def compressed(self):
        return self._get_compressed_name() is not None


        

    # Keywords are passed to logfile.find() method
    def get_logfiles(self, **kwargs):
        if self._logfiles is None:
            self._logfiles = []
            for path in starsmashertools.lib.logfile.find(self.directory, **kwargs):
                self._logfiles += [starsmashertools.lib.logfile.LogFile(path, self)]
        return self._logfiles

    

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

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def get_file(
            self,
            filename_or_pattern : str,
            recursive : bool = True,
    ):
        """Search the simulation directory for a file or files that match a
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
        
    
    def get_outputfiles(self, pattern = None):
        if pattern is None:
            pattern = preferences.get_default('Simulation', 'output files', throw_error=True)
        matches = self.get_file(pattern)
        if matches:
            matches = sorted(matches)
        return matches

    # The initial file is always the one that was written first.
    def get_initialfile(self, pattern = None):
        return self.get_outputfiles(pattern=pattern)[0]

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def get_output_iterator(
            self,
            start = None,
            stop = None,
            step = None,
            **kwargs
    ):
        """
        Return an `OutputIterator` containing all the `Output` object present in
        this simulation. The `start`, `stop`, and `step` arguments are used with
        `slice(start, stop, step)`.
        
        Parameters
        ----------
        start : int, None, default = None
            The index in the list of all output files to begin the iterator at.

        stop : int, None, default = None
            The index in the list of all output files to stop the iterator at.

        step : int, None, default = None
            How many output files to skip on each iterator step.
        
        Returns
        -------
        `.OutputIterator`
            The created `.OutputIterator` instance.
        
        Other Parameters
        ----------------
        **kwargs : `~starsmashertools.lib.output.OutputIterator` keywords.
            Keywords that are passed to the `OutputIterator`

        """
        s = slice(start, stop, step)
        filenames = self.get_outputfiles()[s]
        
        # Now that we have all the file names, we can create an output iterator
        # from them
        return starsmashertools.lib.output.OutputIterator(filenames, self, **kwargs)

    # If indices is None, returns all the Output objects in this simulation as
    # an OutputIterator. If indices is an integer, returns a single Output
    # object. If indices is an iterable, returns an OutputIterator.
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def get_output(
            self,
            indices = None,
    ):
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



    # Return the Output file which corresponds with the beginning of an envelope
    # disruption event for a star consisting of the given IDs. An envelope is
    # considered 'disrupted' when some fraction of the particles are outside
    # some radius (measured from either the center of mass of the IDs or the
    # position of a core particle if there is one in the IDs).
    #
    # frac: The fraction of particles outside 'radius' compared to particles
    #       inside 'radius' for which the envelope is considered disrupted
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def get_envelope_disruption(
            self,
            IDs : list | tuple | np.ndarray,
            radius : float,
            frac : float,
            omit_large : bool = True,
            boundonly : bool = True,
            # Give times in simulation units here
            search_window : tuple = (0, None),
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





    
    def get_compression_methods(self):
        methods = list(tarfile.TarFile.OPEN_METH.keys())
        if 'tar' in methods: methods.remove('tar')
        return methods

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def compress(
            self,
            filename = None,
            method = None,
            include_patterns = None,
            exclude_patterns = None,
            delete : bool = True,
            delete_after : bool = True,
            recursive : bool = True,
            verbose : bool = True,
    ):
        """Create a compressed version of the Simulation. File creation times
        are preserved.

        A special file named `compression.sstools` is added to the final 
        compressed file for identification later when using
        ``starsmashertools.lib.simulation.Simulation.decompress()``.

        Parameters
        ----------
        filename : str, None, default = None
            The name of the resulting compressed file. If `None` then the name
            of the file will be the name of the simulation directory with
            `.tar.{method}` on the end, where `{method}` is replaced with the
            compression method shorthand, e.g. `gz`, `bz2`, etc.
        method : str, None, default = None
            Compression method to be used. If `None`, uses the default method
            from ``starsmashertools.preferences``. You can check the available
            compression methods on your system using
            ``starsmashertools.lib.simulation.Simulation.get_compression_methods()``.
        include_patterns : list, None, default = None
            File name patterns to include in the compression. If `None`, uses 
            the `compress include` value in ``starsmashertools.preferences``.
        exclude_patterns : list, None, default = None
            File name patterns to exclude in the compression. If `None`, uses
            the `compress exclude` value from ``starsmashertools.preferences``.
        delete : bool, default = True
            If `True`, the files which are compressed are deleted.
        delete_after : bool, default = True
            If `True`, compressed files are deleted after compression has
            completed. If `False`, files are deleted after compression has 
            completed. If `delete` is `False` this option is ignored.
        recursive : bool, default = True
            If `True`, subdirectories are also searched for files matching the
            given patterns. If `False`, only searches the main simulation
            directory.
        verbose : bool, default = True
            If `True`, debug messages are printed to the console.

        Returns
        -------
        None

        See Also
        --------
        ``starsmashertools.lib.simulation.Simulation.decompress()``
        ``starsmashertools.lib.simulation.Simulation.get_compression_methods()``
        ``starsmashertools.preferences``
        """

        methods = self.get_compression_methods()
        if method is None:
            method = preferences.get_default('Simulation', 'compression method')
            if method not in methods: method = methods[0]
            
        starsmashertools.helpers.argumentenforcer.enforcevalues({
            'method' : methods,
        })
        
        if self.compressed:
            raise FileExistsError("Cannot compress simulation '%s' because it is already compressed. If you wish to change the contents of the compressed file, please decompress the simulation first and then re-compress it." % self.directory)
        
        filename = self._get_compression_filename(method)
        files = self._get_files_for_compression(
            include_patterns = include_patterns,
            exclude_patterns = exclude_patterns,
            recursive = recursive,
        )

        fname = path.join(self.directory, 'compression.sstools')
        with starsmashertools.helpers.file.open(fname, 'w') as f:
            f.write("This file is used by starsmashertools to identify compressed archives it created. If you are reading this, then it probably means something went wrong during a compression. It is always safe to delete this file, but if it is removed from the tar ball it belongs to then starsmashertools might have trouble understanding which files to decompress. See the starsmashertools for more information: https://starsmashertools.readthedocs.io")
        files += [fname]

        
        with tarfile.open(filename, 'w:'+method) as tar:
            for f in files:
                if verbose: print("Adding '%s' to '%s'" % (f, filename))
                try:
                    tar.add(f, arcname=path.relpath(f, self.directory))
                except:
                    tar.extractall()
                    path.remove(filename)
                    path.remove(fname)
                    raise
                if delete and not delete_after: path.remove(f)

        # Remove the old files
        if delete and delete_after:
            for f in files: path.remove(f)

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def decompress(
            self,
            filename = None,
            delete : bool = True,
            verbose : bool = False,
    ):
        """Decompress the simulation from the given filename.

        Parameters
        ----------
        filename : str, None default = None
            The filename to decompress. If `None`, the simulation directory is
            searched for a compressed file whose path ends with `.tar.{method}`
            where `{method}` is one of the available compression methods from
            ``starsmashertools.lib.simulation.Simulation.get_compression_methods()``.
            The file chosen is one which has a `compression.sstools` file
            included in it, which is an empty file that is created when using
            ``starsmashertools.lib.simulation.Simulation.compress()``.
        delete : bool, default = True
            If `True`, the compressed file is deleted after decompression.
        verbose : bool, default = False
            If `True`, debug messages are printed to the console.

        Returns
        -------
        None

        See Also
        --------
        ``starsmashertools.lib.simulation.Simulation.compress()``
        ``starsmashertools.lib.simulation.Simulation.get_compression_methods()``
        """
        if not self.compressed:
            raise FileNotFoundError("Cannot decompress simulation because it is not compressed: '%s'" % self.directory)

        if filename is None:
            filename = self._get_compressed_name()
            
        method = self._get_compression_method(filename)
        if method is None:
            raise NotImplementedError("Compression method not recognized for file '%s'" % filename)
        
        extracted = []
        with tarfile.open(filename, 'r:'+method) as tar:
            for member in tar.getmembers():
                fname = path.join(self.directory, member.name)
                if path.isfile(fname):
                    for e in extracted: path.remove(e)
                    raise FileExistsError(fname)
                if verbose: print("Extracting '%s' to '%s'" % (fname, self.directory))
                try:
                    tar.extract(member, path=self.directory)
                except:
                    if path.isfile(fname): path.remove(fname)
                    for e in extracted: path.remove(e)
                    raise
                if member.name == 'compression.sstools':
                    path.remove(fname)
                else:
                    extracted += [fname]
        if delete: path.remove(filename)



    def get_size(self):
        """Returns the size of the simulation in bytes.

        Returns
        -------
        int
        """
        return starsmashertools.helpers.path.get_directory_size(self.directory)
    

    class OutputNotInSimulationError(Exception):
        def __init__(self, simulation, output, message=None):
            if message is None:
                message = "Output is not a member of simulation: '%s' for simulation '%s'" % (output.path, simulation.directory)
            super(Simulation.OutputNotInSimulationError, self).__init__(message)


            

    



