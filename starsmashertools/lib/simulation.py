# These need to be imported at the top because they are used in parts of
# parameter annotations. Everything else should be loaded dynamically to avoid
# errors.
import starsmashertools.lib.units
import starsmashertools.helpers.argumentenforcer
from starsmashertools.helpers.apidecorator import api
from starsmashertools.helpers.clidecorator import cli
import numpy as np

try:
    import matplotlib
    has_matplotlib = True
except ImportError:
    has_matplotlib = False

class Simulation(object):
    ############################################################################
    # private attributes

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(self, directory : str):
        import starsmashertools.lib.input
        import starsmashertools.lib.output
        import starsmashertools.helpers.path
        import starsmashertools.lib.archive
        import starsmashertools.helpers.asynchronous

        # Prepare the directory string
        directory = starsmashertools.helpers.path.realpath(directory.strip())
        
        # Validate the directory
        if not Simulation.valid_directory(directory):
            raise Simulation.InvalidDirectoryError(directory)

        self.directory = directory

        super(Simulation, self).__init__()
        
        self.input = starsmashertools.lib.input.Input(self.directory)
        self._children = None
        self._units = None
        self._teos = None
        self._logfiles = None
        self._compression_task = None
        
        self.reader = starsmashertools.lib.output.Reader(self)

        #if starsmashertools.helpers.asynchronous.is_main_process():
        self.archive = starsmashertools.lib.archive.SimulationArchive(self)
        #else: self.archive = None
            
    def __hash__(self):
        return hash(self.directory)

    """
    # For pickling
    def __getstate__(self):
        return {'directory' : self.directory}

    def __setstate__(self, data):
        self.__init__(data['directory'])
    """

    @property
    def joined_simulations(self):
        import starsmashertools.helpers.path
        import starsmashertools
        import starsmashertools.helpers.warnings
        
        if 'joined simulations' not in self.archive: return []
        simulations = []
        for directory in self.archive['joined simulations'].value:
            directory = starsmashertools.helpers.path.join(
                self.directory,
                directory,
            )
            try:
                simulations += [starsmashertools.get_simulation(directory)]
            except Simulation.InvalidDirectoryError as e:
                starsmashertools.helpers.warnings.warn("A joined simulation is no longer a valid directory, likely because it was moved on the file system. Please split the simulation and re-join it to quell this warning. '%s'" % directory)
        return simulations
    
    @api
    def __eq__(self, other):
        # Check if the basenames are the same, so that e.g. pdc.json
        # files can work on different file systems
        import starsmashertools.helpers.path
        
        if not isinstance(other, Simulation): return False
        return starsmashertools.helpers.path.samefile(self.directory, other.directory)

    @api
    def __getitem__(self, key): return self.input[key]

    @api
    def __contains__(self, item):
        import starsmashertools.helpers.path
        import starsmashertools.lib.output

        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'item' : [str, starsmashertools.lib.output.Output, starsmashertools.lib.output.OutputIterator],
        })

        iterator = self.get_output_iterator()
        if isinstance(item, str):
            if not starsmashertools.helpers.path.isfile(item): return False
        
        if isinstance(item, starsmashertools.lib.output.OutputIterator):
            for filename in item.filenames:
                if filename not in iterator: return False
            return True
        
        return item in iterator

    # Override this in children. Must return a list of Simulation objects
    def _get_children(self):
        raise NotImplementedError

    def _load_children_from_hint_files(self):
        import starsmashertools.preferences
        import starsmashertools.helpers.file
        import starsmashertools.helpers.path
        import starsmashertools
        
        children = None
        hint_filename = starsmashertools.preferences.get_default(
            'Simulation', 'children hint filename', throw_error=True)
        fname = self.get_file(hint_filename, recursive = False)
        if not fname: return children # If the file wasn't found

        fname = fname[0]
        
        with starsmashertools.helpers.file.open(fname, 'r', lock = False) as f:
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
                        import starsmashertools.helpers.warnings
                        starsmashertools.helpers.warnings.warn("Failed to find directory in children hint file '%s': '%s'" % (fname, line))
                    else: raise

                if simulation is not None:
                    if children is None: children = []
                    children += [simulation]
        
        if children is not None:
            import starsmashertools.lib.binary
            if isinstance(self, starsmashertools.lib.binary.Binary):
                if len(children) != 2:
                    raise Exception("File '%s' indicates that the binary simulation has only one child. If one of the stars is a point mass, please add a line 'point mass' to the file." % fname)
        return children

    def _save_children_to_hint_file(self, children, verbose : bool = False):
        import starsmashertools.preferences
        import starsmashertools.helpers.file
        import starsmashertools.helpers.path

        hint_filename = starsmashertools.preferences.get_default(
            'Simulation', 'children hint filename', throw_error=True)
        fname = starsmashertools.helpers.path.join(
            self.directory,
            hint_filename,
        )
        towrite = "\n".join([child if isinstance(child, str) else child.directory for child in children])
        with starsmashertools.helpers.file.open(fname, 'w', lock = False) as f:
            f.write(towrite)
    
    def _get_compression_filename(self):
        import starsmashertools.helpers.path
        dirname = starsmashertools.helpers.path.basename(self.directory)
        return starsmashertools.helpers.path.join(self.directory, dirname+".zip")
        
    def _get_sphinit_filename(self):
        # Obtain the 'sph.init' file this simulation uses by checking the
        # StarSmasher source code for the filename that gets read.
        import starsmashertools.helpers.path
        import starsmashertools.helpers.file
        
        src = starsmashertools.helpers.path.get_src(self.directory)
        initfile = starsmashertools.helpers.path.join(src, 'init.f')
        last_open_line = None
        with starsmashertools.helpers.file.open(initfile, 'r', lock = False) as f:
            for line in f:
                ls = line.strip()
                if ls.startswith('open('): last_open_line = line
                if ls == 'read(12,initt)': break
            else:
                raise Exception("Failed to find the init file name because a line with content 'read(12,initt)' was not found in '%s'" % initfile)

        if last_open_line is None:
            raise Exception("Failed to find a line where the init file gets opened in '%s'" % initfile)

        for s in last_open_line.split(','):
            if s.startswith('file') and '=' in s:
                return s.split('=')[1].replace("'",'').replace('"','')

        raise Exception("Something went wrong while parsing the Fortran code to find the init file name in '%s'" % initfile)
            
            
    







        
    ############################################################################
    # public attributes

    ###
    # Exceptions
    ###

    class InvalidDirectoryError(Exception): pass
    
    class OutputNotInSimulationError(Exception):
        def __init__(self, simulation, output, message=None):
            if message is None:
                message = "Output is not a member of simulation: '%s' for simulation '%s'" % (output.path, simulation.directory)
            super(Simulation.OutputNotInSimulationError, self).__init__(message)


    
    ###
    # Object functionality
    ###
    
    @api
    def keys(self, *args, **kwargs): return self.input.keys(*args, **kwargs)
    @api
    def values(self, *args, **kwargs): return self.input.values(*args, **kwargs)
    @api
    def items(self, *args, **kwargs): return self.input.items(*args, **kwargs)

    @property
    def isRelaxation(self):
        import starsmashertools.lib.relaxation
        return isinstance(self, starsmashertools.lib.relaxation.Relaxation)

    @property
    def isBinary(self):
        import starsmashertools.lib.binary
        return isinstance(self, starsmashertools.lib.binary.Binary)

    @property
    def isDynamical(self):
        import starsmashertools.lib.dynamical
        return isinstance(self, starsmashertools.lib.dynamical.Dynamical)
    
    @property
    def units(self):
        if self._units is None: self._units = starsmashertools.lib.units.Units(self)
        return self._units

    @property
    def teos(self):
        if self._teos is None and self['neos'] == 2:
            import starsmashertools.helpers.path
            import starsmashertools.lib.table
            self._teos = starsmashertools.lib.table.TEOS(
                starsmashertools.helpers.path.realpath(
                    starsmashertools.helpers.path.join(
                        self.directory,
                        self['eosfile'],
                    )
                )
            )
        return self._teos

    @property
    def compressed(self):
        import starsmashertools.helpers.path
        import starsmashertools.helpers.compressiontask
        filename = self._get_compression_filename()
        if not starsmashertools.helpers.path.isfile(filename): return False
        return starsmashertools.helpers.compressiontask.CompressionTask.isCompressedFile(filename)
        
    @api
    def get_search_directory(self, **kwargs):
        """
        Get the default search directory from
        :py:property:`~.preferences.defaults`.
        
        Other Parameters
        ----------------
        **kwargs
            Keywords are passed directly to
            :func:`starsmashertools.preferences.get_default`.

        Returns
        -------
        str
            The "realpath" (:func:`os.path.realpath`) of the default search
            directory.
        """
        import starsmashertools.preferences
        import starsmashertools.helpers.path
        search_directory = starsmashertools.preferences.get_default(
            'Simulation',
            'search directory',
            **kwargs,
        )
        return starsmashertools.helpers.path.realpath(search_directory)
    
    @api
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @staticmethod
    def compare_type(
            sim1 : "type | starsmashertools.lib.simulation.Simulation",
            sim2 : "type | starsmashertools.lib.simulation.Simulation",
    ):
        """
        Return True if "sim1" is the same simulation type as "sim2", and False
        otherwise.
        
        Parameters
        ----------
        sim1 : :class:`starsmashertools.lib.simulation.Simulation`
        sim2 : :class:`starsmashertools.lib.simulation.Simulation`

        Returns
        -------
        bool
        """
        t1 = sim1
        t2 = sim2
        if isinstance(t1, type) and not isinstance(t2, type):
            return isinstance(t2, t1)
        if not isinstance(t1, type) and isinstance(t2, type):
            return isinstance(t1, t2)
        if not isinstance(t1, type) and not isinstance(t2, type):
            return isinstance(t1, type(t2))
        if isinstance(t1, type) and isinstance(t2, type):
            return t1 is t2
        raise Exception("This should never happen")
    
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
        import starsmashertools.helpers.compressiontask
        if not self.compressed: return {}
        filename = self._get_compression_filename()
        return starsmashertools.helpers.compressiontask.CompressionTask.get_compressed_properties(filename)

    
    
    ###
    # Files and directories
    ###

    @staticmethod
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def valid_directory(directory : str):
        import starsmashertools.helpers.path
        try:
            return starsmashertools.helpers.path.get_src(directory) is not None
        except FileNotFoundError as e:
            raise Simulation.InvalidDirectoryError() from e
    
    @api
    def get_logfiles(
            self,
            include_joined : bool = True,
            **kwargs
    ):
        """
        Get a list of :class:`starsmashertools.lib.logfile.LogFile`, sorted by
        oldest-to-newest, including those in subdirectories in the simulation
        directory.

        Other Parameters
        ----------------
        **kwargs
            Passed directly to :func:`starsmashertools.lib.logfile.find`.

        include_joined : bool, default = True
            If `True`, the joined simulations will be included in the search.
            Otherwise, only this simulation will be searched.

        Returns
        -------
        list
            A list of :class:`starsmashertools.lib.logfile.LogFile` sorted from
            oldest-to-newest by system modification times.
        """
        import starsmashertools.lib.logfile
        import starsmashertools.helpers.path
        
        if self._logfiles is None:
            self._logfiles = []
            for _path in starsmashertools.lib.logfile.find(self.directory, **kwargs):
                if starsmashertools.helpers.path.getsize(_path) > 0:
                    self._logfiles += [starsmashertools.lib.logfile.LogFile(_path, self)]

        all_files = [logfile for logfile in self._logfiles]
        if include_joined:
            for simulation in self.joined_simulations:
                all_files += simulation.get_logfiles(
                    include_joined = False,
                    **kwargs
                )
        
        if all_files: # Sort the log files by modification times
            modtimes = [starsmashertools.helpers.path.getmtime(p.path) for p in all_files]
            self._logfiles = [x for _, x in sorted(zip(modtimes, all_files), key=lambda pair: pair[0])]
        
        return all_files
    
    @api
    @cli('starsmashertools')
    def get_children(
            self,
            verbose : bool = False,
            cli : bool = False,
    ):
        """
        Return a list of `starsmashertools.lib.simulation.Simulation` objects
        that were used to create this simulation. For example, if this
        simulation is a dynamical simulation (nrelax = 0), then it has one
        child, which is the binary scan that it originated from. Similarly, a
        binary scan has two children, which are each the stellar relaxations
        that make up the two stars in the binary.

        If the simulation's directory contains a file with the name specified in
        :py:property:`~.preferences.defaults` (key 'children hint filename'; 
        default = 'children.sstools') then it will be read as a text file where
        each line is the path to a different child simulation directory. If that
        file doesn't exist then it will be created after the children have been
        found for the first time.

        This function acts as a wrapper for `~._get_children`, which is
        overridden in subclasses of Simulation.
        
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

            # Get the children from the hint files
            self._children = self._load_children_from_hint_files()
            
            # If loading the children didn't work, or we are doing multiple
            # processes
            if self._children is None:
                if verbose: print("Searching for child simulations for the first time for '%s'" % self.directory)
                
                self._children = self._get_children()
                
                if self._children is None:
                    raise Exception("Children found was 'None'. If you want to specify that a Simulation has no children, then its get_children() method should return empty list '[]', not 'None'.")
                
                # Now save the children (only if we aren't doing multiple
                # processes)
                self._save_children_to_hint_file(
                    self._children,
                    verbose=verbose,
                )

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

        Only returns files belonging to this simulation's directory, and ignores
        any joined simulations' directories.

        Parameters
        ----------
        filename_or_pattern : str
            A file name in the simulation directory or a pattern (e.g., 
            ``out*.sph``)
        recursive : bool, default = True
            If `True`, searches all sub directories in the simulation. If
            `False`, searches just the top level simulation directory.

        Returns
        -------
        `list`
            A list of paths which match the given pattern or filenames.
        """
        import starsmashertools.helpers.path
        import glob
        if recursive: _path = starsmashertools.helpers.path.join(self.directory, '**', filename_or_pattern)
        else: _path = starsmashertools.helpers.path.join(self.directory, filename_or_pattern)
        return glob.glob(_path, recursive=recursive)

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_outputfiles(
            self,
            pattern : str | type(None) = None,
            include_joined : bool = True,
    ):
        """
        Returns a list of output file locations. If this simulation has any
        joined simulations, then each of those simulations' output file paths
        are returned as well. The result is sorted by file modification times.

        Parameters
        ----------
        pattern : str, None, default = None
            Passed directly to :meth:`~.get_file`.
        
        include_joined : bool, default = True
            If `True`, the joined simulations will be included in the search.
            Otherwise, only this simulation will be searched.
        
        Returns
        -------
        list
            A list of output file paths belonging to this simulation and any
            joined simulations, sorted by file modification times.
        """
        import starsmashertools.preferences
        import starsmashertools.helpers.path
        
        if pattern is None:
            pattern = starsmashertools.preferences.get_default('Simulation', 'output files', throw_error=True)
        matches = self.get_file(pattern)

        if include_joined:
            for simulation in self.joined_simulations:
                matches += simulation.get_outputfiles(
                    pattern = pattern,
                    include_joined = False, # Prevent infinite recursion
                )

        # Sort by file modification times
        mtimes = [starsmashertools.helpers.path.getmtime(f) for f in matches]
        return [x for _,x in sorted(zip(mtimes, matches), key=lambda pair: pair[0])]
    
    # The initial file is always the one that was written first.
    @api
    def get_initialfile(self, **kwargs):
        """
        Obtain the output file which was written first, as a 
        :class:`~.lib.output.Output` object. Equivalent to calling
        :meth:`~.get_outputfiles` and getting the first index.

        Other Parameters
        ----------------
        kwargs
            Keyword arguments are passed directly to :meth:`~.get_outputfiles`.
        """
        return self.get_outputfiles(**kwargs)[0]

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

        Does not compress any joined simulations.

        Parameters
        ----------
        filename : str, None, default = None
            The name of the resulting compressed file. If `None` then the name
            of the file will be the name of the simulation directory with
            '.zip' on the end.
        
        include_patterns : list, None, default = None
            File name patterns to include in the compression. If `None`, uses 
            the "compress include" value in 
            :py:property:`~.preferences.defaults`.

        exclude_patterns : list, None, default = None
            File name patterns to exclude in the compression. If `None`, uses
            the "compress exclude" value from 
            :py:property:`~.preferences.defaults`.

        recursive : bool, default = True
            If `True`, subdirectories are also searched for files matching the
            given patterns. If `False`, only searches the main simulation
            directory.

        delete : bool, default = True
            If `True`, the files which are compressed are deleted.

        delete_after : bool, default = True
            If `True`, compressed files are deleted only after all files have
            been compressed. If `False`, each file is deleted after it has
            been compressed. If ``delete = False`` this option is ignored.

        Other Parameters
        ----------------
        **kwargs
            Remaining keyword arguments are passed directly to 
            :func:`~helpers.compressiontask.CompressionTask.compress`.

        See Also
        --------
        :func:`decompress`
        :func:`~.helpers.compressiontask.CompressionTask.compress`
        :py:property:`~.preferences.defaults`
        """
        import starsmashertools.preferences
        import starsmashertools.helpers.path
        import starsmashertools.helpers.compressiontask

        # Obtain the file names to be compressed.
        if include_patterns is None:
            include_patterns = starsmashertools.preferences.get_default('Simulation', 'compress include')
        if exclude_patterns is None:
            exclude_patterns = starsmashertools.preferences.get_default('Simulation', 'compress exclude')

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
            _path = starsmashertools.helpers.path.join(self.directory, val)
            if not starsmashertools.helpers.path.isfile(_path): continue
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

        Does not decompress any joined simulations.

        Parameters
        ----------
        filename : str, None default = None
            The filename to decompress. If `None`, the simulation directory is
            searched for a compressed file whose path ends with '.zip'.
            The file chosen is one which has a ``compression.sstools`` file
            included in it, which created by
            :func:`~helpers.compressiontask.CompressionTask.compress`.

        delete : bool, default = True
            If `True`, the compressed file is deleted after decompression.
        
        Other Parameters
        ----------------
        **kwargs
            Remaining keyword arguments are passed directly to
            `~helpers.compressiontask.CompressionTask.decompress`.

        See Also
        --------
        :func:`compress`
        :func:`~.helpers.compresisontask.CompressionTask.decompress`
        """
        import starsmashertools.helpers.compressiontask

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
        """
        Returns
        -------
        int
            The size of the simulation in bytes. Does not include any joined
            simulations.
        """
        import starsmashertools.helpers.path
        return starsmashertools.helpers.path.get_directory_size(self.directory)






    ###
    # Simulation outputs
    ###
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_output_iterator(
            self,
            start : int | type(None) = None,
            stop : int | type(None) = None,
            step : int | type(None) = None,
            include_joined : bool = True,
            **kwargs
    ):
        """
        Return a :class:`starsmashertools.lib.output.OutputIterator` containing
        all the :class:`starsmashertools.lib.output.Output` objects present in
        this simulation.

        The iterator includes output files from any joined simulations.
        
        Parameters
        ----------
        start : int, None, default = None
            Passed directly to :meth:`~.get_output`.

        stop : int, None, default = None
            Passed directly to :meth:`~.get_output`.

        step : int, None, default = None
            Passed directly to :meth:`~.get_output`.

        include_joined : bool, default = True
            If `True`, the joined simulations will be included in the search.
            Otherwise, only this simulation will be searched. Passed directly to
            :meth:`~.get_output`.

        Other Parameters
        ----------------
        **kwargs
            Other keyword arguments are passed directly to 
            :class:`starsmashertools.lib.output.OutputIterator`.
        
        Returns
        -------
        iterator
            The created :class:`starsmashertools.lib.output.OutputIterator` 
            instance.
        """
        import starsmashertools.lib.output
        # Now that we have all the file names, we can create an output iterator
        # from them
        outputs = self.get_output(
            start = start, stop = stop, step = step,
            include_joined = include_joined,
        )
        if not isinstance(outputs, list): outputs = [outputs]
        filenames = [output.path for output in outputs]
        return starsmashertools.lib.output.OutputIterator(filenames, self, **kwargs)

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_output_at_time(
            self,
            time : int | float | starsmashertools.lib.units.Unit,
            include_joined : bool = True,
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

        include_joined : bool, default = True
            If `True`, the joined simulations will be included in the search.
            Otherwise, only this simulation will be searched.

        Returns
        -------
        `starsmashertools.lib.output.Output`
            The Output object closest to the given time.
        """
        import starsmashertools.helpers.midpoint
        import starsmashertools.helpers.string
        
        if not isinstance(time, starsmashertools.lib.units.Unit):
            time *= self.units['t']
            #time = float(time.convert(self.units['t'].label))

        outputs = self.get_output(include_joined = include_joined)
        if not outputs: raise ValueError("Cannot find output at time %f because there are no output files in simulation '%s'" % (time, str(self.directory)))

        conversion = self.units['t'].convert(time.label)
        
        tend = outputs[-1]['t'] * conversion
        
        if time < 0 or time > tend:
            raise ValueError("Time %s is out of bounds [0, %s]" % (str(time), str(tend)))
        
        m = starsmashertools.helpers.midpoint.Midpoint(outputs)
        m.set_criteria(
            lambda output: output['t'] * conversion < time,
            lambda output: output['t'] * conversion == time,
            lambda output: output['t'] * conversion > time,
        )

        return m.get()

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    @cli('starsmashertools')
    def get_output(
            self,
            start : int | type(None) = None,
            stop : int | type(None) = None,
            step : int | type(None) = None,
            times : int | float | starsmashertools.lib.units.Unit | list | tuple | np.ndarray | type(None) = None,
            indices : list | tuple | np.ndarray | type(None) = None,
            include_joined : bool = True,
            cli : bool = False,
    ):
        """
        Obtain a list of :class:`starsmashertools.lib.output.Output` objects 
        associated with this simulation. Returns all outputs if no arguments are
        specified.
        
        Parameters
        ----------
        start : int, None, default = None
            The starting index of a slice of the list of all output files.
        
        stop : int, None, default = None
            The ending index in the slice of the list of all output files.
        
        step : int, None, default = None
            How many output files to skip in the slice.
        
        times : int, float, :class:`starsmashertools.lib.units.Unit`, list, tuple, np.ndarray, default = None
            If given, returns the output files that are closest to the given
            simulation time or collection of times. This can possibly include
            duplicate items.
        
        indices : list, tuple, np.ndarray, default = None
            If given, returns the output files at each index from the result of
            :meth:`~.get_outputfiles()`.

        include_joined : bool, default = True
            If `True`, the joined simulations will be included in the search.
            Otherwise, only this simulation will be searched.
        
        Returns
        -------
        list, `starsmashertools.lib.output.Output`
            A list of :class:`~.lib.output.Output` objects. If the list contains
            only a single item, that item is returned instead.
        """
        import starsmashertools.lib.output
        filenames = self.get_outputfiles(include_joined = include_joined)
        filenames = np.asarray(filenames, dtype=object)
        if times is not None:
            starsmashertools.helpers.argumentenforcer.enforcevalues({
                'start' : [None],
                'stop' : [None],
                'step' : [None],
                'indices' : [None],
            })
            if hasattr(times, '__iter__'):
                ret = [self.get_output_at_time(time, include_joined = include_joined) for time in times]
                if len(ret) == 1: return ret[0]
                return ret
            else:
                return self.get_output_at_time(times, include_joined = include_joined)
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

        if cli:
            import starsmashertools.bintools.page
            if len(ret) == 1: return ret[0].get_formatted_string('cli')
            return starsmashertools.bintools.page.PAGEBREAK.join([r.get_formatted_string('cli') for r in ret])

        if len(ret) == 1: return ret[0]
        return ret

    @api
    def get_output_headers(self, **kwargs):
        """
        Read all the headers of the output files in this simulation and return
        them as a dictionary.

        Parameters
        ----------
        keys : list
            The keys to query the output file headers for.

        Other Parameters
        ----------------
        kwargs : dict
            Extra keyword arguments passed to :meth:`~.get_output_iterator`. 
            Note that keyword ``return_headers`` is always `True` and 
            ``return_data`` is always `False`. Keyword ``asynchronous`` is set 
            to `False` by default because reading headers is often faster that 
            way.

        Returns
        -------
        `dict`
            Each key is a :class:`starsmashertools.lib.output.Output` object and
            each value is the entire header.
        """
        kwargs['return_headers'] = True
        kwargs['return_data'] = False
        kwargs['asynchronous'] = kwargs.get('asynchronous', False)
        iterator = self.get_output_iterator(**kwargs)
        ret = {}
        for output in iterator:
            ret[output] = output.header
        return ret

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_energyfiles(
            self,
            skip_rows : int | type(None) = None,
            include_joined : bool = True,
    ):
        """
        Get the :class:`~.lib.energyfile.EnergyFile` objects associated with
        this simulation.

        Other Parameters
        ----------------
        skip_rows : int, None, default = None
            Only read every Nth line of each energy file. If `None`, uses the 
            default value in :func:`~.lib.energyfile.EnergyFile.__init__`.

        include_joined : bool, default = True
            If `True`, the joined simulations will be included in the search.
            Otherwise, only this simulation will be searched.

        Returns
        -------
        list
            A list of :class:`~.lib.energyfile.EnergyFile` objects.
        """
        import starsmashertools.lib.energyfile
        energyfiles = []
        for logfile in self.get_logfiles(include_joined = include_joined):
            energyfiles += [starsmashertools.lib.energyfile.EnergyFile(
                logfile,
                skip_rows = skip_rows,
            )]
        return energyfiles
        

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_energy(
            self,
            sort : str | type(None) = None,
            skip_rows : int | type(None) = None,
            include_joined : bool = True,
    ):
        """
        Obtain all the simulation energies as a function of simulation time.
        
        Other Parameters
        ----------------
        sort : str, None, default = None
            Sort the resulting dictionary by the given string key. Raises an 
            error if that key doesn't exist.

        skip_rows : int, None, default = None
            Only read every Nth line of each energy file. If `None`, uses the 
            default value in :func:`~.lib.energyfile.EnergyFile.__init__`.

        include_joined : bool, default = True
            If `True`, the joined simulations will be included in the search.
            Otherwise, only this simulation will be searched.

        Returns
        -------
        dict or None
            A dictionary of NumPy arrays. If no energy files were found, returns
            `None` instead.
        """
        energyfiles = self.get_energyfiles(
            skip_rows = skip_rows,
            include_joined = include_joined,
        )
        if not energyfiles: return None

        expected_keys = energyfiles[0].keys()
        from_different_simulation = []
        for energyfile in energyfiles:
            for key in energyfile.keys():
                if key in expected_keys: continue
                from_different_simulation += [energyfile]
                break
        if len(from_different_simulation) != 0:
            raise Exception("The following energy files were detected as being from a different simulation: " + str(from_different_simulation))
        
        result = {key:[] for key in expected_keys}
        for energyfile in energyfiles:
            for key, val in energyfile.items():
                result[key] += val.tolist()

        for key, val in result.items():
            result[key] = np.asarray(val)

        if sort is not None:
            if sort not in result.keys():
                raise Exception("Cannot sort by key '%s' because it is not one of the keys in the dictionary. Possible keys are: " + ", ".join(["'%s'" % key for key in result.keys()]))
            values = result[sort]
            idx = np.argsort(values)
            for key, val in result.items():
                result[key] = val[idx]
        
        return result
        


    if has_matplotlib:
        @api
        @cli('starsmashertools')
        def plot_energy(
                self,
                cli : bool = False,
                scale : list | tuple | np.ndarray = (1., 1.5),
                **kwargs
        ):
            import starsmashertools.mpl.figure

            # Read all the energy*.sph files
            energies = self.get_energy(sort = 't')

            kwargs['sharex'] = kwargs.get('sharex', True)
            kwargs['nrows'] = len(energies.keys()) - 1
            fig, ax = starsmashertools.mpl.figure.subplots(
                scale = scale,
                **kwargs
            )

            tunit = (np.amax(energies['t']) * self.units.time).auto()
            t = energies['t'] / float(tunit)

            ax[-1].set_xlabel("Time [%s]" % tunit.label)

            keys = list(energies.keys())
            keys.remove('t')
            for a, key in zip(ax, keys):
                val = energies[key]
                a.plot(t, val * self.units.energy)
                a.set_ylabel(key)

            fig.align_ylabels(ax)

            if cli:
                fig.show()
                return ""
            return fig

        @api
        @cli('starsmashertools')
        def plot_animation(
                self,
                x : str = 'x',
                y : str = 'y',
                logx : bool = False,
                logy : bool = False,
                cli : bool = False,
        ):
            import starsmashertools.mpl.figure
            import matplotlib.animation
            import starsmashertools.helpers.string
            import starsmashertools.mpl.animation
            import matplotlib.text
            import copy

            fig, ax = starsmashertools.mpl.figure.subplots()
            if x in ['x','y','z'] and y in ['x','y','z']:
                ax.set_aspect('equal')

            xlabel = copy.deepcopy(x)
            ylabel = copy.deepcopy(y)

            if logx: xlabel = 'log '+xlabel
            if logy: ylabel = 'log '+ylabel
                
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
                
            artists = []
            output = self.get_output(0)
            total = len(self.get_outputfiles())
            
            artist = output.plot(ax, x=x, y=y)
            label = fig.text(
                0.01, 0.01,
                '',
                ha = 'left',
                va = 'bottom',
            )
            time_label = fig.text(
                0.01, 0.99,
                '',
                ha = 'left',
                va = 'top',
            )
            # determine the appropriate unit to use
            tunit = (self.get_output(-1).header['t'] * self.units['t']).auto().label
            def update(i):
                output = self.get_output(min(max(i, 0), total - 1))
                _x = output[x]
                _y = output[y]
                if logx: _x = np.log10(_x)
                if logy: _y = np.log10(_y)
                artist.set_offsets(np.column_stack((_x, _y)))
                label.set_text(str(output))
                time = (output['t'] * self.units['t']).convert(tunit)
                time_label.set_text('t = %10g %s' % (time.value, tunit))

            update(0)
            ani = starsmashertools.mpl.animation.Player(
                fig, update, maxi=total,
            )
            
            if cli:
                fig.show()
                return ""
            return ani


    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    @cli('starsmashertools')
    def join(
            self,
            other,
            cli : bool = False,
    ):
        """
        Merge two simulations of the same type together, by having
        starsmashertools retrieve the output files of all joined simulations
        whenever the ``include_joined`` flag is set to `True` (default) in
        :meth:`~.get_output`, :meth:`~.get_output_at_time`, 
        :meth:`~.get_output_iterator`, :meth:`~.get_outputfiles`, 
        :meth:`~.get_logfiles`, :meth:`~.get_energyfiles`, and 
        :meth:`~.get_energy`.

        To undo this operation, use :meth:`~.split`.
        
        Parameters
        ----------
        other : :class:`~.Simulation` | str
            The other simulation to merge with, or a path to a simulation
            directory.

        See Also
        --------
        :meth:`~.split`
        """
        import starsmashertools
        import starsmashertools.helpers.path

        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'other' : [Simulation, str],
        })
        
        if isinstance(other, str):
            other = starsmashertools.get_simulation(other)

        if self == other:
            message = "Cannot join simulations %s and %s because they are the same simulation" % (self, other)
            if cli: return message
            raise ValueError(message)

        other_path = starsmashertools.helpers.path.relpath(
            other.directory,
            start = self.directory,
        )
        our_path = starsmashertools.helpers.path.relpath(
            self.directory,
            start = other.directory,
        )

        if 'joined simulations' not in self.archive:            
            self.archive.add(
                'joined simulations',
                [other_path],
            )
        else:
            v = self.archive['joined simulations']
            if other_path not in v.value:
                v.value += [other_path]
            self.archive['joined simulations'] = v
        
        if 'joined simulations' not in other.archive:
            other.archive.add(
                'joined simulations',
                [our_path],
            )
        else:
            v = other.archive['joined simulations']
            if our_path not in v.value:
                v.value += [our_path]
            other.archive['joined simulations'] = v
        
        if cli: return "Success"


    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    @cli('starsmashertools')
    def split(
            self,
            which = None,
            cli : bool = False,
    ):
        """
        The opposite of :meth:`~.join`. Split this simulation apart from each
        simulation it is joined to. Any simulation joined to this one will also
        be split apart from it.

        Other Parameters
        ----------------
        which : str, :class:`~.Simulation`, None, default = None
            Which simulation to split from the joined simulations. If a `str` is
            given, it must match exactly one of the directory strings stored in
            this simulation's archive file. If a :class:`~.Simulation` is given,
            it must have a directory which matches one of those directory
            strings. Otherwise, if `None`, this simulation will be split apart 
            from all of its joined simulations.
        
        See Also
        --------
        :meth:`~.join`
        """
        import warnings
        import copy

        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'which' : [str, Simulation, type(None)],
        })

        joined_simulations = self.joined_simulations
        if not joined_simulations:
            if cli: return "This simulation has no joined simulations"
            return

        archive_value = self.archive['joined simulations']
        directories = copy.deepcopy(archive_value.value)

        to_update = []
        
        if isinstance(which, str):
            simulation = None
            
            if which not in directories:
                if cli: return ("No simulation identified by which = '%s'. The joined simulations are identified as:\n" + show_joined_simulations(cli = True)) % which
                else: raise KeyError("No simulation identified by which = '%s'" % which)

            directories.remove(which)
            to_update = [which]

        elif isinstance(which, Simulation):
            has_warnings = False
            simulations = []
            for directory in directories:
                try:
                    simulations += [starsmashertools.get_simulation(directory)]
                except Simulation.InvalidDirectoryError as e:
                    has_warnings = True
                    simulations += [None]
                    
            if which not in simulations:
                message = "No simulation identified by which = '%s'." % str(which)
                if has_warnings: message += " You have warnings related to the joined simulations. Try to resolve those first before trying again. If you are using the CLI, try using the 'show_joined_simulations' method and copying one of the strings to use as the value for 'which'."
                if cli: return message
                else: raise KeyError(message)

            index = simulations.index(which)
            del directories[index]
            to_update = [which.directory]

        elif which is None:
            to_update = copy.deepcopy(directories)
            directories = []
        else:
            raise NotImplementedError("Unrecognized input for keyword 'which': %s" % str(which))

        archive_value.value = directories
        self.archive['joined simulations'] = archive_value

        # Update the joined simulations' joined simulations (removing oursel
        for directory in to_update:
            try:
                simulation = starsmashertools.get_simulation(directory)
            except Simulation.InvalidDirectoryError:
                continue
            simulation.split(which = self)
        
        if cli: return "Success"
        
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @cli('starsmashertools')
    def show_joined_simulations(self, cli : bool = True):
        import starsmashertools.bintools
        newline = starsmashertools.bintools.Style.get('characters', 'newline')

        if 'joined simulations' not in self.archive:
            return "There are no joined simulations"
        return newline.join(self.archive['joined simulations'].value)

