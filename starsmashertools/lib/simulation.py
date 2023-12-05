# These need to be imported at the top because they are used in parts of
# parameter annotations. Everything else should be loaded dynamically to avoid
# errors.
import starsmashertools.lib.units
import starsmashertools.helpers.argumentenforcer
from starsmashertools.helpers.apidecorator import api
from starsmashertools.helpers.clidecorator import cli
import numpy as np

class Simulation(object):
    ############################################################################
    # private attributes

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(
            self,
            directory : str,
    ):
        import starsmashertools.lib.input
        import starsmashertools.lib.output
        import starsmashertools.helpers.path
        
        directory = starsmashertools.helpers.path.realpath(directory.strip())
        
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
        self._isContinuation = None

        self.reader = starsmashertools.lib.output.Reader(self)
            
    def __hash__(self):
        return hash(self.directory)

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
    def _get_children(self):
        raise NotImplementedError

    def _load_children_from_hint_files(self):
        import starsmashertools.preferences
        import starsmashertools.helpers.file
        import starsmashertools.helpers.path
        children = None
        hint_filename = starsmashertools.preferences.get_default(
            'Simulation', 'children hint filename', throw_error=True)
        fname = self.get_file(hint_filename, recursive = False)
        if not fname: return children # If the file wasn't found

        fname = fname[0]
        
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
                    raise Exception("File '%s' indicates that the binary simulation has only one child. If one of the stars is a point mass, please add a line 'point mass' to the file.")
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
        towrite = "\n".join([child.directory for child in children])
        with starsmashertools.helpers.file.open(fname, 'w') as f:
            for child in children:
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
        with starsmashertools.helpers.file.open(initfile, 'r') as f:
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
            import starsmashertools.lib.teos
            self._teos = starsmashertools.lib.teos.TEOS(
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

    @property
    def isContinuation(self):
        """
        True if this simulation is a continuation of another simulation. False
        otherwise.
        """
        if self._isContinuation is None:
            try:
                continued_from = self.get_simulation_continued_from()
            except Exception as e:
                if 'Failed to find the Simulation from which this Simulation was a continuation of' in str(e):
                    self._isContinuation = False
                else: raise(e)
        return self._isContinuation

    @api
    def get_search_directory(self, **kwargs):
        """
        Get the default search directory from `starsmashertools.preferences`.
        
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
            sim1 : "starsmashertools.lib.simulation.Simulation",
            sim2 : "starsmashertools.lib.simulation.Simulation",
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
        return isinstance(sim1, type(sim2))
        
    @api
    def get_simulation_continued_from(self, **kwargs):
        """
        Get the :class:`starsmashertools.lib.simulation.Simulation` from which
        this simulation was a continuation of. If this simulation is not a
        continuation, returns None.

        The default search directory from `starsmashertools.preferences` is
        checked for a duplicate file to this simulation's initial file (called 
        'restartrad.sph.orig' by default). If a duplicate is not found then an
        Exception is raised.

        In case the restartrad file was corrupted, we instead check the first
        output of this simulation and compare it to the other simulations we
        find in the search directory. We interpolate the particle positions to
        try to match the time stamps and then check if all the differences in
        particle positions are within some threshold value. You can set the
        threshold value in `starsmashertools.preferences`.

        Returns
        -------
        :class:`starsmashertools.lib.simulation.Simulation` or `None`
            The simulation from which this simulation continued from.
        """
        import starsmashertools.helpers.path
        import starsmashertools.lib.output

        search_directory = self.get_search_directory(throw_error = True)
        restartradfile = self.get_initialfile()
        duplicate = starsmashertools.helpers.path.find_duplicate_file(
                restartradfile, search_directory, throw_error=False)
        
        if duplicate is not None:
            dirname = starsmashertools.helpers.path.dirname(duplicate)
            return Simulation(dirname)

        # If we didn't find a duplicate file, then there's still a chance that
        # this output file came from a restartrad.sph file of a different
        # simulation, which may or may not have now been overwritten. We'll thus
        # check each simulation in the search directory for:
        #    1) If it's of the same type as this simulation
        #    2) If the number of particles is the same
        #    3) If it has output files near the time stamp of the restartrad
        #    4) If the particles in the nearest time stamp are nearly the same
        #       as those in the restartrad file
        
        all_directories = [f.path for f in starsmashertools.helpers.path.scandir(search_directory) if f.is_dir()]
        
        initial_output = starsmashertools.lib.output.Output(
            restartradfile,
            self,
        )
        try:
            t = initial_output['t']
        except starsmashertools.lib.output.Reader.CorruptedFileError as e:
            import starsmashertools.helpers.warnings
            message = "starsmashertools.lib.output.Reader.CorruptedFileError: "+str(e)+"\nUsing the first output file instead of the restartrad file"
            starsmashertools.helpers.warnings.warn(message)

            # If the restartrad file is corrupted, try to access just the very
            # first output file
            initial_output = self.get_output(0)
            t = initial_output['t']

        threshold = starsmashertools.preferences.get_default(
            'Simulation',
            'get_simulation_continued_from position threshold',
            throw_error = True,
        )
        
        t *= self.units['t']
        xyz = np.column_stack(( # Needs to be in cgs
            initial_output['x'] * float(self.units['x']),
            initial_output['y'] * float(self.units['y']),
            initial_output['z'] * float(self.units['z']),
        ))

        print(all_directories)
        
        ntot = initial_output.header['ntot']
        for directory in all_directories:
            try:
                simulation = starsmashertools.get_simulation(directory)
            except: continue

            if simulation == self: continue # Skip over ourself

            # Skip simulations of a different type
            if not Simulation.compare_type(self, simulation): continue
            
            # No output files
            if not simulation.get_outputfiles(): continue

            # Different number of particles
            if simulation.get_output(0).header['ntot'] != ntot: continue
            
            t0 = simulation.get_output(0).header['t'] * simulation.units['t']
            t1 = simulation.get_output(-1).header['t'] * simulation.units['t']
            # The restartrad file's time stamp is out of range for this
            # simulation
            if not (t0 <= t and t <= t1): continue
            
            closest_file = simulation.get_output(times = [t])
            _t = closest_file.header['t'] * simulation.units['t']
            dt = float((t - _t).convert('s')) # needs to be in cgs
            
            # Now we will 'interpolate' the particle positions from the time in
            # the closest_file to the time in the original file
            vxyz = np.column_stack(( # Needs to be in cgs
                closest_file['vx'] * float(simulation.units['vx']),
                closest_file['vy'] * float(simulation.units['vy']),
                closest_file['vz'] * float(simulation.units['vz']),
            ))
            _xyz = np.column_stack(( # Needs to be in cgs
                closest_file['x'] * float(simulation.units['x']),
                closest_file['y'] * float(simulation.units['y']),
                closest_file['z'] * float(simulation.units['z']),
            ))

            _xyz += vxyz * dt # 'interpolate'
            
            # compare
            print(simulation.directory)
            print("   ",np.amax(np.abs(xyz - _xyz)))
            if np.all(np.abs(xyz - _xyz) <= threshold):
                return simulation
        
        message = "Failed to find the Simulation from which this Simulation was a continuation of: '{simulation}'\nThis means that file '{initialfile}' is not a duplicate of any output file in any simulation in the search directory '{search_directory}'".format(
            simulation = self.directory,
            initialfile = starsmashertools.helpers.path.basename(restartradfile),
            search_directory = search_directory,
        )
        raise Exception(message)

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
        return starsmashertools.helpers.path.get_src(directory) is not None

    # Keywords are passed to logfile.find() method
    @api
    def get_logfiles(self, **kwargs):
        """
        Get a list of :class:`starsmashertools.lib.logfile.LogFile`, sorted by
        oldest-to-newest, including those in subdirectories in the simulation
        directory.

        Other Parameters
        ----------------
        **kwargs
            Passed directly to :func:`starsmashertools.lib.logfile.find`.

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
        if self._logfiles: # Sort the log files by modification times
            modtimes = [starsmashertools.helpers.path.getmtime(p.path) for p in self._logfiles]
            self._logfiles = [x for _, x in sorted(zip(modtimes, self._logfiles), key=lambda pair: pair[0])]
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

        If the simulation's directory contains a file with the name specified in
        `~.preferences` (key 'children hint filename'; default = 
        'children.sstools') then it will be read as a text file where each line
        is the path to a different child simulation directory. If that file
        doesn't exist then it will be created after the children have been found
        for the first time.

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
            
            # If loading the children didn't work
            if self._children is None:
                if verbose: print("Searching for child simulations for the first time for '%s'" % self.directory)
                
                self._children = self._get_children()
                
                if self._children is None:
                    raise Exception("Children found was 'None'. If you want to specify that a Simulation has no children, then its get_children() method should return empty list '[]', not 'None'.")
                
                # Now save the children
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
        """
        import starsmashertools.helpers.path
        import glob
        if recursive: _path = starsmashertools.helpers.path.join(self.directory, '**', filename_or_pattern)
        else: _path = starsmashertools.helpers.path.join(self.directory, filename_or_pattern)
        return glob.glob(_path, recursive=recursive)
        
    @api
    def get_outputfiles(self, pattern : str | type(None) = None):
        import starsmashertools.preferences
        if pattern is None:
            pattern = starsmashertools.preferences.get_default('Simulation', 'output files', throw_error=True)
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
            '.zip' on the end.
        
        include_patterns : list, None, default = None
            File name patterns to include in the compression. If `None`, uses 
            the "compress include" value in :mod:`~.preferences`.

        exclude_patterns : list, None, default = None
            File name patterns to exclude in the compression. If `None`, uses
            the "compress exclude" value from :mod:`~.preferences`.

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
        :mod:`~.preferences`
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
        """Returns the size of the simulation in bytes.

        Returns
        -------
        int
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
            `starsmashertools.lib.output.OutputIterator`.
        
        Returns
        -------
        iterator
            The created `starsmashertools.lib.output.OutputIterator` instance.
        """
        import starsmashertools.lib.output
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
        import starsmashertools.helpers.midpoint
        if not isinstance(time, starsmashertools.lib.units.Unit):
            time *= self.units['t']
            #time = float(time.convert(self.units['t'].label))

        outputs = self.get_output()
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
        import starsmashertools.lib.output
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








            

    



