# These need to be imported at the top because they are used in parts of
# parameter annotations. Everything else should be loaded dynamically to avoid
# errors.
import starsmashertools.preferences
from starsmashertools.preferences import Pref
import starsmashertools.lib.units
import starsmashertools.helpers.argumentenforcer
from starsmashertools.helpers.apidecorator import api
from starsmashertools.helpers.clidecorator import cli, clioptions
from starsmashertools.helpers.archiveddecorator import archived
import numpy as np
import typing

try:
    import matplotlib
    has_matplotlib = True
except ImportError:
    has_matplotlib = False

@starsmashertools.preferences.use
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
        
        # Prepare the directory string
        directory = starsmashertools.helpers.path.realpath(directory.strip())
        
        # Validate the directory
        if not Simulation.valid_directory(directory):
            raise Simulation.InvalidDirectoryError(directory)

        self.directory = directory

        super(Simulation, self).__init__()
        
        self.input = starsmashertools.lib.input.Input(self.directory)
        self._units = None
        self._teos = None
        self._logfiles = None
        self._compression_task = None
        
        self.reader = starsmashertools.lib.output.Reader(self)

        self.archive = starsmashertools.lib.archive.SimulationArchive(self)
            
    def __hash__(self):
        return hash(self.directory)

    def __json_view__(self):
        return self.directory

    def __reduce__(self): # For pickling
        return (self.__class__, (self.directory,))
    
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

    class JoinError(Exception):
        def __init__(self, simulation, message=None):
            if message is None:
                start_file = Simulation.preferences.get('start file')
                message = "Cannot join simulation '{sim1:s}' and '{sim2:s}' because neither simulations were started from one another. That is, '{start_file:s}' in one or both of these simulations does not originate from an output file in one of these simulations.".format(
                    sim1 = self.directory,
                    sim2 = simulation.directory,
                    start_file = start_file,
                )
            super(Simulation.JoinError, self).__init__(message)
    
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
    def get_search_directory(self):
        """
        Get the default search directory from the preferences.

        Returns
        -------
        str
            The :meth:`os.path.realpath` of the default search directory.
        """
        import starsmashertools.helpers.path
        search_directory = self.preferences.get('search directory')
        return starsmashertools.helpers.path.realpath(search_directory)
    
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
            raise Simulation.InvalidDirectoryError(directory) from e
    
    @api
    def get_logfiles(
            self,
            include_joined : bool = True,
            **kwargs
    ):
        """
        Get a list of :class:`~.logfile.LogFile`\, sorted by oldest-to-newest,
        including those in subdirectories in the simulation directory.

        Other Parameters
        ----------------
        **kwargs
            Passed directly to :meth:`~.logfile.find`\.

        include_joined : bool, default = True
            If `True`\, the joined simulations will be included in the search.
            Otherwise, only this simulation will be searched.

        Returns
        -------
        list
            A list of :class:`~.logfile.LogFile` sorted from oldest-to-newest by
            system modification times.
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

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    @cli('starsmashertools')
    @clioptions(display_name = 'Set children')
    def set_children(
            self,
            children : str | list | tuple,
            cli : bool = False,
    ):
        """
        Set the list of :class:`~.Simulation` objects that were used to create
        this simulation. This can save time on searching the file system for 
        child simulations.

        Parameters
        ----------
        children : list, tuple, str
            If a `list` or `tuple` is given, this simulation's children become
            the contents of ``children``\. Thus, each element must at least be of
            type :class:`~.Simulation`\. If a `str` or :class:`~.Simulation` is 
            given, then :meth:`starsmashertools.get_simulation` will be called
            if it is a `str`\, and then this simulation will have one child.

        See Also
        --------
        :meth:`~.get_children`
        """
        import starsmashertools
        import starsmashertools.helpers.path
        
        if isinstance(children, str):
            children = [starsmashertools.get_simulation(children)]
        if not isinstance(children, (list, tuple)): children = [children]
        if isinstance(children, tuple): children = list(children)

        val = []
        for child in children:
            if child in [None, 'point mass']:
                val += [child]
            else:
                val += [starsmashertools.helpers.path.relpath(
                    child.directory, start = self.directory,
                )]
        
        self.archive.add(
            'children',
            val,
            mtime = None,
        )

        if cli:
            string = ["Success:"]
            string += val
            return '\n'.join(string)
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    @cli('starsmashertools')
    @clioptions(display_name = 'Show children')
    def get_children(
            self,
            verbose : bool = False,
            cli : bool = False,
    ):
        """
        Return a list of :class:`~.Simulation` objects that were used to create
        this simulation. For example, if this simulation is a dynamical
        simulation (``nrelax = 0``), then it has one child, which is the binary
        scan that it originated from. Similarly, a binary scan has two children,
        which are each the stellar relaxations that make up the two stars in the
        binary.

        If the children are already known because they ahve been saved in this 
        simulation's :class:`~.archive.SimulationArchive`\, then the children
        will be recovered from the archive. Otherwise, :meth:`~._get_children`
        is called and the result is saved to the archive. You can edit the 
        children in the archive using the ``starsmashertools`` CLI by selecting
        :meth:`~.set_children` in the main menu. This can save time on searching
        the file system for child simulations.
        
        Parameters
        ----------
        verbose : bool, default = False
            If `True`\, debug messages will be printed to the console.
        
        Returns
        -------
        list
            A list of the child simulations.
        
        See Also
        --------
        :meth:`~.set_children`
        """
        import starsmashertools
        import starsmashertools.helpers.path
        
        if 'children' in self.archive.keys():
            children = []
            for directory in self.archive['children'].value:
                if directory in [None, 'point mass']:
                    children += [directory]
                else:
                    path = starsmashertools.helpers.path.join(
                        self.directory, directory,
                    )
                    children += [starsmashertools.get_simulation(path)]
        else:
            children = self._get_children()
            self.set_children(children)
        
        if cli:
            string = []
            for i, child in enumerate(children):
                if child in [None, 'point mass']:
                    string += ["Star %d: point mass" % (i+1)]
                else:
                    string += ["Star %d: %s" % (i+1, child.directory)]
            return "\n".join(string)
        return children

    @api
    def get_start_file(self):
        """
        Return the path to the output file that this simulation started from,
        as identified by ``'Simulation/start file'`` in 
        :mod:`~starsmashertools.preferences`\. If this simulation doesn't
        have a file matching that name, instead return the first output file 
        this simulation produced. If this simulation has not produced any output
        files, a :py:class:`FileNotFoundError` is raised.

        Returns
        -------
        str
            The path to the output file this simulation started from.
        """
        import starsmashertools.helpers.path
        start_file_identifier = self.preferences.get('start file')
        filename = self.get_file(start_file_identifier)
        if len(filename) != 1:
            outputfiles = self.get_outputfiles(include_joined = False)
            if outputfiles: return outputfiles[0]
            raise FileNotFoundError("Cannot get start file because simulation has no output files and no file named '%s': '%s'" % (start_file_identifier, self.directory))
        return filename[0]

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_start_time(
            self,
            include_joined : bool = True,
    ):
        """
        Other Parameters
        ----------------
        include_joined : bool, default = True
            If `True` then the simulations which are joined to this simulation
            will be considered. Otherwise only this simulation will be 
            considered.
        
        Returns
        -------
        :class:`~.units.Unit`
            The simulation time in seconds when this simulation started running.
        """
        import starsmashertools.lib.units
        files = self.get_outputfiles(include_joined = include_joined)
        if not files:
            raise FileNotFoundError("Simulation has no output files: '%s'" % self.directory)
        return starsmashertools.lib.units.Unit(
            self.reader.read_from_header('t', files[0])*float(self.units['t']),
            's',
        )

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_current_time(
            self,
            use_logfiles : bool = False,
            include_joined : bool = True,
    ):
        """
        Obtain the time which the :class:`~.Simulation` is currently at.
        
        Other Parameters
        ----------------
        use_logfiles : bool, default = False
            If `True`\, only the simulation's log files are used, which will in
            general return larger time values than when using just the output
            files. If `False`\, this method is equivalent to the value of 
            ``'t'`` in the header of the very last output file which was written
            in this simulation.

        include_joined : bool, default = True
            If `True` then the simulations which are joined to this simulation
            will be considered. Otherwise only this simulation will be 
            considered.

        Returns
        -------
        :class:`~.units.Unit`
            The simulation time in seconds when this simulation stopped running.
        """
        import starsmashertools.lib.units

        if use_logfiles:
            time = 0
            for logfile in self.get_logfiles(include_joined = include_joined):
                time = max(time, logfile.get_current_time())
        else:
            files = self.get_outputfiles(include_joined = include_joined)
            time = self.reader.read_from_header('t', files[-1])
        
        time *= float(self.units['t'])
        return starsmashertools.lib.units.Unit(time, 's')

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_stop_time(
            self,
            include_joined : bool = True,
    ):
        """
        Obtain the time at which the :class:`~.Simulation` is expected to stop 
        running. This is set by the ``'tf'`` parameter in the StarSmasher input 
        file ``sph.input``\.
        
        Other Parameters
        ----------------
        include_joined : bool, default = True
            If `True`\, the joined simulations will be included.

        Returns
        -------
        tf : :class:`~.units.Unit`
            The stop time for this Simulation.
        """
        import starsmashertools.lib.units
        tf = self['tf'] * self.units.time
        if include_joined:
            for simulation in self.joined_simulations:
                tf = max(tf, simulation['tf'] * simulation.units.time)
        return tf

    @api
    def started_from(self, simulation : 'starsmashertools.lib.simulation.Simulation'):
        """
        Parameters
        ----------
        simulation : :class:`~.Simulation`
        
        Returns
        -------
        bool
            `True` if this simulation was started from ``simulation``\, in the 
            sense that this simulation both has a "start file" (returned by
            :meth:`~.get_start_file`\) and the start file is identical to one of
            the output files in ``simulation``\.
        
        See Also
        --------
        :meth:`~.get_file`
        """
        import starsmashertools.helpers.path
        import filecmp
        
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'simulation' : [Simulation],
        })

        path = self.get_file(self.preferences.get('start file'))
        if len(path) != 1: return False
        path = path[0]

        outputfiles = simulation.get_outputfiles(include_joined = False)
        size1 = starsmashertools.helpers.path.getsize(path)
        for _file in outputfiles:
            size2 = starsmashertools.helpers.path.getsize(_file)
            if size1 != size2: continue
            if filecmp.cmp(path, _file, shallow = False): return True
        return False
        
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
            ``out*.sph``\)
        recursive : bool, default = True
            If `True`\, searches all sub directories in the simulation. If
            `False`\, searches just the top level simulation directory.

        Returns
        -------
        list
            A list of paths which match the given pattern or filenames.
        """
        import starsmashertools.helpers.path
        import glob
        if recursive: _path = starsmashertools.helpers.path.join(
                self.directory, '**', filename_or_pattern,
        )
        else: _path = starsmashertools.helpers.path.join(
                self.directory, filename_or_pattern,
        )
        return glob.glob(_path, recursive=recursive)

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_outputfiles(
            self,
            pattern : str = Pref('get_outputfiles.output files', 'out*.sph'),
            include_joined : bool = True,
    ):
        """
        Returns a list of output file locations. If this simulation has any
        joined simulations, then each of those simulations' output file paths
        are returned as well. The result is sorted by simulation times.
        
        For joined simulations, it is possible that one or more of the joined
        simulations are a "continuation" of this current simulation, or this
        current simulation is a "continuation" of one of its joined simulations.
        Here, a "continuation" is a simulation which was started by copying an
        output file from another simulation and calling it 
        ``'restartrad.sph.orig'``\, without changing the simulation type 
        (``'nrelax'`` in ``sph.input``\). Only the joined simulations will be 
        checked for a file which is identical to this simulation's 
        ``restartrad.sph.orig`` file (the actual name depends on the user's 
        preferences, in :mod:`~starsmashertools.preferences`\).
        
        For example, consider that simulation ``A`` produced output files 
        ``'A/out0000.sph'``\, ``'A/out0001.sph'``\, and ``'A/out0002.sph'``\. 
        Then, simulation ``B`` was created by copying ``'A/out0001.sph'`` and 
        renaming it as ``'B/restartrad.sph.orig'``\, and then 
        ``'B/restartrad.sph.orig'`` was copied to ``'B/restartrad.sph'`` before
        starting simulation ``B``\. Suppose then that simulation ``B`` creates 
        output files ``'B/out0001.sph'``\, ``'B/out0002.sph'``\, and 
        ``'B/out0003.sph'`` and that simulations ``A`` and ``B`` are joined. The
        expected output of this function would be 
        ``['A/out0000.sph', 'B/out0001.sph', 'B/out0002.sph', 'B/out0003.sph']``\. 
        Note that the actual files simulation ``B`` would create depends on its
        input parameters, such as ``'dtout'`` in ``'B/sph.input'``\.
        
        Parameters
        ----------
        pattern : str, default = Pref('get_outputfiles.output files', 'out*.sph')
            Passed directly to :meth:`~.get_file`\.
        
        include_joined : bool, default = True
            If `True`\, the joined simulations will be included in the search.
            Otherwise, only this simulation will be searched.
        
        Returns
        -------
        list
            A list of output file paths belonging to this simulation and any
            joined simulations, unsorted.
        """
        import starsmashertools.helpers.path
        import starsmashertools.helpers.midpoint
        import copy
        
        # We always need to get our own files first
        matches = self.get_file(pattern)

        # Always order our own files by their modification times, which should
        # correspond exactly with ordering by simulation times, except we don't
        # need to read the actual files this way.
        times = [starsmashertools.helpers.path.getmtime(m) for m in matches]
        matches = [x for _,x in sorted(zip(times, matches), key=lambda pair:pair[0])]
        
        if not include_joined: return matches
        
        # To simplify the thought process: consider we are simulation A, with
        # output files A/out0000.sph, A/out0001.sph, and A/out0002.sph, and we
        # are joined with simulation B, which has
        # restartrad.sph.orig == A/out0001.sph and output files B/out0001.sph,
        # B/out0002.sph, and B/out0003.sph. We want to return
        # [A/out0000.sph, B/out0001.sph, B/out0002.sph, B/out0003.sph].
        #
        # If simulation B doesn't originate from A, then we need to simply
        # return files in the order of the joined simulations. This will be done
        # only after the originating joined simulations have been handled first.

        # Because the output files are always written sequentially, the very
        # first match will have the smallest modification time. We use this fact
        # to determine the ordering of the joined simulations that did not
        # originate from us

        # Handle the joined simulations which originate from us
        we_started_from = [
            s for s in self.joined_simulations if self.started_from(s)
        ]
        started_from_us = [
            s for s in self.joined_simulations if s.started_from(self)
        ]

        # If there are any joined simulations which do not start from this
        # simulation, or which this simulation does not start from, then there
        # is a problem with the list of joined simulations.
        for simulation in self.joined_simulations:
            if simulation in we_started_from: continue
            if simulation in started_from_us: continue
            raise Simulation.JoinError(simulation)

        all_paths = []
        simulations = we_started_from + [self] + started_from_us
        if simulations:
            paths = []
            times = []
            for simulation in simulations:
                ms = simulation.get_file(pattern)
                ts = [starsmashertools.helpers.path.getmtime(m) for m in ms]
                ms = [x for _,x in sorted(zip(ts, ms), key=lambda pair:pair[0])]

                paths += [ms]
                times += [simulation.get_start_time(
                    include_joined = False,
                )]

            # Sorted by start times
            simulations = [x for _,x in sorted(zip(times, simulations), key = lambda pair: pair[0])]
            paths = [x for _,x in sorted(zip(times, paths), key=lambda pair:pair[0])]
            times = sorted(times)

            for i, (ps, s, t) in enumerate(zip(paths, simulations, times)):
                if i == len(simulations) - 1:
                    all_paths += ps
                else:
                    tahead = times[i + 1]
                    
                    m = starsmashertools.helpers.midpoint.Midpoint(ps)
                    m.set_criteria(
                        lambda p: s.reader.read_from_header('t', p) * s.units['t'] < tahead,
                        lambda p: s.reader.read_from_header('t', p) * s.units['t'] == tahead,
                        lambda p: s.reader.read_from_header('t', p) * s.units['t'] > tahead,
                    )
                    my_p, idx = m.get(favor = 'low', return_index = True)
                    all_paths += ps[:idx+1]
        return all_paths

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def compress(
            self,
            filename : str | type(None) = None,
            patterns : list = Pref('state files'),
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
            ``'.zip'`` on the end.
        
        patterns : list, default = Pref('state files')
            File name patterns to include in the compression.

        recursive : bool, default = True
            If `True`\, subdirectories are also searched for files matching the
            given patterns. If `False`\, only searches the main simulation
            directory.

        delete : bool, default = True
            If `True`\, the files which are compressed are deleted.

        delete_after : bool, default = True
            If `True`\, compressed files are deleted only after all files have
            been compressed. If `False`\, each file is deleted after it has
            been compressed. If ``delete = False`` this option is ignored.

        Other Parameters
        ----------------
        **kwargs
            Remaining keyword arguments are passed directly to 
            :meth:`starsmashertools.helpers.compressiontask.CompressionTask.compress`\.

        See Also
        --------
        :meth:`~.decompress`
        :meth:`starsmashertools.helpers.compressiontask.CompressionTask.compress`
        :mod:`starsmashertools.preferences`
        """
        import starsmashertools.helpers.path
        import starsmashertools.helpers.compressiontask
        
        files = []
        for pattern in patterns:
            files += self.get_file(pattern, recursive=recursive)
            
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
            The filename to decompress. If `None`\, the simulation directory is
            searched for a compressed file whose path ends with ``'.zip'``\.
            The file chosen is one which has a ``compression.sstools`` file
            included in it, which created by
            :meth:`starsmashertools.helpers.compressiontask.CompressionTask.compress`\.

        delete : bool, default = True
            If `True`\, the compressed file is deleted after decompression.
        
        Other Parameters
        ----------------
        **kwargs
            Remaining keyword arguments are passed directly to
            `starsmashertools.helpers.compressiontask.CompressionTask.decompress`\.

        See Also
        --------
        :meth:`compress`
        :meth:`starsmashertools.helpers.compresisontask.CompressionTask.decompress`
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
        Return a :class:`~.output.OutputIterator` containing all the 
        :class:`~.output.Output` objects present in this simulation.

        The iterator includes output files from any joined simulations.
        
        Parameters
        ----------
        start : int, None, default = None
            Passed directly to :meth:`~.get_output`\.

        stop : int, None, default = None
            Passed directly to :meth:`~.get_output`\.

        step : int, None, default = None
            Passed directly to :meth:`~.get_output`\.

        include_joined : bool, default = True
            If `True`\, the joined simulations will be included in the search.
            Otherwise, only this simulation will be searched. Passed directly to
            :meth:`~.get_output`\.

        Other Parameters
        ----------------
        **kwargs
            Other keyword arguments are passed directly to 
            :class:`~.output.OutputIterator`\.
        
        Returns
        -------
        iterator
            The created :class:`~.output.OutputIterator` instance.
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
        Return the :class:`~.output.Output` object of this simulation whose time
        best matches the given time. Raises a :py:class:`ValueError` if the 
        given time is out of bounds.
        
        Parameters
        ----------
        time : int, float, :class:`~.units.Unit`
            The simulation time to locate. If an `int` or `float` are given then
            this value represents time in the simulation units. If a 
            :class:`~.units.Unit` is given, it will be converted to this 
            simulation's time unit from :attr:`~.units`.

        include_joined : bool, default = True
            If `True`\, the joined simulations will be included in the search.
            Otherwise, only this simulation will be searched.

        Returns
        -------
        :class:`~.output.Output`
            The :class:`~.output.Output` object closest to the given time.
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
            lambda output: output.simulation.reader.read_from_header('t', output.path) * conversion < time,
            lambda output: output.simulation.reader.read_from_header('t', output.path) * conversion == time,
            lambda output: output.simulation.reader.read_from_header('t', output.path) * conversion > time,
        )
        
        return m.get()

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    @cli('starsmashertools')
    @clioptions(display_name = 'Show output files')
    def get_output(
            self,
            start : int | type(None) = None,
            stop : int | type(None) = None,
            step : int | type(None) = None,
            times : int | float | starsmashertools.lib.units.Unit | list | tuple | np.ndarray | type(None) = None,
            time_range : list | tuple | type(None) = None,
            indices : list | tuple | np.ndarray | type(None) = None,
            include_joined : bool = True,
            cli : bool = False,
    ):
        """
        Obtain a list of :class:`~.output.Output` objects associated with this
        simulation. Returns all outputs if no arguments are specified.
        
        Parameters
        ----------
        start : int, None, default = None
            The starting index of a slice of the list of all output files.
        
        stop : int, None, default = None
            The ending index in the slice of the list of all output files.
        
        step : int, None, default = None
            How many output files to skip in the slice.
        
        times : int, float, :class:`~.units.Unit`\, list, tuple, :class:`numpy.ndarray`\, default = None
            If given, returns the output files that are closest to the given
            simulation time or collection of times. This can possibly include
            duplicate items.

        time_range : list, tuple, None, default = None
            If not `None`\, this must be an iterable of 2 elements, the first 
            being the lower time bound and the second being the upper time 
            bound. If either bounds are `None` then only one bound will be used.
        
        indices : list, tuple, :class:`numpy.ndarray`\, default = None
            If given, returns the output files at each index from the result of
            :meth:`~.get_outputfiles`\.

        include_joined : bool, default = True
            If `True`\, the joined simulations will be included in the search.
            Otherwise, only this simulation will be searched.
        
        Returns
        -------
        list, :class:`~.output.Output`
            A list of :class:`~.output.Output` objects. If the list contains
            only a single item, that item is returned instead.
        """
        import starsmashertools.lib.output
        import starsmashertools.helpers.string
        filenames = self.get_outputfiles(include_joined = include_joined)
        filenames = np.asarray(filenames, dtype = object)

        # We accept the input if only either
        #    - (start, stop, step) are valid and not None, or
        #    - Exactly one of times, time_range, or indices are not None
        modes = {
            'times' : times is not None,
            'time_range' : time_range is not None,
            'indices' : indices is not None,
        }

        num_used = sum([1 if item else 0 for item in modes.values()])
        if num_used == 0: # Use slicing
            if start is not None and stop is None and step is None:
                # User is intending to just get a single index
                if start != -1:
                    stop = start + 1
            s = slice(start, stop, step)
            filenames = filenames.tolist()[s]
        elif modes['times']:
            if hasattr(times, '__iter__'):
                ret = [self.get_output_at_time(
                    time,
                    include_joined = include_joined,
                ) for time in times]
                if len(ret) == 1: return ret[0]
                return ret
            else:
                return self.get_output_at_time(
                    times,
                    include_joined = include_joined,
                )
        elif modes['indices']:
            indices = np.asarray(indices, dtype=int)
            filenames = filenames[indices]
        elif modes['time_range']:
            tlo, thi = time_range
            idx0, idx1 = 0, -1
            if tlo is not None:
                o = self.get_output_at_time(
                    tlo,
                    include_joined = include_joined
                )
                idx0 = filenames.tolist().index(o.path)
            if thi is not None:
                o = self.get_output_at_time(
                    thi,
                    include_joined = include_joined
                )
                idx1 = filenames.tolist().index(o.path)
            if idx0 == idx1: filenames = [filenames[0]]
            filenames = filenames[idx0:idx1]
        else:
            raise ValueError("No mode specified. Check your keyword arguments.")
        
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
            Extra keyword arguments passed to :meth:`~.get_output_iterator`\. 
            Note that keyword ``return_headers`` is always `True` and 
            ``return_data`` is always `False`\. Keyword ``asynchronous`` is set 
            to `False` by default because reading headers is often faster that 
            way.

        Returns
        -------
        dict
            Each key is a :class:`~.output.Output` object and each value is the
            entire header.
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
        Get the :class:`~.energyfile.EnergyFile` objects associated with this 
        simulation.

        Other Parameters
        ----------------
        skip_rows : int, None, default = None
            Only read every Nth line of each energy file. If `None`\, uses the 
            default value in :meth:`~.energyfile.EnergyFile.__init__`\.

        include_joined : bool, default = True
            If `True`\, the joined simulations will be included in the search.
            Otherwise, only this simulation will be searched.

        Returns
        -------
        list
            A list of :class:`~.energyfile.EnergyFile` objects.
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
            Only read every :math:`\\mathrm{N^{th}}` line of each energy file. 
            If `None`\, uses the default value in 
            :meth:`~.energyfile.EnergyFile.__init__`\.

        include_joined : bool, default = True
            If `True`\, the joined simulations will be included in the search.
            Otherwise, only this simulation will be searched.

        Returns
        -------
        dict or None
            A dictionary of :class:`numpy.ndarray`\. If no energy files were 
            found, returns `None` instead.
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

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_flux(
            self,
            outputs,
            parallel : bool = True,
            **kwargs
    ):
        import starsmashertools.lib.flux
        import starsmashertools.helpers.asynchronous

        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'outputs' : [
                starsmashertools.lib.output.Output,
                list,
                tuple,
                starsmashertools.lib.output.OutputIterator,
            ]
        })
        
        if isinstance(outputs, starsmashertools.lib.output.Output):
            return starsmashertools.lib.flux.get(outputs, **kwargs)
        
        if isinstance(outputs, (list, tuple)):
            for i, output in enumerate(outputs):
                if not isinstance(output, starsmashertools.lib.output.Output):
                    raise TypeError("All elements of the given outputs iterable must be of type 'starsmashertools.lib.output.Output', but received '%s' at element %d" % (type(output).__name__, i))

        if not parallel:
            #results = []
            for output in outputs:
                yield starsmashertools.lib.flux.get(output, **kwargs)
            #return results
        else:
            if isinstance(outputs, starsmashertools.lib.output.OutputIterator):
                outputs = outputs.tolist()
            return starsmashertools.helpers.asynchronous.ParallelIterator(
                starsmashertools.lib.flux.get,
                [[output] for output in outputs],
                kwargs = [kwargs],
            )


    if has_matplotlib:
        @api
        @cli('starsmashertools')
        @clioptions(display_name = 'Plot energies')
        def plot_energy(
                self,
                cli : bool = False,
                scale : list | tuple | np.ndarray = (1., 1.5),
                **kwargs
        ):
            import starsmashertools.mpl

            # Read all the energy*.sph files
            energies = self.get_energy(sort = 't')

            kwargs['sharex'] = kwargs.get('sharex', True)
            kwargs['nrows'] = len(energies.keys()) - 1
            fig, ax = starsmashertools.mpl.subplots(
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
        @clioptions(display_name = 'Plot animation')
        def plot_animation(
                self,
                x : str = 'x',
                y : str = 'y',
                logx : bool = False,
                logy : bool = False,
                cli : bool = False,
        ):
            import starsmashertools.mpl
            import starsmashertools.mpl.animation
            import copy

            fig, ax = starsmashertools.mpl.subplots()

            if x in ['x','y','z'] and y in ['x','y','z']:
                ax.set_aspect('equal')

            xlabel = copy.deepcopy(x)
            ylabel = copy.deepcopy(y)

            if logx: xlabel = 'log '+xlabel
            if logy: ylabel = 'log '+ylabel
                
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

            all_output = self.get_output()
            
            artist = all_output[0].plot(
                ax,
                x = x,
                y = y,
                logx = logx,
                logy = logy,
            )
            
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
            t = None
            try:
                t = self.get_current_time(
                    use_logfiles = True, include_joined = False,
                )
            except:
                try:
                    t = self.get_current_time(
                        use_logfiles = False, include_joined = False
                    )
                except: pass
                    
            if t is None: tunit = self.units['t'].auto().label
            else: tunit = t.auto().label
            def update(i):
                output = all_output[min(max(i,0), len(all_output) - 1)]
                artist.set_output(output)

                label.set_text(str(output))
                time = (output.header['t'] * self.units['t']).convert(tunit)
                time_label.set_text('t = %10g %s' % (time.value, tunit))
            
            update(0)
            ani = starsmashertools.mpl.animation.Player(
                fig, update, maxi=len(all_output),
                interval = int(1./30. * 1000.), # framerate in ms
            )
            
            if cli:
                fig.show()
                return ""
            return ani


    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    @cli('starsmashertools')
    @clioptions(display_name = 'Concatenate simulations')
    def join(
            self,
            other,
            cli : bool = False,
    ):
        """
        Merge two simulations of the same type together, by having
        starsmashertools retrieve the output files of all joined simulations
        whenever the ``include_joined`` flag is set to `True` (default) in
        :meth:`~.get_output`\, :meth:`~.get_output_at_time`\, 
        :meth:`~.get_output_iterator`\, :meth:`~.get_outputfiles`\, 
        :meth:`~.get_logfiles`\, :meth:`~.get_energyfiles`\, and 
        :meth:`~.get_energy`\.

        The ``restartrad.sph.orig`` file (or equivalent from 
        :mod:`starsmashertools.preferences`\) must originate from ``other``\,
        or ``other`` must have a ``restartrad.sph.orig`` file which originates
        from this simulation. Otherwise, a :class:`~.JoinError` is raised.

        To undo this operation, use :meth:`~.split`\.
        
        Parameters
        ----------
        other : str, :class:`~.Simulation`
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
            message = "Cannot join simulations '%s' and '%s' because they are the same simulation" % (self.directory, other.directory)
            if cli: return message
            raise Simulation.JoinError(other, message = message)

        if not self.started_from(other) and not other.started_from(self):
            raise Simulation.JoinError(other)
        
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
    @clioptions(display_name = 'Detach concatenated simulations')
    def split(
            self,
            which = None,
            cli : bool = False,
    ):
        """
        The opposite of :meth:`~.join`\. Split this simulation apart from each
        simulation it is joined to. Any simulation joined to this one will also
        be split apart from it.

        Other Parameters
        ----------------
        which : str, :class:`~.Simulation`\, None, default = None
            Which simulation to split from the joined simulations. If a `str` is
            given, it must match exactly one of the directory strings stored in
            this simulation's archive file. If a :class:`~.Simulation` is given,
            it must have a directory which matches one of those directory
            strings. Otherwise, if `None`\, this simulation will be split apart 
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
    @clioptions(display_name = 'Show concatenated simulations')
    def show_joined_simulations(self, cli : bool = True):
        import starsmashertools.bintools
        newline = starsmashertools.bintools.Style.get('characters', 'newline')

        if 'joined simulations' not in self.archive:
            return "There are no joined simulations"
        return newline.join(self.archive['joined simulations'].value)


    def get_state(self):
        """ 
        Return a new :class:`~.State` object and call :meth:`~.State.get`\.
        """
        state = State(self)
        state.get()
        return state


class State(object):
    """
    Contains information about a :class:`~.Simulation` which can be used to
    check for changes in a :class:`~.Simulation` over time. A :class:`~.State`
    cares only about changes which StarSmasher has made to the physical 
    simulation and all other changes are ignored.
    """
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def __init__(self, simulation : Simulation | str):
        import starsmashertools
        import starsmashertools.helpers.path
            
        if isinstance(simulation, str):
            simulation = starsmashertools.get_simulation(simulation)
        
        self.simulation = simulation
        self.mtimes = None

    def get(self):
        # Get the modification times of all the files that we expect for
        # StarSmasher to produce. This will set this unique state.
        patterns_list = self.simulation.preferences.get('state files')
        self.mtimes = {}
        for pattern in patterns_list:
            for f in self.simulation.get_file(pattern):
                path = starsmashertools.helpers.path.relpath(
                    f, start = self.simulation.directory,
                )
                self.mtimes[path] = starsmashertools.helpers.path.getmtime(f)
    
    def __getattribute__(self, attr):
        if attr == 'mtimes':
            ret = super(State, self).__getattribute__(attr)
            if ret is None:
                raise Exception("Uninitialized State object. You must call the get() function before using a State object.")
            return ret
        return super(State, self).__getattribute__(attr)

    def __json_view__(self):
        import datetime
        max_mtime = max([mtime for mtime in self.mtimes.values()])
        return str(datetime.datetime.fromtimestamp(max_mtime))

    def pack(self):
        return {
            'simulation' : self.simulation,
            'mtimes' : self.mtimes,
        }
    
    @staticmethod
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def unpack(obj : dict):
        state = State(obj['simulation'])
        state.mtimes = obj['mtimes']
        return state
    
    def __eq__(self, other):
        if not isinstance(other, State):
            if not isinstance(other, State): raise NotImplementedError
        
        if self.simulation != other.simulation:
            raise ValueError("State objects from two different simulations cannot be compared: %s and %s" % (self.simulation, other.simulation))

        keys1 = self.mtimes.keys()
        keys2 = other.mtimes.keys()
        
        if set(keys1) != set(keys2): return False
        for key, val in zip(keys1, self.mtimes.values()):
            if other.mtimes[key] != val: return False
        return True
        
    def __gt__(self, other):
        # A State is 'greater than' another if it contains new or more recent
        # files than the other state. If the other state contains any new or
        # more recent files, this returns False.
        if not isinstance(other, State): raise NotImplementedError
        if self == other: return False
        
        keys = self.mtimes.keys()
        for key, val in other.mtimes.items():
            # If other contains files that we don't have, we are not greater
            # than it
            if key not in keys: return False
            # If other has at least one more recent file than us, we aren't
            # greater than it
            if val > self.mtimes[key]: return False
        
        # If we are not equal to other, then we must be greater than it
        return True

    def __lt__(self, other):
        if not isinstance(other, State): raise NotImplementedError
        return (not self > other) and self != other
    def __ge__(self, other):
        return self == other or self > other
    def __le__(self, other):
        return self == other or self < other
