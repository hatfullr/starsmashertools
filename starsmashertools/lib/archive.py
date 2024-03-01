import starsmashertools.helpers.argumentenforcer
from starsmashertools.helpers.apidecorator import api
import contextlib
import zipfile

def get_file_info():
    import starsmashertools.helpers.jsonfile
    return starsmashertools.helpers.jsonfile.save_bytes({
        '__version__' : starsmashertools.__version__,
    })
def load_file_info(info):
    import starsmashertools.helpers.jsonfile
    return starsmashertools.helpers.jsonfile.load_bytes(info)

def update_archive_version(
        old_path : str,
        new_path : str | type(None) = None,
):
    """
    Convert old-style :py:class:`~.Archive` files to new-style files. This 
    function does nothing if the given :py:class:`~.Archive` is already in the
    latest format.

    Parameters
    ----------
    old_path : str
        The file path to the old :py:class:`~.Archive` to convert.

    Other Parameters
    ----------------
    new_path : str, None, default = None
        If a `str` is given, the updated :py:class:`~.Archive` will be saved at
        that file path. Otherwise it will be saved at the old 
        :py:class:`~.Archive`'s path, overwriting it.
    """
    import starsmashertools.helpers.file
    import starsmashertools.helpers.jsonfile
    import tempfile
    import shutil
    import warnings

    if new_path is None: new_path = old_path
    
    with starsmashertools.helpers.file.open(
            old_path, 'a', **Archive.open_method_kwargs
    ) as zfile:
        namelist = zfile.namelist()
        if len(namelist) == 1 and namelist[0] != 'file info':
            content = zfile.read(namelist[0])
            data = starsmashertools.helpers.jsonfile.load_bytes(content)
            for identifier, val in data.items():
                data[identifier] = ArchiveValue(
                    identifier,
                    val['value'],
                    val['origin'],
                    val['mtime'],
                )
                #data[identifier] = ArchiveValue.deserialize(identifier, val)
        else: # If we were not able to detect the old Archive format
            # We assume it is in the latest format, so just update the version
            # info
            warnings.filterwarnings(action = 'ignore')
            zfile.writestr('file info', get_file_info())
            warnings.resetwarnings()
            return
    
    # We should now have the base data. We now create a new Archive as a
    # temporary file, and then overwrite the old one with the new one.
    new_archive = Archive('new_archive.temp', auto_save = False)
    for identifier, val in data.items():
        new_archive[identifier] = val
    new_archive.save()
    
    shutil.move(new_archive.filename, new_path)



def _remove_zipfile_member(zfile, member):
    # With help from:
    # https://github.com/python/cpython/blob/659eb048cc9cac73c46349eb29845bc5cd630f09/Lib/zipfile.py#L1717
    # get a sorted filelist by header offset, in case the dir order
    # doesn't match the actual entry order
    from operator import attrgetter

    if isinstance(member, str):
        member = zfile.getinfo(member)
    
    fp = zfile.fp
    entry_offset = 0
    filelist = sorted(zfile.filelist, key=attrgetter('header_offset'))
    for i in range(len(filelist)):
        info = filelist[i]
        # find the target member
        if info.header_offset < member.header_offset:
            continue

        # get the total size of the entry
        entry_size = None
        if i == len(filelist) - 1:
            entry_size = zfile.start_dir - info.header_offset
        else:
            entry_size = filelist[i + 1].header_offset - info.header_offset

        # found the member, set the entry offset
        if member == info:
            entry_offset = entry_size
            continue

        # Move entry
        # read the actual entry data
        fp.seek(info.header_offset)
        entry_data = fp.read(entry_size)

        # update the header
        info.header_offset -= entry_offset

        # write the entry to the new position
        fp.seek(info.header_offset)
        fp.write(entry_data)
        fp.flush()

    # update state
    zfile.start_dir -= entry_offset
    zfile.filelist.remove(member)
    del zfile.NameToInfo[member.filename]
    zfile._didModify = True

    # seek to the start of the central dir
    fp.seek(zfile.start_dir)



    

class ArchiveValue(object):
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(
            self,
            identifier : str,
            value,
            origin : str | type(None) = None,
            mtime : int | float | type(None) = None,
    ):
        """
        ArchiveValue constructor.

        Parameters
        ----------
        identifier : str
            A unique string identifier that is used by :class:`~.Archive` 
            objects when storing and accessing data in the archive.
        
        value : serializable types
            A value that is one of the serializable types, found in the keys of
            :py:property:`~.helpers.jsonfile.serialization_methods`.
        
        origin : str, None, default = None
            Path to the file from which this value originated.

        mtime : int, float, None, default = None
            The modification time of the file specified by `origin`. If not
            `None` then `origin` can be a file which doesn't currently exist.
        """
        import starsmashertools.helpers.jsonfile
        import starsmashertools.helpers.path
        
        _types = starsmashertools.helpers.jsonfile.serialization_methods.keys()
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'value' : _types,
        })

        self.identifier = identifier
        self.value = value
        self.origin = origin

        if self.origin is not None and mtime is None:
            mtime = starsmashertools.helpers.path.getmtime(self.origin)
        
        self.mtime = mtime

    def __str__(self):
        return 'ArchiveValue(%s)' % str(self.value)

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def __eq__(self, other : "ArchiveValue"):
        return self.serialize() == other.serialize()

    def serialize(self):
        """
        Return a copy of this value in a format that can be written to the
        :class:`~.Archive` file.
        """
        import starsmashertools.helpers.jsonfile
        return starsmashertools.helpers.jsonfile.save_bytes({
            'value' : self.value,
            'origin' : self.origin,
            'mtime' : self.mtime,
        })

    @staticmethod
    def deserialize(identifier, obj):
        """
        Return a :class:`~.ArchiveValue` object from the given json object.
        """
        import starsmashertools.helpers.jsonfile
        json = starsmashertools.helpers.jsonfile.load_bytes(obj)
        return ArchiveValue(
            identifier,
            json['value'],
            json['origin'],
            json['mtime'],
        )



def REPLACE_OLD(
        old_value : ArchiveValue,
        new_value : ArchiveValue,
):
    """
    A function that decides whether or not to replace an old value. The old
    value is replaced if its file modification time is less than the new
    value's. This function can be registered as a replacement checker in an
    :class:`~.Archive`. This is a valid flag for the `replacement_flags`
    keyword in the constructor of an :class:`~.Archive`.

    Parameters
    ----------
    old_value : :class:`~.ArchiveValue`
        The old value found in the archive.

    new_value : :class:`~.ArchiveValue`
        The new value whose identifier matches that of the `old_value` and who
        will replace ``old_value`` if this function returns `True`.

    Returns
    -------
    bool
        If `True` then ``new_value`` will replace ``old_value`` in the archive.
        Otherwise, no replacement is made.
    """
    return new_value.mtime > old_value.mtime

def REPLACE_NEQ(
        old_value : ArchiveValue,
        new_value : ArchiveValue,
):
    return new_value.value != old_value.value

        

class Archive(object):
    """
    An Archive object stores :class:`ArchiveValue`s in a single file for quick 
    access. Note that :class:`ArchiveValue` can only store data that is JSON
    serializable. This typically includes only standard types str, bool, float,
    int, list, tuple, and dict. However, additional types such as np.ndarray,
    np.integer, and more can also be stored as specified in 
    :py:mod:`~.starsmashertools.helpers.jsonfile`, which you can update to
    include additional data types. See
    :py:property:`~.helpers.jsonfile.serialization_methods` for details.
    
    Examples
    --------
    Suppose you have many StarSmasher output files and you want to store the
    total number of particles each one has in a file called ``mydata.dat``:
    
        import starsmashertools
        import starsmashertools.lib.archive
        
        simulation = starsmashertools.get_simulation(".")
        archive = starsmashertools.lib.archive.Archive("mydata.dat")
        
        for output in simulation.get_output_iterator():
            archive.add(
                output.path, # Some unique string identifier
                output.header['ntot'], # The value to store
                origin = output.path, # The file the value originated from
            )
            # Write the archive to mydata.dat. If we save it on every loop
            # iteration, then interrupting the program won't cause us to lose
            # any data we already calculated.
            archive.save()

    """

    open_method_kwargs = {
        'method' : zipfile.ZipFile,
        'compression' : zipfile.ZIP_DEFLATED,
        'compresslevel' : 9,
        'lock' : True,
    }

    class ReadOnlyError(Exception, object):
        def __init__(self, filename):
            message = "Cannot modify readonly archive: %s" % str(self)
            super(Archive.ReadOnlyError, self).__init__(message)
    class CorruptFileError(Exception, object): pass
    class MissingIdentifierError(Exception, object): pass
    class FormatError(Exception, object): pass
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(
            self,
            filename : str,
            replacement_flags : list | tuple | type(None) = None,
            auto_save : bool = True,
            readonly : bool = False,
            verbose : bool = True,
    ):
        """
        Constructor for :class:`~.Archive`.

        Parameters
        ----------
        filename : str
            The name of the file to use as the archive.

        Other Parameters
        ----------------
        replacement_flags : list, tuple, None, default = None
            A list of flags to use to determine if a replacement should happen
            in the archive whenever a new :class:`ArchiveValue` is about to be
            written to a pre-existing identifier. If `None` then the default
            from :py:property:`~.preferences.defaults` under 'Archive' and
            'replacement flags' will be used.
            Each element of in `replacement_flags` must be a function which
            accepts two arguments, each is :class:`~.ArchiveValue`. The first
            argument is the old value and the second the new value. Each 
            function must return a bool-like value which will be evaluated by
            an "if" statement.

        auto_save : bool, default = True
            If `True`, values will automatically be written to the file on the
            system whenever they are edited. If `False`, it is the user's
            responsibility to call :func:`~.save` to save the contents.

        readonly : bool, default = False
            If `True`, a :py:class:`~.Archive.ReadOnlyError` is raised whenever
            :py:func:`~.Archive.save` is called on this object.
            
        verbose : bool, default = True
            If `False`, messages are suppressed. Otherwise messages are
            printed to the standard output.
        """
        import starsmashertools.helpers.path
        import starsmashertools.preferences
        import atexit

        self._previous_mtime = None
        
        if replacement_flags is None:
            replacement_flags = starsmashertools.preferences.get_default(
                'Archive',
                'replacement flags',
            )
        
        self.filename = filename
        self.replacement_flags = replacement_flags
        self.auto_save = auto_save
        self.readonly = readonly
        self.verbose = verbose
        
        super(Archive, self).__init__()

        if starsmashertools.helpers.path.isfile(self.filename):
            self._check_and_update_format()

        # If the program is about to quit, make sure the Archive has been saved
        def save_at_exit():
            if self.readonly: return
            self.save()
        atexit.register(save_at_exit)
        
        self._to_add = {}
        self._to_remove = []

    @property
    def modified(self):
        import starsmashertools.helpers.path
        if not starsmashertools.helpers.path.exists(self.filename): return False
        current_mtime = starsmashertools.helpers.path.getmtime(self.filename)
        return current_mtime > self._previous_mtime
    
    def __copy__(self, *args, **kwargs):
        raise NotImplementedError
    def __deepcopy__(self, *args, **kwargs):
        raise NotImplementedError
    
    def __repr__(self):
        import starsmashertools.helpers.string
        path = starsmashertools.helpers.string.shorten(
            self.filename,
            30,
            where = 'left',
        )
        return "Archive('%s')" % path

    @contextlib.contextmanager
    def nosave(self):
        """ A context manager for temporarily disabling auto save. """
        import copy
        was_auto_save = copy.deepcopy(self.auto_save)
        try:
            self.auto_save = False
            yield
        finally:
            self.auto_save = copy.deepcopy(was_auto_save)

    @contextlib.contextmanager
    def open(self, mode, verbose : bool | type(None) = None):
        """ Used for opening the Archive for file manipulation. """
        import starsmashertools.helpers.string
        import starsmashertools.helpers.file

        if verbose is None: verbose = self.verbose

        if verbose:
            message = "Loading '%s'" % starsmashertools.helpers.string.shorten(
                self.filename,
                50,
                where = 'left',
            )
            try:
                with starsmashertools.helpers.string.loading_message(message,delay=5):
                    with starsmashertools.helpers.file.open(
                            self.filename, mode, **Archive.open_method_kwargs
                    ) as zfile:
                        yield zfile
            except Exception as e:
                raise Archive.CorruptFileError("Failed to load archive file, likely because it did not save correctly. Please delete it and try again: '%s'" % self.filename) from e
        else:
            try:
                with starsmashertools.helpers.file.open(
                        self.filename, mode, **Archive.open_method_kwargs
                ) as zfile:
                    yield zfile
            except Exception as e:
                raise Archive.CorruptFileError("Failed to load archive file, likely because it did not save correctly. Please delete it and try again: '%s'" % self.filename) from e
            
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __contains__(
            self,
            key : str | ArchiveValue,
    ):
        """ If an :class:`~.ArchiveValue` is specified, calls 
        :py:meth:`~.values`, which reads the entire Archive file and can be 
        quite I/O intensive. Use sparingly. Otherwise, if a str is given then
        the string is checked against the keys, which is faster. """
        if isinstance(key, str): return key in self.keys()
        else: return key in self.values()

    def __delitem__(self, key):
        """ Remove the given key from the Archive. If auto save is enabled, the
        Archive file will be modified. """
        if key in self._to_remove: raise KeyError(key)
        self._to_remove += [key]
        self._to_add.pop(key, None)
        if self.auto_save: self.save()
    
    @api
    def keys(self):
        """ Return a list of Archive identifiers from the file. """
        import starsmashertools.helpers.path
        
        keys = list(self._to_add.keys())
        if not starsmashertools.helpers.path.exists(self.filename):
            return keys
        
        with self.open('r') as zfile:
            namelist = zfile.namelist()

        for key in namelist:
            if (key == 'file info' or
                key in self._to_remove):
                continue
            keys += [key]
        
        return list(set(keys))

    @api
    def items(self):
        """ Return all keys and values in the Archive. This is I/O intensive, 
        so use sparingly. """
        import starsmashertools.helpers.path

        everything = {}
        if not starsmashertools.helpers.path.exists(self.filename):
            return everything.items()

        with self.open('r') as zfile:
            for key in zfile.namelist():
                if key == 'file info': continue
                everything[key] = zfile.read(key)
        
        for key, value in everything.items():
            everything[key] = ArchiveValue.deserialize(key, value)

        everything.update(self._to_add)
        
        return everything.items()
            
    @api
    def values(self):
        """ Return all the keys and values in the Archive. This reads the entire
        Archive file, so use it sparingly. """

        values = []
        for key, value in self.items():
            values += [value]
        return values

    
    @api
    def clear(self, *args, **kwargs):
        """ Clears all contents from the archive and deletes the file if auto
        save is enabled. """
        import starsmashertools.helpers.path
        if self.readonly:
            raise Archive.ReadOnlyError("Cannot call method 'clear' on an Archive in readonly mode.")
        
        self._to_add.clear()
        self._to_remove = []
        if self.auto_save:
            if starsmashertools.helpers.path.exists(self.filename):
                starsmashertools.helpers.path.remove(self.filename)

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __setitem__(
            self,
            key,
            value : ArchiveValue,
    ):
        """ Similar to :py:func:`dict.__setitem__` except the archive file is
        modified if auto save is enabled. Also the replacement flag functions
        specified in :py:func:`~.__init__` are checked. If all the functions 
        return `True` then the value is overwritten. Otherwise, nothing 
        happens. """
        if key in self.keys():
            # Check all the replacement flag functions. If any fail, do nothing.
            if not all([flag(self[key], value) for flag in self.replacement_flags]):
                return
        
        # Insert a new key or add it to the archive
        self._to_add[key] = value
        if self.auto_save: self.save()

    def __getitem__(self, key):
        """ Obtain the value corresponding with the given key. """
        # If we don't have the key, check the file itself
        if key not in self.keys():
            raise KeyError(key)

        # If we do have the key, fetch its value
        if key in self._to_add.keys():
            return self._to_add[key]
        else:
            with self.open('r') as zfile:
                value = zfile.read(key)
            return ArchiveValue.deserialize(key, value)

    def _check_and_update_format(self):
        import starsmashertools
        import warnings
        import starsmashertools.preferences
        import starsmashertools.helpers.path

        if not starsmashertools.helpers.path.exists(self.filename): return

        # We can't update the file if we're in readonly mode.
        if self.readonly: return

        should_update = False
        info = None
        with self.open('r') as zfile:
            if 'file info' in zfile.namelist():
                info = zfile.read('file info')

        if info:
            info = load_file_info(info)

        old_version = None
        if info is None: should_update = True
        elif starsmashertools._is_version_older(info['__version__']):
            should_update = True
            old_version = info['__version__']

        user_allowed = starsmashertools.preferences.get_default(
            'Archive',
            'auto update format',
            throw_error = True,
        )

        if old_version is not None:
            message = "%s was written by starsmashertools version '%s', but the current version is '%s'. This Archive will be updated to the latest format."
            message = message % (str(self), old_version, starsmashertools.__version__)
        else:
            message = "%s was written by an older starsmashertools version, but the current version is '%s'. This Archive will be updated to the latest format."
            message = message % (str(self), starsmashertools.__version__)
        
        if should_update:
            if not user_allowed: raise Archive.FormatError(message)
            warnings.warn(message)
            update_archive_version(self.filename)
            

    """
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def load(
            self,
            verbose : bool | type(None) = None,
    ):
        ""
        Load the entire archive into memory. This is preferred when you expect
        to perform many read/write operations so as to not flood the I/O with
        requests.

        For large Archives, loading can take a long time.

        Parameters
        ----------
        verbose : bool, None, default = None
            Overrides the :py:attr:`~.verbose` option in :py:func:`~.__init__`.
            If `None` then :py:attr:`~.verbose` is used instead.
        ""
        import starsmashertools.helpers.jsonfile
        import starsmashertools.helpers.file
        import starsmashertools.helpers.path
        import starsmashertools.helpers.string
        if verbose is None: verbose = self.verbose
        
        if starsmashertools.helpers.path.exists(self.filename):
            if starsmashertools.helpers.path.getsize(self.filename) == 0:
                starsmashertools.helpers.path.remove(self.filename)

        def do(*args, **kwargs):
            with starsmashertools.helpers.file.open(
                    self.filename, 'r', **Archive.open_method_kwargs
            ) as zfile:
                try:
                    namelist = zfile.namelist()
                except Exception as e:
                    raise Archive.CorruptFileError("Failed to load archive file, likely because it did not save correctly. Please delete it and try again: '%s'" % self.filename) from e
                content = {}
                for key in namelist:
                    if key == 'file info': continue
                    content[key] = zfile.read(key)

            return content
            
                
        if verbose:
            message = "Loading '%s'" % starsmashertools.helpers.string.shorten(
                self.filename,
                50,
                where = 'left',
            )
            with starsmashertools.helpers.string.loading_message(message,delay=5):
                data = do()
        else: data = do()
        
        with self.nosave():
            self.clear()
            for identifier, val in data.items():
                self[identifier] = ArchiveValue.deserialize(
                    identifier,
                    starsmashertools.helpers.jsonfile.load_bytes(val),
                )
        
        self.loaded = True
    """
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def save(
            self,
            verbose : bool | type(None) = None,
    ):
        """
        Update the archive file with the current data.

        Parameters
        ----------
        verbose : bool, None, default = None
            Overrides the :py:attr:`~.verbose` option in :py:func:`~.__init__`.
            If `None` then :py:attr:`~.verbose` is used instead.
        """
        if self.readonly: raise Archive.ReadOnlyError(self.filename)

        # If nothing changed
        if not self._to_add and not self._to_remove: return
        
        # Serialize the data we will add
        data = {}
        for key, val in self._to_add.items():
            data[key] = val.serialize()
        
        # Get file info
        info = get_file_info()

        remove_keys = self._to_remove + list(data.keys())
        
        with self.open('a', verbose = verbose) as zfile:
            keys = zfile.namelist()
            
            # Update the file info
            if 'file info' in keys:
                _remove_zipfile_member(zfile, 'file info')
            
            zfile.writestr('file info', info)
            
            # Remove all keys which are to be either removed or overwritten
            for key in remove_keys:
                if key not in keys: continue
                _remove_zipfile_member(zfile, key)
            
            # Add keys that need to be added
            for key, val in data.items():
                zfile.writestr(key, val)
        
        self._to_add.clear()
        self._to_remove = []
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def add(self, *args, **kwargs):
        """
        Create a new :class:`ArchiveValue` and add it to the archive. If there
        already exists a :class:`ArchiveValue` with the same `identifier` in the
        archive, then if the origin file is different we overwrite the old
        value. Otherwise, if the file's modification time is more recent than
        the archived value, we also overwrite the old value.

        Other Parameters
        ----------
        *args
            Positional arguments are passed directly to :class:`ArchiveValue`.

        **kwargs
            Keyword arguments are passed directly to :class:`ArchiveValue`.
        """
        value = ArchiveValue(*args, **kwargs)
        self[value.identifier] = value
            
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def remove(self, identifier : str):
        """
        Remove an identifier from the Archive. Calls :py:meth:`~.save` if auto
        save is enabled.

        Parameters
        ----------
        identifier : str
            The identifier key to remove.
        """
        del self[identifier]

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def combine(
            self,
            other : 'str | Archive',
    ):
        """
        Merge the contents of 'other' with this Archive, according to this 
        Archive's replacement flags.

        Parameters
        ----------
        other : str, Archive
            The Archive whose contents will be merged into this Archive. Only 
            this Archive will be affected.
        """
        with self.nosave():
            if isinstance(other, str): other = Archive(other)
            else: other.save() # Ensure the other Archive is up-to-date
            
            for key, val in other.items():
                self[key] = val
        if self.auto_save: self.save()





    
class OutputArchiveValue(ArchiveValue, object):
    """
    This class is used specifically for storing an :class:`~.lib.output.Output`
    object in an Archive. If the output file is stored on a different file
    system originally, then we should not identify it by its absolute path.
    Instead we should identify it as its relative path to the simulation
    directory.
    """
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(
            self,
            output : "starsmashertools.lib.output.Output",
            value : dict,
    ):
        origin = None
        identifier = SimulationArchive.get_identifier_static(output)
        mtime = starsmashertools.helpers.path.getmtime(output.path)
        super(OutputArchiveValue, self).__init__(
            identifier,
            value,
            origin = origin,
            mtime = mtime,
        )




















class SimulationArchive(Archive, object):
    """
    When working with data from a :class:`~.lib.simulation.Simulation`, it is
    common to want to save results of calculations in a way that is easily
    retrievable again in the future, such as getting the sum of total energies
    etc. Such operations can take a very long time if there are a lot of output
    files. If the work is interrupted then all the calculations are lost and we
    have to start over again. To mitigate this we can use a SimulationArchive.

    Examples
    --------
    
        import starsmashertools
        import starsmashertools.lib.archive
        simulation = starsmashertools.get_simulation(".")
        archive = starsmashertools.lib.archive.SimulationArchive(simulation)
    
    """
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(
            self,
            simulation : "starsmashertools.lib.simulation.Simulation | str",
    ):
        """
        Constructor for a :class:`SimulationArchive`. Note that a
        :class:`~.SimulationArchive` is not directly loaded or saved. This
        rather happens automatically as archived values are accessed / modified.

        There are no replacement flags available, as all values are forcibly 
        overwritten always.

        Parameters
        ----------
        simulation : :py::class:`~.simulation.Simulation`, str
            The simulation to which this archive belongs, or a string path to a
            simulation directory or a simulation archive within a simulation
            directory.
        """
        import starsmashertools.helpers.path
        import starsmashertools.preferences

        basename = starsmashertools.preferences.get_default(
            'Simulation',
            'archive filename',
            throw_error = True,
        )

        if isinstance(simulation, str):
            if starsmashertools.helpers.path.isdir(simulation):
                # It's a simulation directory
                filename = starsmashertools.helpers.path.join(
                    starsmashertools.helpers.path.realpath(simulation),
                    basename,
                )
            elif starsmashertools.helpers.path.isfile(simulation):
                # It's supposedly a simulation archive file path
                filename = simulation
        else:
            # Ensure that the filename is located in the simulation directory
            filename = starsmashertools.helpers.path.join(
                simulation.directory,
                basename
            )
        
        super(SimulationArchive, self).__init__(
            filename,
            replacement_flags = [],
            auto_save = True,
            readonly = False,
        )

    @api
    def get_identifier(self, output : "starsmashertools.lib.output.Output"):
        """
        Obtain the expected identifier for the given
        :class:`~.lib.output.Output` object.

        Parameters
        ----------
        output : :class:`starsmashertools.lib.output.Output`
            The :class:`starsmashertools.lib.output.Output` object for which to
            get the :class:`~.SimulationArchive` identifier.

        Returns
        -------
        str
            The output file's expected identifier in this archive.
        """
        return SimulationArchive.get_identifier_static(output)

    @api
    @staticmethod
    def get_identifier_static(output : "starsmashertools.lib.output.Output"):
        """
        Obtain the expected identifier for the given
        :class:`~.lib.output.Output` object.

        Parameters
        ----------
        output : :class:`starsmashertools.lib.output.Output`
            The :class:`starsmashertools.lib.output.Output` object for which to
            get the :class:`~.SimulationArchive` identifier.

        Returns
        -------
        str
            The output file's expected identifier in this archive.
        """
        import starsmashertools.helpers.path
        import starsmashertools.lib.output

        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'output' : [starsmashertools.lib.output.Output],
        })
        
        return starsmashertools.helpers.path.relpath(
            output.path,
            start = output.simulation.directory,
        )

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def add_output(self, *args, **kwargs):
        """
        Add an :class:`~.OutputArchiveValue` to the archive.

        Parameters
        ----------
        *args
            Positional arguments are passed directly to
            :class:`~.OutputArchiveValue`.
        
        **kwargs
            Keyword arguments are passed directly to
            :class:`~.OutputArchiveValue`.
        
        """
        val = OutputArchiveValue(*args, **kwargs)
        self[val.identifier] = val

        
