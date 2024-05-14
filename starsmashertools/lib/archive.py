import starsmashertools.preferences
from starsmashertools.preferences import Pref
import starsmashertools.helpers.argumentenforcer
from starsmashertools.helpers.apidecorator import api
from starsmashertools.helpers.clidecorator import cli, clioptions
import contextlib
import zipfile
import atexit
import sys
import gc

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
        verbose : bool = False,
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
    import starsmashertools.helpers.warnings
    import starsmashertools.helpers.string
    
    if new_path is None: new_path = old_path

    def do_atexit(*args, **kwargs):
        import os
        if os.path.exists(new_archive.filename):
            os.remove(new_archive.filename)

    new_archive = starsmashertools.lib.archive.Archive(
        'new_archive.temp',
        auto_save = False,
    )
    # In case we fail to write the new archive, delete it
    atexit.register(do_atexit)

    format_detected = True
    with starsmashertools.helpers.file.open(
            old_path, 'r', **Archive.open_method_kwargs
    ) as zfile:
        namelist = zfile.namelist()
        if len(namelist) == 1 and namelist[0] != 'file info':
            with starsmashertools.helpers.string.loading_message("Reading '%s'" % Archive.to_str(old_path)) as l:
                content = zfile.read(namelist[0])
                data = starsmashertools.helpers.jsonfile.load_bytes(content)
                del content
        else: # If we were not able to detect the old Archive format
            format_detected = False

    if format_detected:
        new_archive._buffers['add'] = {key:ArchiveValue(
            key,
            val['value'],
            origin = val['origin'],
            mtime = val['mtime'],
        ) for key, val in data.items()}
        del data
    
    if not format_detected:
        with starsmashertools.helpers.file.open(
                old_path, 'a', **Archive.open_method_kwargs
        ) as zfile:
            # We assume it is in the latest format, so just update the
            # version info
            starsmashertools.helpers.warnings.filterwarnings(action = 'ignore')

            while 'file info' in zfile.namelist():
                try:
                    _remove_zipfile_member(zfile, 'file info')
                except KeyError: break
            
            zfile.writestr('file info', get_file_info())
            
            starsmashertools.helpers.warnings.resetwarnings()
            return
    else:
        new_archive.save()
        
        # Now change file names
        shutil.move(new_archive.filename, new_path)
    
    atexit.unregister(do_atexit)

    del new_archive
    del format_detected
    


def _remove_zipfile_member(zfile, member):
    # With help from:
    # https://github.com/python/cpython/blob/659eb048cc9cac73c46349eb29845bc5cd630f09/Lib/zipfile.py#L1717
    # get a sorted filelist by header offset, in case the dir order
    # doesn't match the actual entry order
    from operator import attrgetter

    if isinstance(member, str):
        member = zfile.getinfo(member)

    # We need to replace the system excepthook such that whenever an exception
    # is thrown, we finish modifying the file and then throw the exception
    # afterwards.
    
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
        
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'value' : starsmashertools.helpers.jsonfile.serialization_methods.keys(),
        })

        self._value = None
        self.serialized = None
        
        self.identifier = identifier
        self.origin = origin
        
        if self.origin is not None and mtime is None:
            mtime = starsmashertools.helpers.path.getmtime(self.origin)
        self.mtime = mtime
        
        if isinstance(value, bytes):
            # Catch values which are already serialized
            try:
                value = ArchiveValue.deserialize(self.identifier, value)
            except: pass
            
        self.value = value

    def __str__(self):
        return 'ArchiveValue(%s)' % str(self.value)

    def __eq__(self, other):
        if not isinstance(other, ArchiveValue): return False
        return self.serialized == other.serialized
    
    @property
    def size(self): return self.serialized.__sizeof__()
    
    @property
    def value(self): return self._value
    @value.setter
    def value(self, new_value):
        import starsmashertools.helpers.jsonfile
        self._value = new_value
        self.serialized = starsmashertools.helpers.jsonfile.save_bytes({
            'value' : new_value,
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
        ret = ArchiveValue(
            identifier,
            json['value'],
            json['origin'],
            json['mtime'],
        )
        del json
        gc.collect()
        return ret


class ArchiveItems(object):
    """ An iterator to help with reading an Archive. """
    def __init__(self, *args, verbose : bool = False):
        self.verbose = verbose
        self._file = None
        
        if len(args) == 1:
            archive = args[0]

            if archive.thread_safe: self._keys = archive.keys()
            else: self._keys = archive._keys

            # Reading all the contents of an Archive usually isn't what takes
            # the most time. It's the deserializing that is costly.
            self._values = archive.get(self._keys, deserialize = False)
            if not isinstance(self._values, list):
                self._values = [self._values]
        else:
            self._keys, self._values = args
        
        self._iterator = None
        self._process = None

    def __len__(self): return len(self._keys)
    def __iter__(self): return self

    def __next__(self):
        if self._iterator is None: self.start()
        try:
            if self._process is None: # Serial
                key, value = next(self._iterator)
            else: # Multiprocessing
                index, archive_value = next(self._iterator)
                key, value = archive_value.identifier, archive_value
            return key, self.do_deserialize(key, value)
        except StopIteration:
            self.stop()
            raise
    
    def do_deserialize(self, key, value):
        if not isinstance(value, ArchiveValue):
            return ArchiveValue.deserialize(key, value)
        return value

    def start(self):
        import starsmashertools.helpers.asynchronous

        nprocs = min(
            starsmashertools.helpers.asynchronous.max_processes,
            len(self._keys),
        )
        self._process = None
        
        if nprocs == 1 or not starsmashertools.helpers.asynchronous.is_main_process():
            # Run in serial
            self._iterator = zip(self._keys, self._values)
        else:
            arguments = [[key, value] for key, value in zip(self._keys, self._values)]
            self._process = starsmashertools.helpers.asynchronous.ParallelFunction(
                target = self.do_deserialize,
                args = arguments,
                nprocs = nprocs,
                start = True,
            )
            self._iterator = self._process.get_output(sort = False)
    
    def stop(self):
        import starsmashertools.helpers.asynchronous
        if self._process is not None:
            del self._process
            self._process = None
        self._iterator = None


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
    if new_value.mtime is None or old_value.mtime is None: return True
    return new_value.mtime > old_value.mtime

def REPLACE_NEQ(
        old_value : ArchiveValue,
        new_value : ArchiveValue,
):
    return new_value.value != old_value.value


@starsmashertools.preferences.use
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
        #'lock' : True,
    }

    class ReadOnlyError(Exception, object):
        def __init__(self, filename):
            message = "Cannot modify readonly archive: %s" % str(self)
            super(Archive.ReadOnlyError, self).__init__(message)
    class CorruptFileError(Exception, object): pass
    class MissingIdentifierError(Exception, object): pass
    class FormatError(Exception, object): pass
    class ThreadSafeError(Exception, object): pass
            
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(
            self,
            filename : str,
            replacement_flags : list | tuple = Pref('replacement flags'),
            auto_save : bool = True,
            readonly : bool = False,
            verbose : bool = True,
            max_buffer_size : int = Pref('max buffer size', int(1e5)),
            thread_safe : bool = False,
    ):
        """
        Constructor for :class:`~.Archive`.

        Parameters
        ----------
        filename : str
            The name of the file to use as the archive.

        Other Parameters
        ----------------
        replacement_flags : list, tuple, default = Pref('replacement flags')
            A list of flags to use to determine if a replacement should happen
            in the archive whenever a new :class:`ArchiveValue` is about to be
            written to a pre-existing identifier. Each element of in 
            `replacement_flags` must be a function which accepts two arguments, 
            each is :class:`~.ArchiveValue`. The first argument is the old value
            and the second the new value. Each function must return a bool-like 
            value which will be evaluated by an "if" statement.

            If an empty list is given, values in the archive will be
            overwritten (no checks performed), unless if ``readonly`` is `True`.

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

        max_buffer_size : int, default = Pref('max buffer size', int(1e5))
            The maximum allowed size in bytes that the buffer can have when auto
            save is disabled. When the buffer exceeds this value the archive is
            saved.

        thread_safe : bool, default = False
            If `True` then we will try to make thread-safe assumptions about 
            the state of the archive, such as reading the file on the system to
            determine its contents whenever a value is modified. If `False`,
            then it is assumed that this archive was created on the main
            process, such that we can use faster operations for determining the
            current state of the archive.
        """
        import starsmashertools.helpers.path
        import starsmashertools.helpers.asynchronous

        self._previous_mtime = None
        
        self.filename = filename
        self.replacement_flags = replacement_flags
        self.auto_save = auto_save
        self.readonly = readonly
        self.verbose = verbose
        self._max_buffer_size = max_buffer_size
        self.thread_safe = thread_safe

        if (self.thread_safe and
            not starsmashertools.helpers.asynchronous.is_main_process()):
            raise Archive.ThreadSafeError("Cannot open an Archive on processes which aren't the main process when keyword 'thread_safe' is True")
        
        super(Archive, self).__init__()
        
        if starsmashertools.helpers.path.isfile(self.filename):
            self._check_and_update_format()
        
        # If the program is about to quit, make sure the Archive has been saved
        atexit.register(self._on_quit)
        
        self._buffers = {
            'add' : {},
            'remove' : [],
        }
        self._buffer_size = 0
        self._nosave_holders = 0

    @property
    def _keys(self):
        if not hasattr(self, '_keys_internal'):
            self._keys_internal = self._get_keys_from_file()
        return self._keys_internal

    @_keys.setter
    def _keys(self, value):
        self._keys_internal = value
    
    @staticmethod
    def to_str(filename : str):
        import starsmashertools.helpers.string
        path = starsmashertools.helpers.string.shorten(
            filename,
            50,
            where = 'left',
        )
        return "Archive('%s')" % path

    @staticmethod
    def to_repr(filename : str):
        return "Archive('%s')" % archive.filename

    def __str__(self): return Archive.to_str(self.filename)
    def __repr__(self): return Archive.to_repr(self.filename)
    
    def _on_quit(self):
        if not self.readonly:
            self._auto_save()
        atexit.unregister(self._on_quit)
    
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
    
    def _clear_buffers(self):
        self._buffers['add'].clear()
        self._buffers['remove'] = []
        self._buffer_size = 0
    
    @contextlib.contextmanager
    def nosave(self):
        """ A context manager for temporarily disabling auto save. Entering this
        context will first save the Archive. Then, saving is disabled until the
        context is exited, even if auto save is enabled. Upon exiting, the 
        buffers are cleared. """
        import copy

        if not self.thread_safe:
            previous_keys = copy.deepcopy(self._keys)
        
        # This clears the buffers
        self.save()
        
        try:
            self._nosave_holders += 1
            yield
        finally:
            # Clear the buffers
            self._clear_buffers()
            if not self.thread_safe: self._keys = previous_keys
            self._nosave_holders -= 1
    
    def open(
            self,
            mode : str,
            verbose : bool | type(None) = None,
            message : str | type(None) = None,
            progress_max : int = 0,
    ):
        """ Used for opening the Archive for file manipulation. """
        import starsmashertools.helpers.string
        import starsmashertools.helpers.file

        if verbose is None: verbose = self.verbose

        return starsmashertools.helpers.file.open(
            self.filename, mode, verbose = verbose,
            message = message, progress_max = progress_max,
            **Archive.open_method_kwargs
        )
        
    @api
    def __contains__(self, key):
        """ If an :class:`~.ArchiveValue` is specified, calls 
        :py:meth:`~.values`, which reads the entire Archive file and can be 
        quite I/O intensive. Use sparingly. Otherwise, if a str is given then
        the string is checked against the keys, which is faster. """
        if isinstance(key, ArchiveValue): key = key.identifier
        return key in self.keys()
    
    def __delitem__(self, key):
        """ Remove the given key from the Archive. If auto save is enabled, the
        Archive file will be modified. """

        if key not in self.keys(): raise KeyError(key)

        # Move the key out of the 'add' buffer and into the 'remove' buffer.
        # Note that this doesn't change the overall buffer size. Thus, when
        # the 'remove' buffer becomes large enough the archive will be saved
        # on the file system.
        value = self._buffers['add'].pop(key, None)
        self._buffers['remove'] += [key]
        
        if not self.thread_safe: self._keys.remove(key)
        
        self._auto_save()

    def _get_keys_from_file(self):
        """ Return a list of keys as read from the file on the system. """
        import starsmashertools.helpers.path
        # If the file doesn't exist yet, then it has no keys!
        if not starsmashertools.helpers.path.exists(self.filename): return []
        try:
            with self.open(
                    'r', message = 'Loading keys in %s' % self
            ) as zfile:
                keys = zfile.namelist()
        except Exception as e:
            raise Archive.CorruptFileError("Failed to load keys from the archive file. Perhaps it did not save correctly. All data in this archive is lost. Please delete the file and try again: '%s'" % self.filename) from e

        while 'file info' in keys: keys.remove('file info')
        return keys
        
    @api
    def keys(self):
        """ Return a list of Archive identifiers from the file. """
        import starsmashertools.helpers.path
        import copy

        keys = []
        if not self.thread_safe:
            # Obtain a copy so that we do not accidentally hand the user
            # our _keys list. This prevents the user from accidentally
            # editing the _keys list.
            keys = copy.deepcopy(self._keys)
        elif starsmashertools.helpers.path.exists(self.filename):
            keys = self._get_keys_from_file()
        
        for key in self._buffers['remove']:
            if key not in keys: continue
            keys.remove(key)
        
        for key in self._buffers['add'].keys():
            if key not in keys: keys += [key]
        
        return keys

    @api
    def items(self):
        """ Return all keys and values in the Archive. This is I/O intensive, 
        so use it sparingly. """
        return ArchiveItems(self, verbose = self.verbose)
    
    @api
    def values(self):
        """ Return all the values in the Archive. This reads the entire Archive
        file, so use it sparingly. """
        return [value for key, value in self.items()]
    
    @api
    @cli('ssarchive')
    @clioptions(confirm = "Are you sure? This will remove all contents and delete the archive file.")
    def clear(self, cli : bool = False):
        """ Clears all contents from the archive and deletes the file if auto
        save is enabled. """
        # We can allow the 'remove' buffer to go over max_buffer_size here,
        # because we kinda have to.
        for key in self.keys():
            if key in self._buffers['remove']: continue
            self._buffers['remove'] += [key]
            
        self._buffers['add'] = {}
        
        if not self.thread_safe: self._keys = []
        
        self._auto_save()

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
            val = self[key]
            for flag in self.replacement_flags:
                if not flag(val, value): return
        
        # Either add a new key or overwrite an old key
        self._buffers['add'][key] = value
        if key in self._buffers['remove']: # Move from 'remove' into 'add'
            self._buffers['remove'].remove(key)
        else: # Only add to 'add'
            self._buffer_size += value.size

        if not self.thread_safe: self._keys += [key]

        self._auto_save()
    
    def __getitem__(self, key):
        """ Obtain the value corresponding with the given key. """
        if not self.thread_safe and key not in self._keys:
            raise KeyError(key)
        
        # If we do have the key in the buffer, fetch its value
        if key in self._buffers['remove']:
            raise KeyError(key)
        if key in self._buffers['add'].keys():
            return self._buffers['add'][key]

        # Check the archive
        try:
            with self.open(
                    'r',
                    message = "Loading key '%s' in %s" % (str(key), self)
            ) as zfile:
                if self.thread_safe:
                    keys = zfile.namelist()
                    while 'file info' in keys: keys.remove('file info')
                else: keys = self._keys
                if key not in keys: raise KeyError(key)
                value = zfile.read(key)
        except Exception as e:
            if isinstance(e, KeyError): raise
            raise Archive.CorruptFileError("Failed to load key '%s' from the archive file. Perhaps it did not save correctly. All data in this archive is lost. Please delete the file and try again: '%s'" % (key, self.filename)) from e
        return ArchiveValue.deserialize(key, value)
    
    def _check_and_update_format(self):
        import starsmashertools.helpers.warnings
        import starsmashertools.helpers.path

        if not starsmashertools.helpers.path.exists(self.filename): return
        
        # We can't update the file if we're in readonly mode.
        if self.readonly: return

        
        should_update = False
        info = None
        try:
            with self.open(
                    'r',
                    message = "Getting file info from %s" % self
            ) as zfile:
                if 'file info' in zfile.namelist():
                    info = zfile.read('file info')
        except Exception as e:
            raise Archive.CorruptFileError("Failed to load file info from archive file. Perhaps it did not save correctly. All data in this archive is lost. Please delete the file and try again: '%s'" % self.filename) from e

        if info:
            info = load_file_info(info)

        old_version = None
        if info is None: should_update = True
        elif starsmashertools._is_version_older(info['__version__']):
            should_update = True
            old_version = info['__version__']

        user_allowed = self.preferences.get('auto update format')

        if old_version is not None:
            message = "%s was written by starsmashertools version '%s', but the current version is '%s'. This Archive will be updated to the latest format."
            message = message % (str(self), old_version, starsmashertools.__version__)
        else:
            message = "%s was written by an older starsmashertools version, but the current version is '%s'. This Archive will be updated to the latest format."
            message = message % (str(self), starsmashertools.__version__)

        if should_update:
            if not user_allowed: raise Archive.FormatError(message)
            starsmashertools.helpers.warnings.warn(message)
            update_archive_version(self.filename, verbose = self.verbose)

    def _auto_save(self, *args, **kwargs):
        if self.auto_save and self._buffer_size >= self._max_buffer_size:
            self.save(*args, **kwargs)

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    #@starsmashertools.helpers.defer_keyboardinterrupt(message = "KeyboardInterrupt raised, but stopping execution might corrupt an archive. Raise KeyboardInterrupt again to stop execution.")
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
        import starsmashertools.helpers.path
        import starsmashertools.helpers.asynchronous
        import starsmashertools.helpers.warnings
        
        if not starsmashertools.helpers.asynchronous.is_main_process():
            starsmashertools.helpers.warnings.warn("Archive.save() is being called by a process that is not the main process. Archive.save() is not thread safe, so make sure you are calling Archive.save() from a single process only. You can suppress this warning with warnings.filterwarnings(action = 'ignore').")
        
        if self.readonly: raise Archive.ReadOnlyError(self.filename)
        if self._nosave_holders > 0: return
        
        exists = starsmashertools.helpers.path.isfile(self.filename)
        
        # If nothing changed since the last time we saved
        if (exists and 
            (not self._buffers['add'] and not self._buffers['remove'])
            ): return
        
        # Serialize the data we will add
        data = {key:val.serialized for key, val in self._buffers['add'].items()}
        
        # Get file info
        info = get_file_info()
        
        # These are the keys that will change in the file
        remove_keys = list(self._buffers['remove']) + list(data.keys())

        directory = starsmashertools.helpers.path.dirname(
            starsmashertools.helpers.path.realpath(self.filename),
        )
        dir_exists = starsmashertools.helpers.path.isdir(directory)
        
        if not dir_exists: starsmashertools.helpers.path.makedirs(directory)
        
        current_keys = []
        try:
            with self.open(
                    'a' if exists else 'w',
                    verbose = verbose,
                    message = "Saving %s" % self,
                    progress_max = len(data.keys()),
            ) as zfile:
                # Update the file info
                while 'file info' in zfile.namelist():
                    try:
                        _remove_zipfile_member(zfile, 'file info')
                    except KeyError: break
                keys = zfile.namelist()
                zfile.writestr('file info', info)
                # Remove all keys that are going to change
                for key in remove_keys:
                    if key not in keys: continue
                    _remove_zipfile_member(zfile, key)
                # Add keys that need to be added
                for key, val in data.items():
                    zfile.writestr(key, val)
                    # Only true when verbose is True and in the main process. See
                    # helpers/file.py for details
                    if hasattr(zfile, 'progress'):
                        zfile.progress.increment()
                current_keys = zfile.namelist()
        except Exception as e:
            print(e)
            raise Archive.CorruptFileError("Failed to save archive file. Perhaps it did not save correctly. All data in this archive is lost. Please delete the file and try again: '%s'" % self.filename) from e
        
        while 'file info' in current_keys: current_keys.remove('file info')

        # Clear the buffers
        self._clear_buffers()

        if not current_keys:
            starsmashertools.helpers.path.remove(self.filename)

        del exists
        del data
        del info
        del remove_keys
        del directory
        del dir_exists
        del current_keys
        del keys
        gc.collect()
    
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
    @cli('ssarchive')
    def set(self, *args, **kwargs):
        """
        A simple alias for :meth:`~.add`.

        See Also
        --------
        :meth:`~.add`
        """
        return self.add(*args, **kwargs)

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get(
            self,
            keys : str | list | tuple,
            deserialize : bool = True,
    ):
        """
        Obtain the key or keys given. This is useful for if you want to retrieve
        many values from the archive without opening and closing the file each
        time. Raises a :py:class:`KeyError` if any of the keys weren't found.

        Parameters
        ----------
        keys : str, list, tuple
            If a `str`, the behavior is equivalent to :meth:`~.__getitem__`. If
            a `list` or `tuple`, each element must be type `str`. Then the
            archive is opened and the values for the given keys are returned in
            the same order as ``keys``.

        Other Parameters
        ----------------
        deserialize : bool, default = True
            If `True`, the values in the :class:`~.Archive` are converted from
            their raw format as stored in the file on the system into Python
            objects. Otherwise, the raw values are returned.

        Returns
        -------
        :class:`~.ArchiveValue` or `list`
            If a `str` was given for ``keys``, a :class:`~.ArchiveValue` is 
            returned. Otherwise, a `list` of :class:`~.ArchiveValue` objects is
            returned.
        """
        import starsmashertools.helpers.string
        
        if isinstance(keys, (list, tuple)):
            for key in keys:
                if isinstance(key, str): continue
                raise TypeError("Argument 'keys' must contain only elements of type 'str', not '%s'" % type(key).__name__)
        else: keys = [keys]

        remaining_keys = []
        values = [None] * len(keys)
        # Search the buffers first before loading the file values
        for i, key in enumerate(keys):
            # If we do have the key in the buffer, fetch its value
            if key in self._buffers['remove']: raise KeyError(key)
            if key in self._buffers['add'].keys():
                values[i] = self._buffers['add'][key]
            else: remaining_keys += [[i, key]]

        if None not in values: # We only had to search the buffers! :)
            return values

        # Check the archive file
        to_deserialize = []
        with self.open(
                'r',
                message = "Loading values in %s" % self,
                progress_max = len(remaining_keys),
        ) as zfile:
            try:
                file_keys = zfile.namelist()
                while 'file info' in file_keys: file_keys.remove('file info')
            except Exception as e:
                raise Archive.CorruptFileError("Failed to load a key from the archive file. Perhaps it did not save correctly. All data in this archive is lost. Please delete the file and try again: '%s'" % (key, self.filename)) from e
                
            for i, key in remaining_keys:
                if key not in file_keys: raise KeyError(key)
                values[i] = zfile.read(key)
                to_deserialize += [[i, key]]
                if hasattr(zfile, 'progress'):
                    zfile.progress.increment()

        keys_to_deserialize = []
        vals_to_deserialize = []
        for i, key in to_deserialize:
            keys_to_deserialize += [key]
            vals_to_deserialize += [values[i]]

        if deserialize:
            deserialized_vals = self._deserialize(
                keys_to_deserialize, vals_to_deserialize,
            )
            for i, key in to_deserialize:
                values[i] = deserialized_vals[i]

        del keys_to_deserialize
        del vals_to_deserialize
        del to_deserialize
        del remaining_keys
        gc.collect()
        
        if len(values) == 1: return values[0]
        return values
            
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    @cli('ssarchive')
    def remove(self, key : str, cli : bool = False):
        """
        Remove an identifier from the Archive. Calls :py:meth:`~.save` if auto
        save is enabled.

        Parameters
        ----------
        key : str
            The identifier key to remove.
        """
        del self[key]
        if cli: return 'Success'
    
    @api
    def combine(self, other):
        """
        Merge the contents of 'other' with this Archive, according to this 
        Archive's replacement flags.

        Parameters
        ----------
        other : str, Archive
            The Archive whose contents will be merged into this Archive. Only 
            this Archive will be affected.
        """
        import copy

        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'other' : [str, Archive],
        })
        
        if isinstance(other, str): other = Archive(
                other,
                thread_safe = self.thread_safe,
                readonly = True,
                auto_save = False,
        )
        previous_auto_save = copy.deepcopy(self.auto_save)
        self.auto_save = False
        for key, val in other.items():
            self[key] = val
        self.auto_save = previous_auto_save
        self._auto_save()

    # Some convenience methods for the CLI only
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @cli('ssarchive')
    def show_keys(self, cli : bool = True):
        import starsmashertools.bintools
        newline = starsmashertools.bintools.Style.get('characters', 'newline')
        return newline.join(sorted(self.keys()))

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @cli('ssarchive')
    def show_value(self, key : str, cli : bool = True):
        import starsmashertools.helpers.jsonfile

        newline = starsmashertools.bintools.Style.get('characters', 'newline')
        
        if key not in self:
            if cli: return "No key '%s' found in the Archive" % key
            else: raise KeyError(key)

        # Use JSON format for human-readable formatting
        return starsmashertools.helpers.jsonfile.view(self[key].value)
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @cli('ssarchive')
    def show_raw_value(self, key : str, cli : bool = True):
        import starsmashertools.bintools
        
        if key not in self:
            if cli: return "No key '%s' found in the Archive" % key
            else: raise KeyError(key)

        newline = starsmashertools.bintools.Style.get('characters', 'newline')
        
        val = self[key]
        header = newline.join([
            'origin: ' + str(val.origin),
            'mtime: ' + str(val.mtime),
        ])
        
        return ("Showing '%s':" % key) + newline + newline + header + newline + newline + str(val.value)
        

    def _deserialize(self, keys : list, values : list):
        """ Deserialize many :class:`~.ArchiveValue` objects efficiently. Uses
        parallel processing to speed things up. """
        import starsmashertools.helpers.asynchronous
        import starsmashertools.helpers.string

        nprocs = min(
            starsmashertools.helpers.asynchronous.max_processes,
            len(keys),
        )

        def func(progress_message = None):
            ret = []
            if nprocs == 1 or not starsmashertools.helpers.asynchronous.is_main_process():
                if progress_message:
                    progress_message.message += " in serial"
                # Run in serial
                for key, value in zip(keys, values):
                    ret += [ArchiveValue.deserialize(key, value)]
                    progress.increment()
            else:
                if progress_message:
                    progress_message.message += " on %d processes" % nprocs
                arguments = [[key, value] for key, value in zip(keys, values)]
                p = starsmashertools.helpers.asynchronous.ParallelFunction(
                    target = ArchiveValue.deserialize,
                    args = arguments,
                    nprocs = nprocs,
                    progress_message = progress,
                    start = True,
                )
                ret = p.get_output()
                del p
            return ret

        if self.verbose:
            with starsmashertools.helpers.string.progress_message(
                    message = "Deserializing archive values",
                    max = len(keys), delay = 5,
            ) as progress:
                ret = func(progress_message = progress)
        else: ret = func()
        return ret

    def find_old_values(self):
        """
        Using :property:`~.replacement_flags`, locate the 
        :class:`~.ArchiveValue` objects whose origins have a newer modification
        time than that which is currently saved in the :class:`~.Archive`. This
        is useful for detecting which values need to be updated in an Archive.
        
        Ignores any :class:`~.ArchiveValue` object with an origin that is not 
        currently on the file system or whose origin is `None`, or whose mtime
        value is `None`.

        Returns
        -------
        list
            A list of :class:`~.ArchiveValue` objects that are out-of-date
            compared to the files currently on the system.
        """
        import starsmashertools.helpers.path
        old_values = []
        for value in self.values():
            if value.origin is None: continue
            if value.mtime is None: continue
            if not starsmashertools.helpers.path.exists(value.origin): continue
            current_mtime = starsmashertools.helpers.path.getmtime(value.origin)
            if current_mtime > value.mtime: old_values += [value]
        return old_values


@starsmashertools.preferences.use
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
    
    @api
    def __init__(self, simulation):
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

        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'simulation' : [str, starsmashertools.lib.simulation.Simulation],
        })
        
        basename = self.preferences.get('filename')
        
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

        del basename
        del filename
