import starsmashertools.helpers.argumentenforcer
from starsmashertools.helpers.apidecorator import api
import contextlib

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

    def _to_json(self):
        """
        Return a copy of this value in a JSON serializable format.
        """
        return {
            'value' : self.value,
            'origin' : self.origin,
            'mtime' : self.mtime,
        }

    @staticmethod
    def _from_json(identifier, json):
        """
        Return a :class:`ArchiveValue` object from the given json object.
        """
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

        

class Archive(dict, object):
    """
    An Archive object stores :class:`ArchiveValue`s in a single file for quick 
    access. Note that :class:`ArchiveValue` can only store data that is JSON
    serializable. This typically includes only standard types str, bool, float,
    int, list, tuple, and dict. However, additional types such as np.ndarray,
    np.integer_, etc. can also be stored. See
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

    class ReadOnlyError(Exception, object):
        def __init__(self, filename):
            message = "Cannot modify readonly archive: %s" % str(self)
            super(Archive.ReadOnlyError, self).__init__(message)
    class CorruptFileError(Exception, object): pass
    class MissingIdentifierError(Exception, object): pass
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(
            self,
            filename : str,
            load : bool = True,
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
        load : bool, default = True
            If `False` then the archive will not be loaded on initialization,
            so it must be loaded manually using :func:`~.load`.

        replacement_flags : list, tuple, None, default = None
            A list of flags to use to determine if a replacement should happen
            in the archive whenever a new :class:`ArchiveValue` is about to be
            written to a pre-existing identifier. If `None` then the default
            from :py:property:`~.preferences.defaults` under 'Archive' and
            'replacement flags' will be used.
            Each element of in `replacement_flags` must be a function which
            accepts two arguments, each is `:class:`~.ArchiveValue`. The first
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

        self.loaded = False
        
        if load and starsmashertools.helpers.path.isfile(self.filename):
            self.load()

    def __repr__(self):
        import starsmashertools.helpers.string
        name = starsmashertools.helpers.string.shorten(
            self.filename,
            25,
            where = 'left',
        )
        return 'Archive(%s)' % name
    
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
        """A context manager for temporarily disabling the auto save feature"""
        import copy
        was_auto_save = copy.deepcopy(self.auto_save)
        try:
            self.auto_save = False
            yield
        finally:
            self.auto_save = copy.deepcopy(was_auto_save)

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __contains__(
            self,
            key : str | ArchiveValue,
    ):
        if isinstance(key, str): return key in self.keys()
        else: return key in self.values()

    @api
    def __delitem__(self, key):
        """ The same as dict.__delitem__ except the archive file is modified. 
        If the archive has no more contents then its file is deleted. """
        ret = super(Archive, self).__delitem__(key)
        if self.auto_save: self.save()
        return ret

    @api
    def clear(self, *args, **kwargs):
        """ The same as dict.clear except the archive file can is deleted when 
        ``auto_save == True``. """
        import starsmashertools.helpers.path
        ret = super(Archive, self).clear(*args, **kwargs)
        if self.auto_save:
            starsmashertools.helpers.path.remove(self.filename)
        return ret

    @api
    def pop(self, *args, **kwargs):
        """ The same as dict.pop except the archive file is modified. """
        ret = super(Archive, self).pop(*args, **kwargs)
        if self.auto_save: self.save()
        return ret

    @api
    def popitem(self, *args, **kwargs):
        """ The same as dict.popitem except the archive file is modified. """
        ret = super(Archive, self).popitem(*args, **kwargs)
        if self.auto_save: self.save()
        return ret

    @api
    def setdefault(self, *args, **kwargs):
        """ The same as dict.setdefault except the archive file is modified. """
        ret = super(Archive, self).setdefault(*args, **kwargs)
        if self.auto_save: self.save()
        return ret
        
    @api
    def update(self, *args, **kwargs):
        """ The same as dict.update except the archive file is modified. """
        ret = super(Archive, self).update(*args, **kwargs)
        if self.auto_save: self.save()
        return ret

    @api
    def reset(self):
        """ Reload this archive from the archive file. """
        self.load()
    
    @api
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def __setitem__(
            self,
            key,
            value : ArchiveValue,
    ):
        if key in self.keys():
            if all([flag(self[key], value) for flag in self.replacement_flags]):
                super(Archive, self).__setitem__(key, value)
                if self.auto_save: self.save()
        else:
            super(Archive, self).__setitem__(key, value)
            if self.auto_save: self.save()

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def load(
            self,
            verbose : bool | type(None) = None,
    ):
        """
        Load this archive from its filename.

        Parameters
        ----------
        verbose : bool, None, default = None
            Overrides the :py:attr:`~.verbose` option in :py:func:`~.__init__`.
            If `None` then :py:attr:`~.verbose` is used instead.
        """
        import zipfile
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
                    self.filename, 'r', method = zipfile.ZipFile,
                    compression = zipfile.ZIP_DEFLATED,
                    compresslevel = 9, lock=True,
            ) as zfile:
                namelist = zfile.namelist()
                if not namelist:
                    raise Archive.CorruptFileError("Failed to load archive file because it did not save correctly. Please delete it and try again: '%s'" % self.filename)
                content = zfile.read(namelist[0])

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
        
        data = starsmashertools.helpers.jsonfile.load_bytes(data)
        with self.nosave():
            self.clear()
            for identifier, val in data.items():
                self[identifier] = ArchiveValue._from_json(identifier, val)
        
        self.loaded = True

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def save(
            self,
            verbose : bool | type(None) = None,
    ):
        """
        Overwrite the archive file with the current data.

        Parameters
        ----------
        verbose : bool, None, default = None
            Overrides the :py:attr:`~.verbose` option in :py:func:`~.__init__`.
            If `None` then :py:attr:`~.verbose` is used instead.
        """
        if self.readonly: raise Archive.ReadOnlyError(self.filename)
        import zipfile
        import starsmashertools.helpers.jsonfile
        import starsmashertools.helpers.file
        import starsmashertools.helpers.path
        import starsmashertools.helpers.string
        if verbose is None: verbose = self.verbose
        
        data = {key: val._to_json() for key, val in self.items()}
        
        # Convert the data to a readable JSON format which supports
        # serialization of many types
        datastr = starsmashertools.helpers.jsonfile.save_bytes(data)

        def do(*args, **kwargs):
            with starsmashertools.helpers.file.open(
                    self.filename, 'w', method = zipfile.ZipFile,
                    compression = zipfile.ZIP_DEFLATED,
                    compresslevel = 9,
                    lock = True,
            ) as zfile:
                zfile.writestr('archive', datastr)

        if verbose:
            message = "Saving '%s'" % starsmashertools.helpers.string.shorten(
                self.filename,
                50,
                where = 'left',
            )
            with starsmashertools.helpers.string.loading_message(message,delay=5):
                do()
        else: do()
    

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
        if identifier not in self.keys():
            raise Archive.MissingIdentifierError("No identifier '%s' found in Archive '%s'" % (identifier, self.filename))
        del self[identifier]


    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def combine(
            self,
            other : 'str | Archive',
    ):
        with self.nosave():
            if isinstance(other, str): other = Archive(other)
            self.update(other)
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
            simulation : "starsmashertools.lib.simulation.Simulation",
            replacement_flags : list | tuple | type(None) = None,
    ):
        """
        Constructor for a :class:`SimulationArchive`. Note that a
        :class:`~.SimulationArchive` is not directly loaded or saved. This
        rather happens automatically as archived values are accessed / modified.

        Parameters
        ----------
        simulation : :class:`starsmashertools.lib.simulation.Simulation`
            The simulation to which this archive belongs.
        
        replacement_flags : list, tuple, None, default = None
            A collection of functions which return boolean operators to
            determine if a new :class:`~.ArchiveValue` should overwrite an old
            one with the same identifier. See :func:`~.Archive.__init__` for a 
            full description. 
        """
        import starsmashertools.helpers.path
        import starsmashertools.preferences

        # Ensure that the filename is located in the simulation directory
        filename = starsmashertools.helpers.path.join(
            simulation.directory,
            starsmashertools.preferences.get_default(
                'Simulation',
                'archive filename',
                throw_error = True,
            ),
        )
        super(SimulationArchive, self).__init__(
            filename,
            load = True,
            replacement_flags = replacement_flags,
            auto_save = True,
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

        
