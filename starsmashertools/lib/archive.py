import starsmashertools.helpers.jsonfile
import starsmashertools.helpers.argumentenforcer
import starsmashertools.helpers.path
from starsmashertools.helpers.apidecorator import api
import zipfile
import copy
import inspect
import warnings

class Archive(dict, object):
    """
    An Archive object stores :class:`ArchiveValue`s in a single file for quick 
    access.
    
    Examples
    --------
    Suppose you have many StarSmasher output files and you want to store the
    total number of particles each one has in a file called ``mydata.dat``:
    
        import starsmashertools.lib.archive
        import starsmashertools
    
        def get_ntot(output):
            return output['ntot']
        
        simulation = starsmashertools.get_simulation(".")

        archive = starsmashertools.lib.archive.Archive("mydata.dat")
        for output in simulation.get_output_iterator():
            archive.add(
                'ntot.%s' % output.path, # A unique identifier
                output['ntot'],
                output.path,
            )
        archive.save()

    """
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(
            self,
            filename : str,
            load : bool = True,
    ):
        self.filename = filename

        super(Archive, self).__init__()

        if load and starsmashertools.helpers.path.isfile(self.filename):
            self.load()


    def __setitem__(self, identifier, value):
        frame = inspect.currentframe().f_back
        if frame.f_code.co_filename not in [__file__, copy.__file__]:
            raise Exception("Setting values of an Archive is forbidden. You must use the Archive.add() and Archive.remove() functions instead.")
        return super(Archive, self).__setitem__(identifier, value)

    def load(self):
        with zipfile.ZipFile(self.filename, mode='r') as zfile:
            data = zfile.read('data')
        obj = starsmashertools.helpers.jsonfile.load_bytes(data)
        self._from_json(obj)

    def save(self):
        data = starsmashertools.helpers.jsonfile.save_bytes(self._to_json())
        zinfo = zipfile.ZipInfo('data')
        with zipfile.ZipFile(self.filename, mode='a', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zfile:
            zfile.writestr(zinfo, data)

    def _to_json(self):
        cpy = copy.deepcopy(self)
        for identifier, val in self.items():
            cpy[identifier] = val._to_json()
        return cpy

    def _from_json(self, json):
        for identifier, val in json.items():
            self[identifier] = ArchiveValue._from_json(val)

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def add(self, identifier : str, *args, **kwargs):
        """
        Create a new :class:`ArchiveValue` and add it to the archive. If there
        already exists a :class:`ArchiveValue` with the same `identifier` in the
        archive, then if the origin file is different we overwrite the old
        value. Otherwise, if the file's modification time is more recent than
        the archived value, we also overwrite the old value.

        Parameters
        ----------
        identifier : str
            The string by which this new value will be known in the archive.

        Other Parameters
        ----------
        *args
            Positional arguments are passed directly to :class:`ArchiveValue`.

        **kwargs
            Keyword arguments are passed directly to :class:`ArchiveValue`.
        """

        value = ArchiveValue(*args, **kwargs)
        if identifier in self.keys():
            if value.is_newer_than(self[identifier]):
                self[identifier] = value
            else:
                warnings.warn("'%s' is older than the archived value '%s' so it was not added to archive '%s'" % (str(value), str(self[identifier]), self.filename))
        else:
            self[identifier] = value
            
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def remove(self, identifier : str):
        if identifier not in self.keys():
            raise KeyError("No identifier '%s' found in Archive '%s'" % (identifier, self.filename))
        del self[identifier]









        


class ArchiveValue(object):
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(
            self,
            value,
            origin : str,
            mtime : int | float | type(None) = None
    ):
        """
        ArchiveValue constructor.

        Parameters
        ----------
        value : serializable types
            A value that is one of the serializable types, found in the keys of
            :py:property:`~.helpers.jsonfile.serialization_methods`.
        
        origin : str
            Path to the file from which this value originated.

        mtime : int, float, None, default = None
            The modification time of the file specified by `origin`. If not
            `None` then `origin` can be a file which doesn't currently exist.
        """
        
        _types = starsmashertools.helpers.jsonfile.serialization_methods.keys()
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'value' : _types,
        })

        self.value = value
        self.origin = origin

        if mtime is None:
            if not starsmashertools.helpers.path.isfile(self.origin):
                raise FileNotFoundError(self.origin)
            mtime = starsmashertools.helpers.path.getmtime(self.origin)
        
        self.mtime = mtime

    def __str__(self):
        return 'ArchiveValue(%s)' % str(self.value)

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def is_newer_than(self, other : 'ArchiveValue'):
        """
        Returns `True` if this :class:`ArchiveValue` is more recent than
        `other`.

        Parameters
        ----------
        other : ArchiveValue
            The value to compare to this one.

        Returns
        -------
        bool
            `True` if this value is more recent than `other` and `False`
            otherwise. Returns `False` if both values have the same modification
            times.
        """
        return self.mtime > other.mtime


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
    def _from_json(json):
        """
        Return a :class:`ArchiveValue` object from the given json object.
        """
        return ArchiveValue(
            json['value'],
            json['origin'],
            json['mtime'],
        )
    
