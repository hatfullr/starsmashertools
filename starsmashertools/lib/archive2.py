import starsmashertools.helpers.path
import starsmashertools.helpers.file
import starsmashertools.helpers.argumentenforcer
import pathlib
import pickle
import struct
import mmap
import time
import base64
import collections

FOOTERSTRUCT = struct.Struct('<Q')

def is_pickled(value):
    return isinstance(value, bytes) and value[-1] == pickle.STOP

def clean_base64(value):
    try: ret = base64.b64decode(value)
    except: return value
    try: base64.b64encode(ret) # Fails if it wasn't encoded with base64
    except: return value
    else: return ret

class InvalidArchiveError(Exception): pass

class Archive(object):
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def __init__(
            self,
            path : str | pathlib.Path,
            auto_save : bool = True,
            readonly : bool = False,
            madvise : list | tuple = [
                mmap.MAP_SHARED, # share between processes
                mmap.MADV_HUGEPAGE,
            ],
    ):
        self.auto_save = auto_save
        self.madvise = madvise
        
        # Obtain the file memory buffer
        if not starsmashertools.helpers.path.exists(path):
            # Kickstart the file to avoid mmap 'empty file' error
            with open(path, 'wb') as f: f.write(b' ')
        with open(path, 'rb' if readonly else 'rb+') as f:
            self._buffer = mmap.mmap(
                f.fileno(),
                0,
                access = mmap.ACCESS_READ if readonly else mmap.ACCESS_WRITE
            )
        
        # Indicate to the file system that the buffer is going to be edited, and
        # that those edits should be reflected in any other process which has
        # mapped this file. Note that this should allow multiple Archive
        # instances to access and modify a single, even if they are on different
        # processes.
        for madv in self.madvise: self._buffer.madvise(madv)

    def __del__(self, *args, **kwargs):
        if hasattr(self, '_buffer'):
            if not self._buffer.closed:
                self._buffer.flush()
                self._buffer.close()
    
    def __setitem__(self, key, value):
        # If the value isn't already pickled, pickle it now
        if not is_pickled(value): value = pickle.dumps(value)
        
        # If this key is already in the Archive, get its location and size.
        footer, footer_size = self.get_footer()
        
        if key in footer:
            pos = footer[key]['pos']
            size = footer[key]['size']
            new_pos = pos

            if size == len(value): # Lucky!
                self._buffer[pos:pos + size] = value
                footer[key]['mtime'] = time.time()
                if self.auto_save:
                    # Only flush what we have changed
                    offset = mmap.PAGESIZE * (pos // mmap.PAGESIZE)
                    self._buffer.flush(offset, offset + size)
                return

            # Not clear to me yet if seeking and writing is appropriate or if
            # modifying the buffer is appropriate.

            #self._buffer.seek(pos)
            
            if size > len(value):
                # Move everything "up"
                #self._buffer.write(value + self._buffer[pos + size:])
                self._buffer[pos:] = value + self._buffer[pos+size:] + b' '*(size - len(value))
                footer[key]['mtime'] = time.time()
                
                self._buffer.resize(self._buffer.size() - size + len(value))
                for madv in self.madvise: self._buffer.madvise(madv)
                
            else: # size < len(value)
                length = self._buffer.size()
                self._buffer.resize(length - size + len(value))
                for madv in self.madvise: self._buffer.madvise(madv)
                
                # Move everything "down"
                #self._buffer.write(value + self._buffer[pos + size : length])
                self._buffer[pos:] = value + self._buffer[pos + size:length]
                footer[key]['mtime'] = time.time()

            # All the footers that are after this key now have their locations
            # modified. So we update them all here.
            diff = len(value) - size
            if diff != 0:
                isafter = False
                for k, obj in footer.items():
                    if k == key:
                        isafter = True
                        continue
                    if not isafter: continue
                    footer[k]['pos'] += diff
        
        else: # This is an entirely new key
            current_size = self.size()
            
            self._buffer.resize(current_size + len(value))
            for madv in self.madvise: self._buffer.madvise(madv)
            
            new_pos = self._buffer.size() - len(value) - footer_size
            footer[key] = { 'pos' : new_pos }
            self._buffer[new_pos:new_pos + len(value)] = value
            #self._buffer.seek(new_pos)
            #self._buffer.write(value)
            footer[key]['time'] = time.time()
            
        footer[key]['size'] = len(value)
        
        # Write the footer
        self._update_footer(footer, footer_size)
        #self._buffer.seek(-new_footer_size, 2)
        #self._buffer.write(new_footer + FOOTERSTRUCT.pack(len(new_footer)))
        
        if self.auto_save:
            # Only flush what we have changed
            offset = mmap.PAGESIZE * (footer[key]['pos'] // mmap.PAGESIZE)
            self._buffer.flush(offset, self._buffer.size() - offset)
    
    def __getitem__(self, key):
        footer, _ = self.get_footer()
        self._buffer.seek(footer[key]['pos'])
        return pickle.load(self._buffer)

    def __delitem__(self, key):
        """ Remove the given key from the Archive. If auto save is enabled, the
        Archive file will be modified. """
        footer, footer_size = self.get_footer()
        if key not in footer: raise KeyError(key)
        pos = footer[key]['pos']
        size = footer[key]['size']
        self._buffer[pos:-size] = self._buffer[pos + size:]
        self._buffer.resize(self._buffer.size() - size)
        for madv in self.madvise: self._buffer.madvise(madv)
        
        del footer[key]

        # Update the footer
        self._update_footer(footer, footer_size)
        
        if self.auto_save:
            offset = mmap.PAGESIZE * (pos // mmap.PAGESIZE)
            self._buffer.flush(offset, self._buffer.size() - offset)
        
    
    def __contains__(self, key): return key in self.get_footer()[0]

    def _update_footer(self, footer, footer_size):
        new_footer = pickle.dumps(footer)
        new_footer_size = len(new_footer) + FOOTERSTRUCT.size
        self._buffer.resize(self._buffer.size() + - footer_size + new_footer_size)
        for madv in self.madvise: self._buffer.madvise(madv)
        
        self._buffer[-new_footer_size:] = new_footer + FOOTERSTRUCT.pack(len(new_footer))

    def size(self):
        # This is an artifact from when we first made the file
        if self._buffer.size() == 1: return 0
        return self._buffer.size()

    def keys(self): return self.get_footer()[0].keys()
    def values(self):
        """ Returns a generator which iteratively obtains all the archived 
        values. """
        footer, footer_size = self.get_footer()
        self._buffer.seek(0)
        while self._buffer.tell() < self._buffer.size() - footer_size:
            yield pickle.load(self._buffer)
    def items(self): return zip(self.keys(), self.values())
    
    def get_footer(self):
        if self._buffer.size() < FOOTERSTRUCT.size: size = 0
        else:
            self._buffer.seek(-FOOTERSTRUCT.size, 2)
            size = FOOTERSTRUCT.unpack(self._buffer.read())[0] + FOOTERSTRUCT.size
        self._buffer.seek(-size, 2)
        try: return pickle.load(self._buffer), size
        except: return collections.OrderedDict({}), size

    def save(self):
        self._buffer.flush()

    def add(self, *args, **kwargs): return self.set(*args, **kwargs)
        
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def set(
            self,
            key,
            value,
            origin : str | pathlib.Path | type(None) = None,
            replace : str = 'mtime',
    ):
        """
        If ``replace = 'mtime'``\, then the modification time of ``origin`` is
        checked against the currently stored ``mtime`` in the archive. If the
        mtime value is greater, then the key is replaced in this archive.
        Otherwise, no replacement happens. If ``origin`` is `None`\, the value
        of ``replace`` is ignored and the key is always replaced if it exists.
        """
        starsmashertools.helpers.argumentenforcer.enforcevalues({
            'replace' : ['mtime',],
        })
        
        footer, _ = self.get_footer()
        if key in footer:
            if origin is None:
                self[key] = value
            else:
                if replace == 'mtime':
                    if footer[key]['mtime'] < starsmashertools.helpers.path.getmtime(origin):
                        self[key] = value

    def get(
            self,
            keys : str | list | tuple,
            deserialize : bool = True,
    ):
        """
        Obtain the key or keys given. This is useful for if you want to retrieve
        many values from the archive without opening and closing the file each
        time. Raises a :class:`KeyError` if any of the keys weren't found.

        Parameters
        ----------
        keys : str, list, tuple
            If a `str`\, the behavior is equivalent to 
            :meth:`~.Archive.__getitem__`\. If a `list` or `tuple`\, each 
            element must be type `str`\. Then the archive is opened and the 
            values for the given keys are returned in the same order as 
            ``keys``\.

        Other Parameters
        ----------------
        deserialize : bool, default = True
            If `True`\, the values in the :class:`~.Archive` are converted from
            their raw format as stored in the file on the system into Python
            objects. Otherwise, the raw values are returned.

        Returns
        -------
        :py:class:`object` or generator
            If a `str` was given for ``keys``\, a :py:class:`object` is
            returned. Otherwise, a generator is returned which yields the values
            in the archive at the given keys.
        """
        
        footer, _ = self.get_footer()
        
        if isinstance(keys, str):
            if not deserialize:
                return self._buffer[footer[keys]['pos']:footer[keys]['pos']+footer[keys]['size']]
            return self[keys]

        def gen():
            for key in keys:
                self._buffer.seek(footer[key]['pos'])
                yield pickle.load(self._buffer)
        
        return gen()
