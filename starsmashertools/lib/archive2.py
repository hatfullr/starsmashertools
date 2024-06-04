import starsmashertools.helpers.path
import starsmashertools.helpers.file
import starsmashertools.helpers.argumentenforcer
import pathlib
import pickle
import struct
import time
import base64
import collections
import os
import io

FOOTERSTRUCT = struct.Struct('<Q')

def is_pickled(value):
    return isinstance(value, bytes) and value.endswith(pickle.STOP)

def clean_base64(value):
    try: ret = base64.b64decode(value)
    except: return value
    try: base64.b64encode(ret) # Fails if it wasn't encoded with base64
    except: return value
    else: return ret

class InvalidArchiveError(Exception): pass
class CorruptArchiveError(InvalidArchiveError): pass

class Buffer(io.BytesIO, object):
    def __init__(
            self,
            path : str | pathlib.Path,
            readonly : bool = False,
    ):
        self._closed = False
        if not starsmashertools.helpers.path.exists(path): mode = 'wb+'
        else: mode = 'rb' if readonly else 'rb+'
        self.lock = starsmashertools.helpers.file.Lock(path, mode)
        
        self._f = open(path, mode)
    
    @property
    def closed(self): return self._closed

    def __str__(self):
        # Show a helpful string which gives a window around the current position
        curpos = self.tell()
        
        self.seek(0, 2)
        buffer_size = self.tell()
        self.seek(curpos)
        
        self.seek(-min(20, buffer_size - (buffer_size - curpos)), 1)
        start = self.tell()
        window = self.read(min(40, buffer_size - 1))
        stop = self.tell()

        self.seek(start)
        left_content = self.read(curpos - start)
        right_content = self.read(stop - curpos)
        first_string = str(left_content + b' ' + right_content)[2:-1]
        bottom_string = '^'*len(str(left_content)[2:-1]) + ' '*len(str(right_content)[2:-1]) + '\n' + ' '*len(str(left_content)[2:-len('pos')]) + 'pos'

        self.seek(curpos)
        
        return '\n' + first_string + '\n' + bottom_string

    def __del__(self, *args, **kwargs):
        try: self.close()
        except Exception: pass
        try: return super(Buffer, self).__del__(*args, **kwargs)
        except AttributeError: pass

    def __getitem__(self, index):
        os.fsync(self._f.fileno())
        if isinstance(index, slice):
            start, stop, step = index.indices(self.size())
            self.seek(start, os.SEEK_SET)
            return self.read(stop - start)[::step]
        else:
            self.seek(index, os.SEEK_SET)
            return self.read(1)
    
    def __setitem__(self, index, value):
        os.fsync(self._f.fileno())
        with io.BytesIO(value) as other_buffer:
            if isinstance(index, slice):
                if index.step in [None, 1]:
                    index = index.indices(self.size())
                    self.seek(index[0], os.SEEK_SET)
                    self.write(other_buffer.read())
                else:
                    for i in range(*index.indices(self.size())):
                        self.seek(i, os.SEEK_SET)
                        self.write(other_buffer.read(1))
            else:
                self.seek(index, os.SEEK_SET)
                self.write(value.read())

    def seek(self, *args, **kwargs): return self._f.seek(*args, **kwargs)
    def tell(self, *args, **kwargs): return self._f.tell(*args, **kwargs)
    def read(self, *args, **kwargs): return self._f.read(*args, **kwargs)
    def write(self, *args, **kwargs): return self._f.write(*args, **kwargs)
    
    def flush(self):
        with self.lock:
            os.fsync(self._f.fileno())
            self._f.flush()
    
    def close(self):
        try: self.flush()
        except ValueError as e:
            if str(e) != 'I/O operation on closed file': raise
        self._f.close()
        self._closed = True
    
    def size(self):
        with self.lock:
            curpos = self.tell()
            self.seek(0, os.SEEK_END)
            ret = self.tell()
            self.seek(curpos, os.SEEK_SET)
            return ret
    
    def resize(self, newsize : int):
        with self.lock:
            self._f.truncate(newsize)
            self.flush()


class Archive(object):
    """
    Make sure that when you are using this class with multiple processes or
    threads that you implement a lock whenever you get/set values.
    """
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def __init__(
            self,
            path : str | pathlib.Path,
            auto_save : bool = True,
            readonly : bool = False,
    ):
        self.auto_save = auto_save
        self.set_path(path, readonly = readonly)

    @property
    def path(self):
        if hasattr(self, '_path'): return self._path
        return None

    def __repr__(self): return 'Archive(%s)' % str({key:val for key,val in self.items()})
    
    def __del__(self, *args, **kwargs):
        if hasattr(self, '_buffer'):
            if not self._buffer.closed:
                self._buffer.close()
    
    def __setitem__(self, key, value):
        if not is_pickled(value): value = pickle.dumps(value)
        
        # If this key is already in the Archive, get its location and size.
        footer, footer_size = self.get_footer()
        
        if key in footer:
            pos = footer[key]['pos']
            size = footer[key]['size']
            
            if size == len(value): # Lucky!
                self._buffer[pos : pos + size] = value
                footer[key]['mtime'] = time.time()
                if self.auto_save: self._buffer.flush()
                return
            
            if size > len(value):
                # Move everything "up"
                self._buffer[pos:] = value + self._buffer[pos+size:] + b'\x00'*(size - len(value))
                footer[key]['mtime'] = time.time()
                
                self._buffer.resize(self._buffer.size() - size + len(value))
                
            else: # size < len(value)
                length = self._buffer.size()
                self._buffer.resize(length - size + len(value))
                
                # Move everything "down"
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
            
            new_pos = self._buffer.size() - len(value) - footer_size
            footer[key] = { 'pos' : new_pos, 'n' : 1 }
            self._buffer[new_pos:new_pos + len(value)] = value
            footer[key]['time'] = time.time()
            
        footer[key]['size'] = len(value)
        
        # Write the footer
        self._update_footer(footer, footer_size)
        
        if self.auto_save: self._buffer.flush()
    
    def __getitem__(self, key):
        footer, _ = self.get_footer()
        self._buffer.seek(footer[key]['pos'])
        if footer[key]['n'] == 1:
            return pickle.load(self._buffer)
        else:
            def get():
                for _ in range(footer[key]['n']):
                    yield pickle.load(self._buffer)
            return get()

    def __delitem__(self, key):
        """ Remove the given key from the Archive. If auto save is enabled, the
        Archive file will be modified. If there are any values appended to this
        item, they are also removed. """
        footer, footer_size = self.get_footer()
        if key not in footer: raise KeyError(key)
        pos = footer[key]['pos']
        size = footer[key]['size']
        self._buffer[pos:] = self._buffer[pos + size:] + b'\x00'*size
        self._buffer.resize(self._buffer.size() - size)
        
        del footer[key]

        # Update the footer
        self._update_footer(footer, footer_size)
        
        if self.auto_save: self._buffer.flush()
        
    
    def __contains__(self, key): return key in self.get_footer()[0]
    def __len__(self): return len(self.get_footer()[0])

    def _update_footer(self, footer, footer_size):
        if not footer:
            # If there's no contents in the footer, it's safe to empty the file
            self._buffer.resize(0)
            return
        
        new_footer = pickle.dumps(footer)
        new_footer_size = len(new_footer) + FOOTERSTRUCT.size
        self._buffer.resize(self._buffer.size() - footer_size + new_footer_size)
        
        self._buffer[-new_footer_size:] = new_footer + FOOTERSTRUCT.pack(len(new_footer))

    def set_path(
            self,
            path : str | pathlib.Path,
            readonly : bool = False,
    ):
        if hasattr(self, '_buffer'):
            self.save()
            self._buffer.close()
        self._buffer = Buffer(path, readonly = readonly)
        self._path = path
    
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
        for _ in range(len(footer.keys())):
            yield pickle.load(self._buffer)
    def items(self): return zip(self.keys(), self.values())

    def clear(self):
        self._buffer.resize(0)
    
    def get_footer(self):
        try: self._buffer.seek(-FOOTERSTRUCT.size, 2)
        except OSError as e:
            if 'Invalid argument' not in str(e): raise
            else: return collections.OrderedDict({}), 0

        size = FOOTERSTRUCT.unpack(self._buffer.read())[0] + FOOTERSTRUCT.size

        try: self._buffer.seek(-size, 2)
        except OSError as e:
            if 'Invalid argument' not in str(e): raise
            else: return collections.OrderedDict({}), 0
        
        return pickle.load(self._buffer), size

    def save(self):
        self._buffer.flush()

    def append(self, key : str, value):
        """
        Append a value to the end of an existing key. Calls to 
        :meth:`~.__getitem__` will return a generator all appended items. The
        ``key`` being appended to will have its mtime replaced with the current
        timestamp.
        """
        footer, footer_size = self.get_footer()
        if key not in footer.keys(): raise KeyError(key)

        if not is_pickled(value): value = pickle.dumps(value)
        
        pos = footer[key]['pos']
        size = footer[key]['size']
        lenvalue = len(value)

        # Move everything "down"
        self._buffer.resize(self._buffer.size() + lenvalue)
        self._buffer[pos + size:] = value + self._buffer[pos + size:-lenvalue]
        
        footer[key]['mtime'] = time.time()
        footer[key]['size'] += lenvalue
        footer[key]['n'] += 1
        after_key = False
        for k, val in footer.items():
            if k == key:
                after_key = True
                continue
            if not after_key: continue
            footer[key]['pos'] += lenvalue
        self._update_footer(footer, footer_size)

    def add(self, *args, **kwargs): return self.set(*args, **kwargs)
        
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def set(
            self,
            key : str,
            value,
            origin : str | pathlib.Path | type(None) = None,
            replace : str | type(None) = 'mtime',
    ):
        """
        If ``replace = 'mtime'``\, then the modification time of ``origin`` is
        checked against the currently stored ``mtime`` in the archive. If the
        mtime value is greater, then the key is replaced in this archive.
        Otherwise, no replacement happens. If ``origin`` is `None`\, the value
        of ``replace`` is ignored and the key is always replaced if it exists.

        If ``replace`` is `None`\, the value is always replaced.

        Removes any appended values to ``key``\.
        """
        starsmashertools.helpers.argumentenforcer.enforcevalues({
            'replace' : ['mtime',],
        })
        
        footer, _ = self.get_footer()
        if key in footer:
            if None in [origin, replace]:
                self[key] = value
            elif replace == 'mtime':
                if footer[key]['mtime'] < starsmashertools.helpers.path.getmtime(origin):
                    self[key] = value
            else: raise NotImplementedError(replace)

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
