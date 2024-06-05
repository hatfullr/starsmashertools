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
import tempfile
import numpy as np
import mmap
import copy

def serialize(value):
    if is_pickled(value): return value
    if isinstance(value, np.ndarray):
        # These take up an enormous amount of memory. We compress them
        # instead.
        return pickle.dumps(NumPyArray(value))
    return pickle.dumps(value)

def deserialize(value):
    if isinstance(value, NumPyArray): return value.value
    if isinstance(value, List): return list(value)
    if not is_pickled(value): return value
    return pickle.loads(value)

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
class ReadOnlyError(Exception): pass

class archive_infos(collections.abc.KeysView):
    def __init__(self, mapping):
        if not isinstance(mapping, Archive):
            raise TypeError("{0.__class__.__name__} can only be used with mappables of type Archive, not '{:s}'".format(self, type(mapping).__name__))
        self.mapping = mapping
        self._length = None

    @staticmethod
    def _get_mapping_gen(_buffer):
        _buffer.seek(0, os.SEEK_SET)
        while True:
            try: info = pickle.load(_buffer)
            except EOFError: return
            yield info
            _buffer.seek(info.size, os.SEEK_CUR)

    @property
    def _mapping(self):
        if not hasattr(self.mapping, '_buffer'): return []
        return self._get_mapping_gen(self.mapping._buffer)
    def __len__(self):
        if self._length is None: self._length = sum([1 for _ in self._mapping])
        return self._length
    def __contains__(self, obj):
        if not is_pickled(obj): obj = pickle.dumps(obj)
        for info in self._mapping:
            if info == obj: return True
        return False
collections.abc.KeysView.register(archive_infos)


class archive_keys(archive_infos):
    @staticmethod
    def _get_mapping_gen(_buffer):
        return [info.key for info in archive_infos._get_mapping_gen(_buffer)]
    
    def __contains__(self, key):
        for info in archive_infos(self.mapping):
            if info.key == key: return True
        return False
collections.abc.KeysView.register(archive_keys)

class archive_values(collections.abc.ValuesView):
    def __init__(self, mapping):
        if not isinstance(mapping, Archive):
            raise TypeError("{0.__class__.__name__} can only be used with mappables of type Archive, not '{:s}'".format(self, type(mapping).__name__))
        self.mapping = mapping
        self._length = None

    @staticmethod
    def _get_mapping_gen(_buffer):
        _buffer.seek(0, os.SEEK_SET)
        while True:
            try: info = pickle.load(_buffer)
            except EOFError: return
            try: yield deserialize(pickle.load(_buffer))
            except EOFError as e: raise CorruptArchiveError from e
    
    @property
    def _mapping(self):
        if not hasattr(self.mapping, '_buffer'): return []
        return self._get_mapping_gen(self.mapping._buffer)
    def __iter__(self): yield from self._mapping
    def __len__(self):
        if self._length is None: self._length = sum([1 for _ in self._mapping])
        return self._length
    def __contains__(self, value):
        if not is_pickled(value):
            return self.mapping._buffer.find(pickle.dumps(value)) != -1
        else: return self.mapping._buffer.find(value) != -1
collections.abc.ValuesView.register(archive_values)


class archive_items(collections.abc.ItemsView):
    def __init__(self, mapping):
        if not isinstance(mapping, Archive):
            raise TypeError("{0.__class__.__name__} can only be used with mappables of type Archive, not '{:s}'".format(self, type(mapping).__name__))
        self.mapping = mapping
        self._length = None
    
    @staticmethod
    def _get_mapping_gen(_buffer):
        _buffer.seek(0, os.SEEK_SET)
        while True:
            try: info = pickle.load(_buffer)
            except EOFError: return
            try: value = deserialize(pickle.load(_buffer))
            except EOFError as e: raise CorruptArchiveError from e
            yield info.key, value
    
    @property
    def _mapping(self):
        if not hasattr(self.mapping, '_buffer'): return []
        return self._get_mapping_gen(self.mapping._buffer)
    def __iter__(self): yield from self._mapping
    def __len__(self):
        if self._length is None: self._length = sum([1 for _ in self._mapping])
        return self._length
collections.abc.ItemsView.register(archive_items)

"""
class Buffer(object):
    def __init__(
            self,
            path : str | pathlib.Path | type(None) = None,
            readonly : bool = False,
    ):
        self._closed = False
        if path is None:
            with tempfile.NamedTemporaryFile(dir = os.getcwd(), suffix = '.tmp') as f:
                path = f.name
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
        if stop - curpos <= 0: right_content = b''
        else: right_content = self.read(stop - curpos)
        first_string = str(left_content + b' ' + right_content)[2:-1]
        bottom_string = '^'*(len(str(left_content)[2:-1])) + '*' + ' '*len(str(right_content)[2:-1]) + '\n' + ' '*len(str(left_content)[2:-1]) + 'pos'

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
        with io.BytesIO(value) as other_buffer:
            if isinstance(index, slice):
                start, stop, step = index.indices(self.size())
                if step > 1:
                    for i in range(start, stop, step):
                        self.seek(i, os.SEEK_SET)
                        self.write(other_buffer.read(1))
                else:
                    self.seek(start, os.SEEK_SET)
                    self.write(other_buffer.read())
            else:
                self.seek(index, os.SEEK_SET)
                self.write(value.read())

    def seek(self, *args, **kwargs): return self._f.seek(*args, **kwargs)
    def tell(self, *args, **kwargs): return self._f.tell(*args, **kwargs)
    def read(self, *args, **kwargs): return self._f.read(*args, **kwargs)
    def readline(self, *args, **kwargs): return self._f.readline(*args, **kwargs)
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
        
        if self._f.name.endswith('.tmp'):
            starsmashertools.helpers.path.remove(self._f.name)
    
    def size(self):
        try: return starsmashertools.helpers.path.getsize(self._f.name)
        except OSError: return 0
    
    def resize(self, newsize : int):
        with self.lock:
            self._f.truncate(newsize)
            self.seek(min(self.tell(), newsize), os.SEEK_SET)
"""

class NumPyArray(object):
    def __init__(self, value):
        self.value = value

    def __getstate__(self):
        compressed = io.BytesIO()
        np.savez_compressed(compressed, self.value)
        compressed.seek(0)
        return compressed.read()
    def __setstate__(self, state):
        self.value = np.load(io.BytesIO(state))['arr_0']

class Info(object):
    """ Contains information about an Archive entry for processing reading. """
    def __init__(
            self,
            size : int,
            key = None,
            mtime : int | type(None) = None,
    ):
        if mtime is None: mtime = time.time()
        self.size = size
        self.key = key
        self.mtime = mtime
    
    def __reduce__(self):
        return (self.__class__, (self.size, self.key, self.mtime))

class List(list): pass
        
            

class Archive(object):
    """
    Make sure that when you are using this class with multiple processes or
    threads that you implement a lock whenever you get/set values.
    """
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def __init__(
            self,
            path : str | pathlib.Path | type(None) = None,
            auto_save : bool = True,
            readonly : bool = True,
            madvise = [
                mmap.MAP_SHARED,
            ],
    ):
        self.auto_save = auto_save
        self.readonly = readonly
        self.madvise = madvise
        self.set_path(path)

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
        value = serialize(value)

        if isinstance(key, Info): info = key
        else: info = Info(len(value), key = key)
        
        if not hasattr(self, '_buffer'):
            # Add a new key
            self._create_buffer(serialize(info) + value)
            if self.auto_save: self.save()
            return
        
        towrite = serialize(info) + value

        # Handle resizing
        with self._lock:
            for i in self.infos():
                if i.key != info.key: continue
                start = self._buffer.tell() - len(pickle.dumps(i))
                stop = start + len(towrite)
                prev_len = i.size + len(pickle.dumps(i))
                if len(towrite) > prev_len:
                    prevsize = self._buffer.size()
                    self._buffer.resize(self._buffer.size() - prev_len + len(towrite))
                    for madv in self.madvise: self._buffer.madvise(madv)
                    src = start + prev_len
                    try:
                        self._buffer.move(stop, src, prevsize - src)
                    except ValueError:
                        print(stop, src, prevsize)
                        raise
                elif len(towrite) < prev_len:
                    src = start + prev_len
                    try:
                        self._buffer.move(stop, src, self._buffer.size() - src)
                    except ValueError:
                        print(stop, src, self._buffer.size())
                        raise
                    self._buffer.resize(self._buffer.size() - prev_len + len(towrite))
                    for madv in self.madvise: self._buffer.madvise(madv)
                break
            else:
                start = self._buffer.size()
                self._buffer.resize(self._buffer.size() + len(towrite))
                for madv in self.madvise: self._buffer.madvise(madv)
                stop = self._buffer.size()

            self._buffer[start:stop] = towrite

            if self.auto_save: self.save()
    
    def __getitem__(self, key):
        if hasattr(self, '_lock'):
            with self._lock:
                for info in self.infos():
                    if info.key != key: continue
                    return deserialize(pickle.load(self._buffer))
        else:
            for info in self.infos():
                if info.key != key: continue
                return deserialize(pickle.load(self._buffer))
        raise KeyError(key)

    def __delitem__(self, key):
        """ Remove the given key from the Archive. If auto save is enabled, the
        Archive file will be modified. If there are any values appended to this
        item, they are also removed. """
        with self._lock:
            for info in self.infos():
                if info.key != key: continue
                leninfo = len(pickle.dumps(info))
                self._buffer.seek(-leninfo, os.SEEK_CUR)
                start = self._buffer.tell()
                src = start + leninfo + info.size
                newsize = self._buffer.size() - (leninfo + info.size)
                self._buffer.move(start, src, self._buffer.size() - src)
                if newsize == 0: self.clear()
                else:
                    self._buffer.resize(newsize)
                    for madv in self.madvise: self._buffer.madvise(madv)
                break
            else:
                raise KeyError(key)

            if self.auto_save: self.save()
        
    
    def __contains__(self, key):
        if hasattr(self, '_lock'):
            with self._lock: return key in self.keys()
        return key in self.keys()
    def __len__(self):
        if hasattr(self, '_lock'):
            with self._lock: return len(self.keys())
        return len(self.keys())

    def _create_buffer(self, content : bytes):
        if self.readonly: raise ReadOnlyError
        with starsmashertools.helpers.file.open(
                self.path, 'wb+',
        ) as f:
            f.write(content)
            f.flush()
            self._buffer = mmap.mmap(
                f.fileno(),
                0,
                access = mmap.ACCESS_READ if self.readonly else mmap.ACCESS_WRITE,
            )
            for madv in self.madvise: self._buffer.madvise(madv)
            if hasattr(self, '_lock'):
                if self._lock.locked: self._lock.unlock()
            self._lock = f.sstools_lock
    
    def set_path(
            self,
            path : str | pathlib.Path | type(None),
    ):
        if hasattr(self, '_buffer'):
            if not self.readonly: self.save()
            self._buffer.close()
            del self._buffer
        if path is not None and starsmashertools.helpers.path.exists(path):
            with starsmashertools.helpers.file.open(
                    path, 'rb' if self.readonly else 'rb+'
            ) as f:
                self._buffer = mmap.mmap(
                    f.fileno(),
                    0,
                    access = mmap.ACCESS_READ if self.readonly else mmap.ACCESS_WRITE,
                )
                for madv in self.madvise: self._buffer.madvise(madv)
                if hasattr(self, '_lock'):
                    if self._lock.locked: self._lock.unlock()
                self._lock = f.sstools_lock
        self._path = path
    
    def size(self):
        if hasattr(self, '_lock'):
            with self._lock:
                if hasattr(self, '_buffer'): return self._buffer.size()
                return 0
        else:
            if hasattr(self, '_buffer'): return self._buffer.size()
        return 0
    
    def keys(self): return archive_keys(self)
    def values(self): return archive_values(self)
    def items(self): return archive_items(self)
    def infos(self): return archive_infos(self)

    def clear(self):
        del self._buffer
        if self.auto_save: self.save()

    def save(self, path : str | pathlib.Path | type(None) = None):
        if not hasattr(self, '_buffer'):
            if self.readonly: raise ReadOnlyError
            if self.path is not None:
                if starsmashertools.helpers.path.exists(self.path):
                    starsmashertools.helpers.path.remove(self.path)
        else:
            if path is None:
                if self.readonly: raise ReadOnlyError
                with self._lock:
                    self._buffer.flush()
            else: # This means copy the buffer
                with starsmashertools.helpers.file.open(path, 'wb') as f:
                    f.write(self._buffer.read())

    def append(self, key : str, *values):
        """
        Append a value to the end of an existing key. Calls to 
        :meth:`~.__getitem__` will return a list of all appended items. The
        ``key`` being appended to will have its mtime replaced with the current
        timestamp.
        """
        with self._lock:
            for info in self.infos():
                if info.key != key: continue
                value = pickle.load(self._buffer)
                if not isinstance(value, List): value = List([value] + list(values))
                else: value += list(values)
                self[key] = value
                return
            raise KeyError(key)

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

        mtime = None
        if replace == 'mtime' and origin is not None:
            mtime = starsmashertools.helpers.path.getmtime(origin)

        towrite = serialize(value)
        with self._lock:
            for info in self.infos():
                if info.key != key: continue
                if replace == 'mtime' and mtime is not None and info.mtime > mtime: return

        self[Info(len(towrite), key=key, mtime=mtime)] = towrite
