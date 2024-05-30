import os
import struct
import pickle
import collections
import mmap
import gzip
import contextlib
import shutil
import starsmashertools.helpers.path
import starsmashertools.helpers.file

class InvalidArchiveError(Exception): pass
class InvalidIdentifierError(Exception): pass
class CompressedError(Exception): pass

encoding = 'ascii'
endian = '<'

class Meta:
    """ Meta information about a :py:class:`bytes` data set. """
    _format = endian + 'ii{len:d}s'
    def __init__(
            self,
            identifier : str | bytes,
            start : int,
            stop : int,
    ):
        if start >= stop:
            raise ValueError('start >= stop (%d >= %d)' % (start, stop))
        if isinstance(identifier, bytes):
            identifier = identifier.decode(encoding)
        self.identifier = identifier
        self.start = start
        self.stop = stop
    def __str__(self):
        return "Meta('{identifier:s}',{start:d},{stop:d})".format(
            identifier = self.identifier,
            start = self.start,
            stop = self.stop,
        )
    def __repr__(self): return str(self)
    def to_bytes(self):
        if isinstance(self.identifier, str):
            identifier = self.identifier.encode(encoding)
        else: identifier = self.identifier
        fmt = Meta._format.format(len = len(identifier))
        return struct.pack(
            fmt, # must come first
            self.start,
            self.stop,
            identifier,
        )
    @staticmethod
    def from_bytes(b : bytes):
        try:
            identifier = struct.unpack(endian + '%ds' % (len(b) - 8), b[8:])[0]
        except Exception as e:
            raise InvalidIdentifierError from e
        
        return Meta(
            identifier,
            *struct.unpack(endian + 'ii', b[:8]), # start and stop
        )


class Archive:
    """ The file is structured as::
    
        ... bytes ... (byte data, no delimiters)
        ... bytes ... (Meta objects, delimited by ASCII character US)
        length of the footer (4-byte representation <i)
        EOF (not an actual character, just the end of the file)
    
    """

    footer_delimiter = chr(31).encode(encoding) # ASCII character US
    
    def __init__(
            self,
            path : str,
    ):
        if starsmashertools.helpers.path.exists(path):
            try:
                self._footer = self.get_footer(path)
            except Exception as e:
                raise InvalidArchiveError(str(path)) from e
        else:
            self._footer = collections.OrderedDict({})
        self.path = path

    def __contains__(self, identifier : str): return identifier in self._footer
    
    @property
    def _mode(self): return 'rb+' if starsmashertools.helpers.path.exists(self.path) else 'wb+'

    def _get(self, identifier : str, _buffer):
        if identifier not in self._footer:
            raise InvalidIdentifierError("No identifier in footer: '%s'" % str(identifier))
        _buffer.seek(self._footer[identifier].start)
        return _buffer.read(self._footer[identifier].stop - self._footer[identifier].start)
    
    def _set(
            self,
            identifier : str,
            data,
            _buffer,
    ):
        new = pickle.dumps(data)
        if identifier in self._footer:
            diff = len(new) - len(self.get_raw(identifier))
            # We are SOL, here. There's no possible way to insert/delete file
            # contents in the middle. We have to use mmap as our crutch.
            orig_size = _buffer.size()
            new_size = orig_size + diff

            # Resize the file to accommodate the new data
            if diff > 0: # new data is larger than old data
                _buffer.resize(new_size)
                _buffer.move(
                    self._footer[identifier].stop + diff, # dst (offset)
                    self._footer[identifier].stop, # src (offset)
                    orig_size - self._footer[identifier].stop, # count
                )
            else:
                # Shift the contents "up", overwriting content
                _buffer.move(
                    self._footer[identifier].stop + diff,
                    self._footer[identifier].stop,
                    orig_size - self._footer[identifier].stop,
                )
                _buffer.resize(_buffer.size() + diff)

            # Now replace the old content with the new content
            _buffer.seek(self._footer[identifier].start)
            _buffer.write(new)
            self._footer[identifier].stop = _buffer.tell()

            _buffer.flush(
                self._footer[identifier].start,
                new_size - self._footer[identifier].start,
            )

            past = False
            for key, val in self._footer.items():
                if key == identifier:
                    past = True
                    continue # Don't edit for the rewritten identifier
                if not past: continue
                self._footer[key].start += diff
                self._footer[key].stop += diff
        
        else: # Append the data to the file
            if self._footer:
                _buffer.seek(list(self._footer.values())[-1].stop)
            start = _buffer.tell()
            _buffer.write(new)
            self._footer[identifier] = Meta(identifier, start, _buffer.tell())

    def _write_footer(self, f = None):
        """ This method doesn't care about the current contents of the file. It
        will append the footer to the end of the file regardless of the contents.
        If there is already a footer, it should be removed first. """
        footer = Archive.footer_delimiter.join([meta.to_bytes() for meta in self._footer.values()])
        wasNone = f is None
        if wasNone: f = self._open(self._mode)
        f.seek(list(self._footer.values())[-1].stop)
        f.write(footer)
        f.write(struct.pack(endian + 'i', len(footer)))
        f.truncate()
        if wasNone: f.close()

    def _open(self, mode):
        if self.compressed:
            if mode not in ('r', 'rb'):
                raise CompressedError("Must call decompress() before modifying an archive.")
            return gzip.open(self.path, mode = mode)
        return starsmashertools.helpers.file.open(self.path, mode = mode)

    @property
    def compressed(self):
        if not starsmashertools.helpers.path.exists(self.path): return False
        # thank you, https://stackoverflow.com/a/47080739/4954083
        with open(self.path, 'rb') as f:
            compressed = f.read(2) == b'\x1f\x8b'
        return compressed

    def compress(self, replace : bool = True):
        if self.compressed:
            raise CompressedError("Archive is already compressed")
        with open(self.path, 'rb') as f_in:
            with gzip.open(self.path + '.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        if replace: os.replace(self.path + '.gz', self.path)

    def decompress(self, replace : bool = True):
        if not self.compressed:
            raise CompressedError("Archive is not compressed")
        with gzip.open(self.path, 'rb') as f_in:
            with open(self.path + '.tmp', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.replace(self.path + '.tmp', self.path)

    @staticmethod
    def get_footer(path):
        with self._open('rb') as f:
            f.seek(-4, 2)
            length = f.read(4)
            if not length: # This was an empty file
                return collections.OrderedDict({})
            length = struct.unpack(endian + 'i', length)[0]
            f.seek(-(length + 4), 2)
            footer = f.read(length)
        return collections.OrderedDict({
            m.identifier : m for m in [Meta.from_bytes(b) for b in footer.split(Archive.footer_delimiter)]
        })

    def get_raw(
            self,
            identifier : str,
    ):
        """ Return the data (bytes) associated with the given :class:`~.Meta` 
        ``identifier``\. """
        with self._open('rb') as f:
            return self._get(identifier, f)
    
    def get(self, *args, **kwargs):
        return pickle.loads(self.get_raw(*args, **kwargs))

    def get_many(self, identifiers):
        with self._open('rb') as f:
            for identifier in identifiers:
                yield self._get(identifier, f)
    
    def set(
            self,
            identifier : str,
            data,
    ):
        """ Given some ``identifier``\, add ``data`` to the Archive. If 
        ``identifier`` is already in the Archive, its contents are replaced.
        Otherwise, the data is appended to the Archive. This is appropriate only 
        for setting single values in the file. If setting multiple values, use 
        :meth:`~.set_many` instead. """
        with self._open(self._mode) as f:
            if identifier in self._footer:
                with mmap.mmap(
                        f.fileno(),
                        0, # map the entire file
                        # This affects memory, but doesn't update the file (?)
                        access = mmap.ACCESS_WRITE,
                ) as _buffer:
                    # Indicate to the file system that the buffer is going to be
                    # edited, and that those edits should be reflected in any
                    # other process which has mapped this file.
                    _buffer.madvise(mmap.MAP_SHARED)
                    self._set(identifier, data, _buffer)
                    _buffer.close()
            else:
                self._set(identifier, data, f)
            self._write_footer(f = f)

    def set_many(self, identifiers, data):
        # Doing it in this way ensures that generators aren't consumed improperly
        with self._open(self._mode) as f:
            with mmap.mmap(
                    f.fileno(),
                    0, # map the entire file
                    # This affects memory, but doesn't update the file (?)
                    access = mmap.ACCESS_WRITE,
            ) as _buffer:
                # Indicate to the file system that the buffer is going to be
                # edited, and that those edits should be reflected in any
                # other process which has mapped this file.
                _buffer.madvise(mmap.MAP_SHARED)
                
                for identifier, d in zip(identifiers, data):
                    if identifier in self._footer:
                        self._set(identifier, d, _buffer)
                    else:
                        self._set(identifier, d, f)
                
                _buffer.close()
            self._write_footer(f = f)
    
