import starsmashertools.helpers.argumentenforcer
import contextlib
import builtins
import numpy as np
import typing
import atexit

fortran_comment_characters = ['c','C','!','*']

downloaded_files = []

class FileModeError(Exception, object): pass

# This works even with using multiprocessing.
class Lock(object):
    instances = []
    def __init__(self, path, pid = None):
        import os
        import starsmashertools.helpers.path
        
        self.timeout = None
        if path.endswith('.lock'):
            raise Exception("Cannot lock a lock file: '%s'" % path)
        self.path = starsmashertools.helpers.path.realpath(path)
        self.lockfile = self.path + '.lock'
        
        if pid is None: pid = os.getpid()
        self.pid = pid
        Lock.instances += [self]

    @property
    def locked(self):
        import starsmashertools.helpers.path
        return starsmashertools.helpers.path.isfile(self.lockfile)

    @property
    def isLocker(self):
        # Returns True if this Lock's process ID (pid) is the same as
        # what is written in the lock file
        if not self.locked: return False
        return self.pid == self.get_current_pid()

    @staticmethod
    def unlock_all(*args, **kwargs):
        for instance in Lock.instances:
            instance.unlock()

    def get_current_pid(self):
        import starsmashertools.helpers.path
        
        if not starsmashertools.helpers.path.isfile(self.lockfile):
            return None
        with builtins.open(self.lockfile, 'r') as f:
            content = f.read()
        return int(content.strip())
    
    def lock(self, timeout = None):
        import time
        import copy
        import starsmashertools.preferences

        if timeout is None:
            # Check the preferences
            pref = starsmashertools.preferences.get_default(
                'IO', 'Lock', throw_error = False,
            )
            if pref is None: timeout = float('inf')
            else: timeout = pref.get('timeout', float('inf'))
        
        self.timeout = timeout
        
        # Wait for the given file to become available
        timer = 0.
        t0 = time.time()
        while timer < self.timeout:
            if not self.locked:
                try:
                    with builtins.open(self.lockfile, 'x') as f:
                        f.write(str(self.pid))
                    break
                except FileExistsError: pass

            t1 = time.time()
            timer += t1 - t0
            t0 = copy.deepcopy(t1)
    
    def unlock(self):
        import starsmashertools.helpers.path
        if not self.locked: return
        if starsmashertools.helpers.path.isfile(self.lockfile):
            if self.isLocker:
                starsmashertools.helpers.path.remove(self.lockfile)
        self.timeout = None
    
    @staticmethod
    def get(path):
        import starsmashertools.helpers.file
        import starsmashertools.helpers.path
        import os
        
        ret = Lock(path)
        pid = ret.get_current_pid()
        if pid is None: pid = os.getpid()
        ret.pid = pid
        return ret
            

# Catch all interrupts and unlock all the files before exiting
atexit.register(Lock.unlock_all)
        

@starsmashertools.helpers.argumentenforcer.enforcetypes
@contextlib.contextmanager
def open(
        path : str,
        mode : str,
        lock : bool = True,
        method : typing.Callable | type(None) = None,
        timeout : int | float | np.generic | type(None) = None,
        **kwargs
):
    import starsmashertools.helpers.path
    import zipfile
    import os

    if method is None: method = builtins.open
    
    if (not starsmashertools.helpers.path.isfile(path) and
        mode not in ['w', 'a', 'w+', 'r+', 'wb', 'wb+', 'rb+', 'ab']):
        raise FileNotFoundError(path)
    
    if lock:
        _lock = Lock(path)
        _lock.lock(timeout = timeout)
    
    f = method(path, mode, **kwargs)
    try:
        yield f
    finally:
        f.close()
        if lock: _lock.unlock()

# Check if the file at the given path is a MESA file
def isMESA(path):
    import starsmashertools.helpers.path
    import starsmashertools.helpers.string
    
    if not starsmashertools.helpers.path.isfile(path): return False
    
    with open(path, 'r') as f:

        # The first line is always a bunch of sequential numbers
        line = f.readline()
        linesplit = line.split()
        ncols = len(linesplit)

        for i, item in enumerate(linesplit):
            parsed = starsmashertools.helpers.string.parse(item)
            if not isinstance(parsed, int): return False
            if i + 1 != parsed: return False

        # The next line is always a header with only strings. It should
        # have the same length as the previous line
        line = f.readline()
        linesplit = line.split()
        if len(linesplit) != ncols: return False
        for item in linesplit:
            parsed = starsmashertools.helpers.string.parse(item)
            if not isinstance(parsed, str): return False

        # The next line contains a bunch of data, but it's not all necessarily
        # numerical data
        line = f.readline()
        linesplit = line.split()
        if len(linesplit) != ncols: return False

        # A newline is next
        line = f.readline().strip()
        if len(line) != 0: return False

        # More numbers now
        line = f.readline()
        linesplit = line.split()
        ncols = len(linesplit)
        for i, item in enumerate(linesplit):
            parsed = starsmashertools.helpers.string.parse(item)
            if not isinstance(parsed, int): return False
            if i + 1 != parsed: return False

        # Another header
        line = f.readline()
        linesplit = line.split()
        if len(linesplit) != ncols: return False
        for item in linesplit:
            parsed = starsmashertools.helpers.string.parse(item)
            if not isinstance(parsed, str): return False

        # The rest of the file should be data with the same number of columns
        # as the previous header
        for line in f:
            linesplit = line.split()
            if len(linesplit) != ncols: return False
    
    return True


# Check if 2 files are identical
#@profile
def compare(file1, file2):
    import starsmashertools.helpers.path
    import filecmp
    #file1 = starsmashertools.helpers.path.realpath(file1)
    #file2 = starsmashertools.helpers.path.realpath(file2)
    
    # If they have the same path, then they are the same file
    if file1 == file2: return True
    
    size1 = starsmashertools.helpers.path.getsize(file1)
    size2 = starsmashertools.helpers.path.getsize(file2)

    if size1 == size2:
        return filecmp.cmp(file1, file2, shallow=False)
    return False



@starsmashertools.helpers.argumentenforcer.enforcetypes
def get_phrase(
        path : str,
        phrase : str,
        end : str = '\n',
):
    """Return instances of the given phrase in the given file as a list of strings where each element is the contents of the file after the phrase appears, up to the specified `end`, or the next instance of the phrase.
    
    Parameters
    ----------
    path : str
        The path to the file to read.
    phrase : str
        The phrase to search for in the file specified by `path`. Whatever is
        between `phrase` and `end` will be returned.
    end : str, default = '\n'
        The ending to search for after finding `phrase`.

    Returns
    -------
    str

    """
    import mmap
    
    with open(path, 'rb') as f:
        buffer = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    bphrase = phrase.encode('utf-8')
    index = buffer.find(bphrase)
    if index == -1: return None
    ret = []
    startidx = index
    bend = end.encode('utf-8')
    while startidx != -1:
        # Go to the position just after the previous instance of the phrase
        startidx += len(bphrase)
        
        buffer.seek(startidx)
        
        # Search for the next end character
        next_end = buffer.find(bend)

        # Read the contents between the previous instance of the phrase and the
        # closest end character
        content = buffer.read(next_end - startidx)

        # Search for the phrase to see if it appears anywhere in the content
        ret += content.split(bphrase)

        # Search for the next instance of the phrase
        startidx = buffer.find(bphrase)

    ret = [r.decode('utf-8') for r in ret]
    return ret
    
@starsmashertools.helpers.argumentenforcer.enforcetypes
def sort_by_mtimes(
        files : list
):
    """Sort the given files by their modification times.

    Parameters
    ----------
    files : list
        The list or tuple of files to sort.

    Returns
    -------
    list
    """
    import copy
    import starsmashertools.helpers.path
    
    ret = copy.deepcopy(files)
    ret.sort(key=lambda f: starsmashertools.helpers.path.getmtime(f))
    return ret[::-1]




def reverse_readline(path, **kwargs):
    """A generator that returns the lines of a file in reverse order
    https://stackoverflow.com/a/23646049/4954083"""
    import mmap
    with open(path, 'rb') as fh:
        buffer = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    return reverse_readline_buffer(buffer, **kwargs)

def reverse_readline_buffer(_buffer, buf_size=8192):
    import os
    _buffer.seek(0, os.SEEK_END)
    segment = None
    offset = 0
    file_size = remaining_size = _buffer.tell()
    while remaining_size > 0:
        offset = min(file_size, offset + buf_size)
        _buffer.seek(file_size - offset)
        buffer = _buffer.read(min(remaining_size, buf_size)).decode(encoding='utf-8')
        remaining_size -= buf_size
        lines = buffer.split('\n')
        # The first line of the buffer is probably not a complete line so
        # we'll save it and append it to the last line of the next buffer
        # we read
        if segment is not None:
            # If the previous chunk starts right from the beginning of line
            # do not concat the segment to the last line of new chunk.
            # Instead, yield the segment first 
            if buffer[-1] != '\n':
                lines[-1] += segment
            else:
                yield segment
        segment = lines[0]
        for index in range(len(lines) - 1, 0, -1):
            if lines[index]:
                yield lines[index]
    # Don't yield None if the file was empty
    if segment is not None:
        yield segment
