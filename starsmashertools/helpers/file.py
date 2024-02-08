import starsmashertools.helpers.argumentenforcer
import contextlib
import builtins
import numpy as np
import typing

fortran_comment_characters = ['c','C','!','*']

downloaded_files = []

class FileModeError(Exception, object): pass

# This works even with using multiprocessing.
class Lock(object):
    def __init__(self, path):
        import starsmashertools.helpers.path
        import os
        
        self.path = starsmashertools.helpers.path.realpath(path) + '.lock'

        pid_str = str(os.getpid())

        while True:
            try:
                with builtins.open(self.path, 'x') as f:
                    f.write(pid_str)
            except FileExistsError as e: continue
            break
        self.locked = True
    
    def unlock(self):
        import starsmashertools.helpers.path
        starsmashertools.helpers.path.remove(self.path)
        self.locked = False

@starsmashertools.helpers.argumentenforcer.enforcetypes
@contextlib.contextmanager
def open(
        path : str,
        mode : str,
        lock : bool = True,
        lock_file : str | type(None) = None,
        method : typing.Callable | type(None) = None,
        **kwargs
):
    import starsmashertools.helpers.path
    import zipfile
    import os

    if method is None: method = builtins.open
    
    if (not starsmashertools.helpers.path.isfile(path) and
        mode not in ['w', 'a', 'w+', 'r+', 'wb', 'wb+', 'rb+', 'ab']):
        #print("error raised")
        raise FileNotFoundError(path)
    
    if lock:
        #print("New lock")
        _lock = Lock(path)
    
    f = method(path, mode, **kwargs)
    try:
        yield f
    finally:
        f.close()
        if lock:
            _lock.unlock()

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
