import starsmashertools.preferences
import starsmashertools.helpers.argumentenforcer
import starsmashertools.helpers.string
import starsmashertools.helpers.path
import starsmashertools.helpers.asynchronous
import contextlib
import builtins
import numpy as np
import typing
import atexit
import zipfile
import os

fortran_comment_characters = ['c','C','!','*']

downloaded_files = []

modes = {
    'readonly' : ['r', 'rb'],
    'write' : ['w', 'a', 'w+', 'r+', 'wb', 'wb+', 'rb+', 'ab', 'x'],
}
all_modes = []
for v in modes.values(): all_modes += v

class FileModeError(Exception, object): pass

@starsmashertools.preferences.use
class Lock(object):
    """
    If a file is being read, writing is not allowed.
    If a file is being written, reading/writing is not allowed.
    """
    
    def __init__(
            self,
            path : str,
            mode : str,
            timeout : int | float | type(None) = None,
    ):
        import os
        import starsmashertools
        import starsmashertools.helpers.path
        import copy
        
        if timeout is None:
            timeout = self.preferences.get('timeout')
        
        self.path = starsmashertools.helpers.path.realpath(path)
        self.mode = mode
        self.timeout = timeout

        self.identifier = starsmashertools.helpers.path.realpath(path).replace(
            os.sep,
            '_',
        )

        basename = self.path.replace(os.sep, '_')
        basename_with_mode = copy.deepcopy(basename)
        for key, val in modes.items():
            if self.mode not in val: continue
            basename_with_mode += '.' + key
            break
        else:
            raise NotImplementedError("Unknown file mode '%s'" % self.mode)
        
        directory = starsmashertools.LOCK_DIRECTORY
        
        full_name = starsmashertools.helpers.path.join(
            directory, basename_with_mode,
        )
        
        # Find other files that start with the same name and mode identifier. It
        # doesn't need to match the mode we are in, it just needs to exist. This
        # keeps our order of operations consistent with what user's code wants
        # to do. It works like a "stack".
        while True: # Keep trying until we make a correct file
            existing = []
            for filename in starsmashertools.helpers.path.listdir(directory):
                if not filename.startswith(basename): continue
                existing += [filename]

            self.lockfile = full_name + '.%d' % len(existing)
            
            try:
                # Touch to create the lock file, regardless if we are reading or
                # writing
                builtins.open(self.lockfile, 'x').close()
            except FileExistsError:
                continue
            
            break

        
        # Always unlock before program exit
        atexit.register(self.unlock)

    @staticmethod
    def get_lockfile_mode(lockfile : str):
        return lockfile.split('.')[-2]
    
    @staticmethod
    def get_lockfile_number(lockfile : str):
        return int(lockfile.split('.')[-1])

    @staticmethod
    def get_lockfile_basename(lockfile : str):
        return '.'.join(lockfile.split('.')[:-2])

    def get_all_locks(self):
        # Obtain all the lock files including us.
        import starsmashertools.helpers.path
        directory = starsmashertools.helpers.path.dirname(self.lockfile)
        basename = starsmashertools.helpers.path.basename(self.lockfile)
        base_lock_name = Lock.get_lockfile_basename(basename)
        ret = []
        numbers = []
        for filename in starsmashertools.helpers.path.listdir(directory):
            if Lock.get_lockfile_basename(filename) != base_lock_name: continue
            path = starsmashertools.helpers.path.join(directory, filename)
            ret += [path]
            numbers += [Lock.get_lockfile_number(path)]
        # Sort by the modification times
        ret = [x for _, x in sorted(zip(numbers, ret), key=lambda pair:pair[0])]
        return ret

    def get_locks(self):
        ret = {}
        for path in self.get_all_locks():
            mode = Lock.get_lockfile_mode(path)
            if mode not in ret: ret[mode] = []
            ret[mode] += [path]
        # This is already sorted by the numbers from get_all_locks
        return ret

    def get_active_lockfile(self):
        # The 'active' lockfile is the one which is in write mode and has the
        # lowest number, or which is in read 
        paths = self.get_locks()['write']
        if paths: return paths[0]
        raise FileNotFoundError("Failed to find the active lockfile. This should never be possible.")
    
    def ready(self):
        # If we're next on the stack to hold the lock on this file
        locks = self.get_all_locks() # Includes us
        if locks[0] == self.lockfile: return True
        
        # If we want to write, then we aren't ready until we're at the bottom of
        # the stack.
        if self.mode in modes['write']: return False
        
        # Check the locks below us on the stack.
        
        # If all we want to do is read, then we're not ready if the file is
        # going to be written on the stack.
        if self.mode in modes['readonly']:
            for lock in locks:
                if Lock.get_lockfile_mode(lock) == 'write': return False
            return True

        raise NotImplementedError("Unknown file mode '%s'" % mode)    

    def unlock(self):
        import starsmashertools.helpers.path
        try:
            starsmashertools.helpers.path.remove(self.lockfile)
        except FileNotFoundError: pass
    
    def wait(self):
        import time
        
        t0 = time.time()
        while not self.ready():
            #print("Still waiting", time.time() - t0)
            if time.time() - t0 >= self.timeout:
                self.unlock()
                raise TimeoutError("Timeout while waiting for lock on file '%s'" % self.path)
            time.sleep(1.e-6)


@starsmashertools.helpers.argumentenforcer.enforcetypes
@contextlib.contextmanager
def open(
        path : str,
        mode : str,
        lock : bool = True,
        method : typing.Callable | type(None) = None,
        timeout : int | float | np.generic | type(None) = None,
        verbose : bool = True,
        message : str | type(None) = None,
        progress_max : int = 0,
        **kwargs
):
    # File IO can take a long time, so we use some helper functions to tell the
    # user what is going on

    if mode not in all_modes:
        raise NotImplementedError("Unsupported file mode '%s'" % mode)    

    if verbose:
        verbose = starsmashertools.helpers.asynchronous.is_main_process()
        short_path = starsmashertools.helpers.string.shorten(
            path, 40, where = 'left',
        )
    
    if method is None: method = builtins.open

    # Do we really need this??
    #writable = mode in modes['write']
    #if not writable and not starsmashertools.helpers.path.isfile(path):
    #    raise FileNotFoundError(path)
    
    # Always lock no matter what, unless we have been told explicitly that we
    # don't need to
    _lock = None
    if lock:
        _lock = Lock(path, mode, timeout = timeout)
        if _lock.timeout > 0:
            if not verbose: _lock.wait()
            else:
                with starsmashertools.helpers.string.loading_message(
                        "Waiting for '%s'" % short_path, delay = 5,
                ) as loading_message:
                    _lock.wait()
    
    f = None
                
    try:
        f = method(path, mode, **kwargs)
        f.sstools_lock = _lock
        if not verbose: yield f
        else:
            if message is None:
                if mode in modes['write']: message = "Writing '%s'" % short_path
                else: message = "Reading '%s'" % short_path
                
            if progress_max > 0:
                with starsmashertools.helpers.string.progress_message(
                        message = message, delay = 5, max = progress_max,
                ) as progress:
                    f.progress = progress
                    yield f
            else:
                with starsmashertools.helpers.string.loading_message(
                        message = message, delay = 5,
                ) as loading_message:
                    yield f
    except:
        if f is not None: f.close()
        if _lock is not None: _lock.unlock()
        _lock = None
        raise
    finally:
        if f is not None: f.close()
        if _lock is not None: _lock.unlock()
        _lock = None
        f = None



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
    """
    Return instances of the given phrase in the given file as a list of strings
    where each element is the contents of the file after the phrase appears, up
    to the specified `end`, or the next instance of the phrase.
    
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
    """
    Sort the given files by their modification times.

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
