import starsmashertools.helpers.path
import starsmashertools.helpers.ssh
import starsmashertools.helpers.string
import starsmashertools.helpers.argumentenforcer
import tempfile
import builtins
import os
import filecmp
import numpy as np
import contextlib
import copy
import io

fortran_comment_characters = ['c','C','!','*']

downloaded_files = []


def get_file(path, mode):
    path = starsmashertools.helpers.path.realpath(path)
    if starsmashertools.helpers.ssh.isRemote(path):
        # Shucks. We'll have to copy the contents to the local machine
        address, filepath = starsmashertools.helpers.ssh.split_address(path)
        basename = starsmashertools.helpers.path.basename(path)
        contents = starsmashertools.helpers.ssh.run(
            address,
            "cat %s" % filepath,
            text = not (basename[:3] == "out" and basename[-4:] == ".sph"),
        )
        _mode = 'w+'
        if 'b' in mode: _mode += 'b'
        tfile = tempfile.NamedTemporaryFile(mode=_mode, delete=False)
        tfile.write(contents)
        tfile.close()

        print("Downloaded '%s' as '%s'" % (path, tfile.name))
        downloaded_files += [tfile.name]
        return tfile.name
    return path


@contextlib.contextmanager
def open(path, mode, **kwargs):
    path = get_file(path, mode)
    f = builtins.open(path, mode, **kwargs)
    try: yield f
    finally: f.close()



# Check if the file at the given path is a MESA file
def isMESA(path):
    if not starsmashertools.helpers.path.isfile(path): return False
    
    with starsmashertools.helpers.file.open(path, 'r') as f:

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

"""
def read_MESA(path):
    f = starsmashertools.helpers.file.open(path, 'r')
    
    f.readline() # numbers
    header = f.readline().strip().split()
    obj = {key:val for key, val in zip(header, f.readline().strip().split())}
    f.readline() # newline
    f.readline() # numbers
    header = f.readline()
    for key in header: obj[key] = []
    for line in f:
        for key, val in zip(header, line.strip().split()):
            obj[key] += [val]
    f.close()

    for key, val in obj.items():
        if isinstance(val, list):
            obj[key] = np.asarray(val, dtype=object).astype(float)
        else:
            obj[key] = starsmashertools.helpers.string.parse(val)
    
    return obj
"""


# Check if 2 files are identical
#@profile
def compare(file1, file2):
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
    
    with open(path, 'r') as f:
        contents = f.read()

    if phrase not in contents: return None

    ret = contents.split(phrase)[1:]
    for i, string in enumerate(ret):
        if end in string:
            ret[i] = string[:string.index(end)]
    
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
    ret = copy.deepcopy(files)
    ret.sort(key=lambda f: starsmashertools.helpers.path.getmtime(f))
    return ret[::-1]




def reverse_readline(path, buf_size=8192):
    """A generator that returns the lines of a file in reverse order
    https://stackoverflow.com/a/23646049/4954083"""
    with open(path, 'rb') as fh:
        segment = None
        offset = 0
        fh.seek(0, os.SEEK_END)
        file_size = remaining_size = fh.tell()
        while remaining_size > 0:
            offset = min(file_size, offset + buf_size)
            fh.seek(file_size - offset)
            buffer = fh.read(min(remaining_size, buf_size)).decode(encoding='utf-8')
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



