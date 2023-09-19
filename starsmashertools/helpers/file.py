import starsmashertools.helpers.path
import starsmashertools.helpers.ssh
import starsmashertools.helpers.string
import tempfile
import builtins
import os
import filecmp
import numpy as np
import contextlib

fortran_comment_characters = ['c','C','!','*']

downloaded_files = []

@contextlib.contextmanager
def open(path, mode, **kwargs):
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

        f = builtins.open(tfile.name, mode, **kwargs)
    else:
        f = builtins.open(path, mode, **kwargs)
    
    try:
        yield f
    finally:
        f.close()



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

def getsize(path):
    if starsmashertools.helpers.ssh.isRemote(path):
        address, remote_path = starsmashertools.helpers.ssh.split_address(path)
        result = starsmashertools.helpers.ssh.run_python(
            address,
            """
import os
print(os.path.getsize('%s'))
            """ % remote_path,
        )
        return result
    return os.path.getsize(path)


# Check if 2 files are identical
#@profile
def compare(file1, file2):
    #file1 = starsmashertools.helpers.path.realpath(file1)
    #file2 = starsmashertools.helpers.path.realpath(file2)
    
    # If they have the same path, then they are the same file
    if file1 == file2: return True
    
    size1 = getsize(file1)
    size2 = getsize(file2)

    if size1 == size2:
        return filecmp.cmp(file1, file2, shallow=False)
    return False
