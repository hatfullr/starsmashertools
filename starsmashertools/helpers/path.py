# Path helper methods
import os
import starsmashertools.helpers.argumentenforcer
import glob
import starsmashertools.helpers.ssh
import numpy as np
import collections
import warnings
import starsmashertools.helpers.string

def makedirs(path : str, **kwargs):
    if starsmashertools.helpers.ssh.isRemote(path):
        raise NotImplementedError
    if isdir(path): return
    return os.makedirs(path, **kwargs)

def getcwd():
    return os.getcwd()

def samefile(file_or_dir1 : str, file_or_dir2 : str):
    if (starsmashertools.helpers.ssh.isRemote(file_or_dir1) or
        starsmashertools.helpers.ssh.isRemote(file_or_dir2)):
        raise NotImplementedError
    return os.path.samefile(file_or_dir1, file_or_dir2)

def replace(file1 : str, file2 : str):
    if (starsmashertools.helpers.ssh.isRemote(file1) or
        starsmashertools.helpers.ssh.isRemote(file2)):
        raise NotImplementedError
    return os.replace(file1, file2)

def utime(path, *args, **kwargs):
    if starsmashertools.helpers.ssh.isRemote(path):
        raise NotImplementedError
    return os.utime(path, *args, **kwargs)

def islink(path):
    if starsmashertools.helpers.ssh.isRemote(path):
        raise NotImplementedError
    return os.path.islink(path)

def getsize(path : str):
    r"""
    Return the size, in bytes, of `path`\.

    Parameters
    ----------
    path : str
        The path to a file or directory.

    Returns
    -------
    int
        The size, in bytes, of `path`\.
    """
    if starsmashertools.helpers.ssh.isRemote(path):
        address, remote_path = starsmashertools.helpers.ssh.split_address(path)
        result = starsmashertools.helpers.ssh.run_python(
            address,
            r"""
import os
print(os.path.getsize('%s'))
            """ % remote_path,
        )
        return result
    return os.path.getsize(path)

@starsmashertools.helpers.argumentenforcer.enforcetypes
def get_directory_size(path : str):
    r"""Returns the total size of the directory given by `path` in bytes
    
    Parameters
    ----------
    path : str
        The path of the directory. If the path is not a valid directory, raises
        a FileNotFoundError.
    
    Returns
    -------
    int
    """
    if not isdir(path): raise FileNotFoundError(path)
    total = 0
    for entry in scandir(path):
        try: total += entry.stat().st_size
        except: pass
    return total
    

#@profile
def realpath(path):
    if starsmashertools.helpers.ssh.isRemote(path):
        address, remote_path = starsmashertools.helpers.ssh.split_address(path)
        result = starsmashertools.helpers.ssh.run_python(
            address,
            r"""
import os
import glob
path='%s'
if '*' in path:
    g = glob.glob(path)
    if g: 
        print(os.path.realpath(os.path.expandvars(os.path.expanduser(g[0]))))  
else: print(os.path.realpath(os.path.expandvars(os.path.expanduser(path))))
            """ % remote_path,
        )
        return address + ":" + result
    
    if '*' in path:
        g = glob.glob(path)
        if g: return os.path.realpath(os.path.expandvars(os.path.expanduser(g[0])))
    return os.path.realpath(os.path.expandvars(os.path.expanduser(path)))

def relpath(path, start=os.curdir):
    if starsmashertools.helpers.ssh.isRemote(path):
        raise NotImplementedError
    return os.path.relpath(path, start=start)

def exists(path, isRemote=None):
    if isRemote is None:
        isRemote = starsmashertools.helpers.ssh.isRemote(path)
    if isRemote:
        address, remote_path = starsmashertools.helpers.ssh.split_address(path)
        return starsmashertools.helpers.string.parse(
            starsmashertools.helpers.ssh.run_python(
            address, "import os; print(os.path.exists('%s'))" % remote_path
        ))
    return os.path.exists(path)

def isfile(path, isRemote=None):
    if isRemote is None and starsmashertools.helpers.ssh.isRemote(path):
        address, remote_path = starsmashertools.helpers.ssh.split_address(path)
        return starsmashertools.helpers.string.parse(
            starsmashertools.helpers.ssh.run_python(
            address, "import os; print(os.path.isfile('%s'))" % remote_path
        ))
    return os.path.isfile(path)

def basename(path):
    return os.path.basename(path)

def getmtime(path):
    if starsmashertools.helpers.ssh.isRemote(path):
        raise NotImplementedError
    # Use stat instead of os.path.getmtime, for reasons detailed in the official
    # documentation:
    #    Note that the exact times you set here may not be returned by a
    #    subsequent stat() call, depending on the resolution with which your
    #    operating system records access and modification times; see stat(). The
    #    best way to preserve exact times is to use the st_atime_ns and
    #    st_mtime_ns fields from the os.stat() result object with the ns
    #    parameter to utime().
    stat_result = os.stat(path)
    return int(stat_result.st_mtime_ns * 1e-9)
    #return os.path.getmtime(path)

def rename(path, newpath):
    if starsmashertools.helpers.ssh.isRemote(path):
        raise NotImplementedError
    return os.rename(path, newpath)

def walk(directory):
    if starsmashertools.helpers.ssh.isRemote(directory):
        address, remote_directory = starsmashertools.helpers.ssh.split_address(directory)
        # eval converts the resulting string to a list
        return eval(starsmashertools.helpers.ssh.run_python(
            address,
            "import os; print([stuff for stuff in os.walk('%s')])" % remote_directory,
        ))
    return [stuff for stuff in os.walk(directory)]

def listdir(directory):
    if starsmashertools.helpers.ssh.isRemote(directory):
        raise NotImplementedError
    return os.listdir(directory)

def find_files(
        directory,
        recursive : bool = True,
        include_symlinks : bool = True,
):
    if starsmashertools.helpers.ssh.isRemote(directory):
        raise NotImplementedError
    def get(path):
        with scandir(path) as it:
            for entry in it:
                if not include_symlinks and (entry.is_symlink() or entry.is_junction()):
                    continue
                if entry.is_dir():
                    if recursive: yield from get(realpath(entry.path))
                    continue
                # Must be a file
                yield realpath(entry.path)
    return get(realpath(directory))

def remove(path):
    if starsmashertools.helpers.ssh.isRemote(path):
        raise NotImplementedError
    return os.remove(path)

def join(*args):
    if len(args) == 1:
        raise Exception("You must pass more than 1 argument to join")
    isRemote = [starsmashertools.helpers.ssh.isRemote(a) for a in args]

    if not any(isRemote):
        return os.path.join(*args)
    else:
        if any(isRemote[1:]):
            raise Exception("Only the first argument can be a remote file path")

        address, remote_path = starsmashertools.helpers.ssh.split_address(args[0])
        return address + ":" + os.path.join(remote_path, *args[1:])

def dirname(path):
    if starsmashertools.helpers.ssh.isRemote(path):
        raise NotImplementedError
    return os.path.dirname(path)

def isdir(path):
    if starsmashertools.helpers.ssh.isRemote(path):
        raise NotImplementedError
    return os.path.isdir(path)

def scandir(path):
    if starsmashertools.helpers.ssh.isRemote(path):
        raise NotImplementedError
    return os.scandir(path)

pattern_matches = collections.OrderedDict()

# Given the path to a file, search search_directory for the first duplicate file.
#@profile
def find_duplicate_file(
        filepath : str,
        search_directory : str,
        pattern : str = 'out*.sph',
        throw_error : bool = False,
        **kwargs,
):
    for f in find_duplicate_files(
            filepath, search_directory, pattern = pattern, **kwargs
    ):
        return f
    if throw_error: 
        raise Exception("Found no duplicate '%s' file for '%s' in search directory '%s'" % (pattern, filepath, search_directory))


def find_duplicate_files(
        filepath : str,
        search_directory : str,
        pattern : str = "out*.sph",
        exclude : list | tuple = [],
):
    import starsmashertools.helpers.file
    
    filepath = realpath(filepath)

    isRemote = starsmashertools.helpers.ssh.isRemote(filepath)
    
    search_directory = realpath(search_directory)
    filedirectory = dirname(filepath)

    search_string = join(search_directory,"**",pattern)
    
    for match in pattern_matches.get(
            search_string,
            glob.glob(search_string, recursive=True),
    ):
        if match in exclude: continue
        if (match != filepath and
            isfile(match, isRemote=isRemote) and
            starsmashertools.helpers.file.compare(filepath, match)):
            yield match




def get_src(directory, throw_error=False):
    import starsmashertools.lib.simulation
    
    if not isdir(directory):
        raise FileNotFoundError("Directory does not exist: '%s'" % str(directory))
    
    if directory == os.path.abspath(os.sep): # root directory
        raise Exception("A search for the StarSmasher source directory extended to the root directory and thus we failed to find the source directory. Please make sure there is a copy of the StarSmasher source code in your simulation directory.")
    #print("get_src",directory)
    directory = realpath(directory)

    src_identifiers = starsmashertools.lib.simulation.Simulation.preferences.get(
        'src identifiers',
    )

    if starsmashertools.helpers.ssh.isRemote(directory):
        address, remote_path = starsmashertools.helpers.ssh.split_address(directory)
        result = starsmashertools.helpers.ssh.run_python(
            address,
            r"""
import os
for obj in os.scandir('%s'):
    if obj.is_dir():
        path = obj.path
        contents = [o.name for o in os.scandir(path) if o.is_file()]
        for filename in %s:
            if filename not in contents: break
        else:
            print(path)
            break
print('')
            """ % (remote_path, str(src_identifiers)),
        )
        if result:
            return address + ":" + result
    else:
        # Sometimes it seems we get a strange warning like:
        # 
        # : ResourceWarning: unclosed scandir iterator <posix.ScandirIterator object at 0x7f94c5907b40>
        #   return path
        # ResourceWarning: Enable tracemalloc to get the object allocation traceback
        #
        # This doesn't seem to be our fault, so we ignore the warnings here.

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ResourceWarning)
            for obj in scandir(directory):
                if obj.is_dir():
                    contents =[o.name for o in scandir(obj.path) if o.is_file()]
                    for filename in src_identifiers:
                        if filename not in contents: break
                    else:
                        return obj.path

    if throw_error:
        raise FileNotFoundError("Failed to find the source directory in '%s'. Please make sure there is a directoy which contains all the following file names: %s" % (directory, starsmashertools.helpers.string.list_to_string(src_identifiers)))





