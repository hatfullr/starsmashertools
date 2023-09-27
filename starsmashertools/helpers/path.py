# Path helper methods
import os
import starsmashertools.preferences as preferences
import glob
import starsmashertools.helpers.ssh
import numpy as np
import fnmatch
import collections
import warnings

def islink(path):
    return os.path.islink(path)

def getsize(path):
    return os.path.getsize(path)

#@profile
def realpath(path):
    if starsmashertools.helpers.ssh.isRemote(path):
        address, remote_path = starsmashertools.helpers.ssh.split_address(path)
        result = starsmashertools.helpers.ssh.run_python(
            address,
            """
import os
import glob
path='%s'
g = path
if '*' in path:
    g = glob.glob(path)
    if g: g = g[0]
expanded = os.path.expandvars(os.path.expanduser(g))
print(os.path.realpath(expanded))
            """ % remote_path,
        )
        return address + ":" + result
    
    g = path
    if '*' in path:
        g = glob.glob(path)
        if g: g = g[0]
    expanded = os.path.expandvars(os.path.expanduser(g))
    return os.path.realpath(expanded)

def isfile(path, isRemote=None):
    if isRemote is not None and not isRemote:
        return os.path.isfile(path)
    
    if starsmashertools.helpers.ssh.isRemote(path):
        address, remote_path = starsmashertools.helpers.ssh.split_address(path)
        return bool(starsmashertools.helpers.ssh.run_python(
            address, "import os; print(os.path.isfile('%s'))" % remote_path
        ))
    return os.path.isfile(path)

def basename(path):
    return os.path.basename(path)

def getmtime(path):
    return os.path.getmtime(path)

def rename(path, newpath):
    return os.rename(path, newpath)

def walk(directory):
    if starsmashertools.helpers.ssh.isRemote(directory):
        address, remote_directory = starsmashertools.helpers.ssh.split_address(directory)
        result = starsmashertools.helpers.ssh.run_python(
            address,
            "import os; print([stuff for stuff in os.walk('%s')])" % remote_directory,
        )
        return eval(result) # eval converts the resultign string to a list
    return [stuff for stuff in os.walk(directory)]

def remove(path):
    if starsmashertools.helpers.ssh.isRemote(path):
        raise NotImplementedError
    else:
        os.remove(path)

def join(*args):
    if len(args) == 1:
        raise Exception("You must pass more than 1 argument to join")
    isRemote = []
    for a in args:
        isRemote += [ starsmashertools.helpers.ssh.isRemote(a) ]

    if not any(isRemote):
        return os.path.join(*args)
    else:
        if any(isRemote[1:]):
            raise Exception("Only the first argument can be a remote file path")

        address, remote_path = starsmashertools.helpers.ssh.split_address(args[0])
        return address + ":" + os.path.join(remote_path, *args[1:])

def dirname(path):
    return os.path.dirname(path)

def isdir(path):
    return os.path.isdir(path)

def scandir(path):
    return os.scandir(path)


pattern_matches = collections.OrderedDict()

# Given the path to a file, search search_directory for the first duplicate file.
#@profile
def find_duplicate_file(filepath, search_directory, pattern="out*.sph", throw_error=False):
    filepath = realpath(filepath)

    isRemote = starsmashertools.helpers.ssh.isRemote(filepath)
    
    search_directory = realpath(search_directory)
    filedirectory = dirname(filepath)

    search_string = join(search_directory,"**",pattern)

    # Try to limit exhaustive searches
    matches = None
    if search_string in pattern_matches.keys():
        matches = pattern_matches[search_string]
    else:
        matches = glob.glob(search_string, recursive=True)
        pattern_matches[search_string] = matches
    
    for match in matches:
        if (isfile(match, isRemote=isRemote) and
            match != filepath and
            starsmashertools.helpers.file.compare(filepath, match)):
            return match
    if throw_error:
        raise Exception("Found no duplicate '%s' file for '%s' in search directory '%s'" % (pattern, filepath, search_directory))
    return None




def get_src(directory, throw_error=False):
    if not isdir(directory):
        raise FileNotFoundError("Directory does not exist: '%s'" % str(directory))
    
    if directory == '/': raise Exception
    #print("get_src",directory)
    directory = realpath(directory)
    
    src_identifiers = preferences.get_default('Simulation', 'src identifiers', throw_error=True)

    if starsmashertools.helpers.ssh.isRemote(directory):
        address, remote_path = starsmashertools.helpers.ssh.split_address(directory)
        result = starsmashertools.helpers.ssh.run_python(
            address,
            """
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
                    path = obj.path
                    contents = [o.name for o in scandir(path) if o.is_file()]
                    for filename in src_identifiers:
                        if filename not in contents: break
                    else:
                        return path

    if throw_error:
        raise Exception("Failed to find the source directory in '%s'" % directory)





