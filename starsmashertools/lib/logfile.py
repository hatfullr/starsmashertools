import starsmashertools.preferences as preferences
import starsmashertools.helpers.path
import starsmashertools.helpers.file
from glob import glob
import numpy as np
import mmap
import copy

def find(directory, pattern=None, throw_error=False):
    if pattern is None:
        pattern = preferences.get_default('LogFile', 'file pattern', throw_error=True)
    direc = starsmashertools.helpers.path.realpath(directory)
    tosearch = starsmashertools.helpers.path.join(
        direc,
        '**',
        pattern,
    )

    matches = glob(tosearch, recursive=True)
    if matches: matches = sorted(matches)
    elif throw_error: raise FileNotFoundError("No log files matching pattern '%s' in directory '%s'" % (pattern, directory))

    return matches

    
class LogFile(object):
    def __init__(self, path, simulation):
        self.path = starsmashertools.helpers.path.realpath(path)
        self.simulation = simulation
        self._header = None

        with starsmashertools.helpers.file.open(self.path, 'rb') as f:
            self._buffer = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    @property
    def header(self):
        if self._header is None: self.read_header()
        return self._header
    
    def read_header(self):
        self._header = b""
        self._buffer.seek(0)
        for line in iter(self._buffer.readline, b""):
            if b'output: end of iteration' in line: break
            self._header += line
        self._header = self._header.decode('utf-8')

    def get(self, phrase):
        if phrase not in self.header:
            raise LogFile.PhraseNotFoundError("Failed to find '%s' in '%s'" % (phrase, self.path))
        i0 = self.header.index(phrase) + len(phrase)
        i1 = i0 + self.header[i0:].index('\n')
        return self.header[i0:i1]

    # Return a list of boolean values of the same length as 'outputfiles', where
    # an element is 'True' if it is included in this LogFile.
    def has_output_files(self, filenames):
        first_file = self.get_first_output_file(throw_error=False)
        last_file = self.get_last_output_file(throw_error=False)
        
        # This is only going to work with out*.sph files
        def get_filenum(filename):
            filename = starsmashertools.helpers.path.basename(filename)
            if 'out' not in filename:
                raise Exception("has_output_files only works with files of type out*.sph")
            return int(filename.replace('out','').replace('.sph',''))
        
        start_num = get_filenum(first_file)
        stop_num = get_filenum(last_file)

        nums = [get_filenum(filename) for filename in filenames]
        return [num >= start_num and num <= stop_num for num in nums]
    
    # Find the time that this log file stopped at
    def get_stop_time(self):
        string = 'time='
        bstring = string.encode('utf-8')
        index = self._buffer.rfind(bstring, len(self.header), self._buffer.size())
        if index == -1:
            raise LogFile.PhraseNotFoundError(string)
        index += len(bstring)
        end_idx = self._buffer.find(b'\n', index, self._buffer.size())
        self._buffer.seek(index)
        return float(self._buffer.read(end_idx - index).decode('utf-8').strip())
        
    def get_start_time(self):
        string = 'time='
        bstring = string.encode('utf-8')
        index = self._buffer.find(bstring, len(self.header), self._buffer.size())
        if index == -1:
            raise LogFile.PhraseNotFoundError(string)
        index += len(bstring)
        end_idx = self._buffer.find(b'\n', index, self._buffer.size())
        self._buffer.seek(index)
        return float(self._buffer.read(end_idx - index).decode('utf-8').strip())

    def get_first_output_file(self, throw_error=True):
        string = ' duout: writing file '
        bstring = string.encode('utf-8')
        string2 = 'at t='
        bstring2 = string2.encode('utf-8')
        
        index = self._buffer.find(bstring, len(self.header), self._buffer.size())
        if index != -1:
            index += len(bstring)
            self._buffer.seek(index)
            end_idx = self._buffer.find(bstring2)
            return self._buffer.read(end_idx - index).decode('utf-8').strip()

        if throw_error:
            raise LogFile.PhraseNotFoundError(string)
        

    def get_last_output_file(self, throw_error=True):
        string = ' duout: writing file '
        bstring = string.encode('utf-8')
        string2 = 'at t='
        bstring2 = string2.encode('utf-8')
        
        index = self._buffer.rfind(bstring, len(self.header), self._buffer.size())
        if index != -1:
            index += len(bstring)
            self._buffer.seek(index)
            end_idx = self._buffer.find(bstring2, index, self._buffer.size())
            return self._buffer.read(end_idx - index).decode('utf-8').strip()

        if throw_error:
            raise LogFile.PhraseNotFoundError(string)
        

    class PhraseNotFoundError(Exception): pass
