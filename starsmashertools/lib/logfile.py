import starsmashertools.preferences as preferences
import starsmashertools.helpers.path
import starsmashertools.helpers.file
from glob import glob
import numpy as np
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

    @property
    def header(self):
        if self._header is None: self.read()
        return self._header
    
    def read(self):
        self._header = ""
        with starsmashertools.helpers.file.open(self.path, 'r') as f:
            for line in f:
                if 'output: end of iteration' in line: break
                self._header += line

    def get(self, phrase):
        if phrase not in self.header:
            raise LogFile.PhraseNotFoundError("Failed to find '%s' in '%s'" % (phrase, self.path))
        i0 = self.header.index(phrase) + len(phrase)
        i1 = i0 + self.header[i0:].index('\n')
        return self.header[i0:i1]

    # Return a list of boolean values of the same length as 'outputfiles', where
    # an element is 'True' if it is included in this LogFile.
    def hasOutputFiles(self, filenames):
        first_file = None
        last_file = None
        
        with starsmashertools.helpers.file.open(self.path, 'r') as f:
            # Search for the first file
            f.seek(len(self.header))
            for line in f:
                if ' duout: writing file ' in line:
                    l = line.replace(' duout: writing file ','')
                    idx = l.index('at t=')
                    first_file = l[:idx]
                    break

            # Search for the last file
            f.seek(0, 2)
            size = f.tell()
            c = 1
            while last_file is None:
                f.seek(max(0, size - 1048*c))
                c += 1
                content = f.read(int(1048 * 1.5))
                if ' duout: writing file ' in content:
                    i0 = content.rindex(' duout: writing file ') + len(' duout: writing file ')
                    i1 = i0 + content[i0:].index('at t=')
                    last_file = content[i0:i1]

        # This is only going to work with out*.sph files
        def get_filenum(filename):
            filename = starsmashertools.helpers.path.basename(filename)
            if 'out' not in filename:
                raise Exception("hasOutputFiles only works with files of type out*.sph")
            return int(filename.replace('out','').replace('.sph',''))
        
        start_num = get_filenum(first_file)
        stop_num = get_filenum(last_file)

        nums = [get_filenum(filename) for filename in filenames]
        return [num >= start_num and num <= stop_num for num in nums]
        """
        # Figure out the time domain of this log file
        start_time = self.get_start_time()
        stop_time = self.get_stop_time()

        outputfiles = [starsmashertools.lib.output.Output(filename, self.simulation) for filename in filenames]

        times = [None]*len(outputfiles)
        
        # Use the midpoint rule to figure out which output file is closest to
        # either side of the time domain
        def locate_file(time, precision=1.e-4):
            low = 0
            high = len(outputfiles)
            mid = int((high + low) * 0.5)

            while True:
                #if times[low] is None:
                #    times[low] = outputfiles[low]['t']
                #if times[high] is None:
                #    times[high] = outputfiles[high]['t']
                if times[mid] is None:
                    times[mid] = outputfiles[mid]['t']

                print("%15.7E %15.7E %5d %5d %5d" % (time, times[mid], low, mid, high))
                    
                if time < times[mid]:
                    if mid == high: break
                    high = copy.copy(mid)
                    mid = int(0.5*(low + mid))
                    #if mid == low: break
                elif time > times[mid]:
                    if mid == low: break
                    low = copy.copy(mid)
                    mid = int(0.5*(mid + high))
                    #if mid == high: break
            return outputfiles[mid]
        print(locate_file(start_time))
        print(locate_file(stop_time))
        quit()
        """
    
    # Find the time that this log file stopped at
    def get_stop_time(self):
        string = 'time='
        with starsmashertools.helpers.file.open(self.path, 'r') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 10480))
            content = f.read()

        if string not in content:
            raise LogFile.PhraseNotFoundError(string)
        
        i0 = content.rindex(string) + len(string)
        i1 = i0 + content[i0:].index('\n')
        return float(content[i0:i1])
        
    def get_start_time(self):
        string = 'time='
        with starsmashertools.helpers.file.open(self.path, 'r') as f:
            f.seek(len(self.header))
            for line in f:
                if string in line:
                    i0 = line.index(string) + len(string)
                    i1 = i0 + line[i0:].index('\n')
                    return float(line[i0:i1])
        raise LogFile.PhraseNotFoundError(string)

    def get_first_out_file(self):
        string = ' duout: writing file '
        with starsmashertools.helpers.file.open(self.path, 'r') as f:
            f.seek(len(self.header))
            for line in f:
                if string in line:
                    i0 = line.index(string) + len(string)
                    i1 = i0 + line[i0:].index('at t=')
                    return line[i0:i1]
        raise LogFile.PhraseNotFoundError(string)

    def get_last_out_file(self):
        string = ' duout: writing file '
        read_size = 10480
        with starsmashertools.helpers.file.open(self.path, 'r') as f:
            f.seek(0, 2)
            size = f.tell()
            nchunks = int(size / read_size) + 1
            pos = max(0, size - read_size)
            diff = 0
            content = ""
            i = 0
            while True:
                # Move the head to the new position
                f.seek(pos)
                
                # Add file contents to content
                content = f.read(min(size, read_size + diff)) + content
                
                # Check for string in the content
                if string in content:
                    i0 = content.rindex(string) + len(string)
                    i1 = i0 + content[i0:].index('at t=')
                    return content[i0:i1]
                
                # If we reached the end of the file, break
                if pos == 0: break
                
                # Move the head position up the file
                diff = pos - read_size # Can be < 0 only near end of reading
                pos = max(0, diff)
                
                # Discard old content that we don't need to check anymore
                if i > 0: content = content[:-read_size]
                i += 1

        raise LogFile.PhraseNotFoundError(string)

    class PhraseNotFoundError(Exception): pass
