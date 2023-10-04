import starsmashertools.preferences as preferences
import starsmashertools.helpers.path
import starsmashertools.helpers.file
import starsmashertools.helpers.string
from glob import glob
import numpy as np
import mmap
import copy
import re

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
        
        self._dts = None
        self._first_iteration = None
        self._last_iteration = None
        self._iteration_content_length = None

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
    
    def get_number_of_iterations(self):
        first = self.get_first_iteration()
        last = self.get_last_iteration()
        if None not in [first, last]:
            return last['number'] - first['number']
        return 0

    def get_iteration_content_length(self):
        if self._iteration_content_length is not None: return self._iteration_content_length
        startline = LogFile.Iteration.startline.encode('utf-8')
        endline = LogFile.Iteration.endline.encode('utf-8')

        stop = self._buffer.size()
        
        # The iteration lines are always the same length
        index = self._buffer.find(startline, len(self.header), stop)
        if index == -1: return None

        _start = index
        
        index2 = self._buffer.find(endline, index, stop)
        index3 = self._buffer.find(b'\n', index2, stop)
        self._iteration_content_length = index3 - index
        return self._iteration_content_length

    def get_first_iteration(self):
        if self._first_iteration is not None: return self._first_iteration
        string = LogFile.Iteration.startline.encode('utf-8')
        string2 = LogFile.Iteration.endline.encode('utf-8')
        size = self._buffer.size()
        index = self._buffer.find(string, len(self.header), size)
        if index == -1: return None

        length = self.get_iteration_content_length()
        self._buffer.seek(index)
        content = self._buffer.read(length)
        self._first_iteration = LogFile.Iteration(content, self)
        return self._first_iteration

    def get_last_iteration(self):
        if self._last_iteration is not None: return self._last_iteration
        string = LogFile.Iteration.startline.encode('utf-8')
        string2 = LogFile.Iteration.endline.encode('utf-8')
        size = self._buffer.size()
        index = self._buffer.rfind(string, len(self.header), size)
        if index == -1: return None

        length = self.get_iteration_content_length()
        self._buffer.seek(index)
        content = self._buffer.read(length)
        try:
            self._last_iteration = LogFile.Iteration(content, self)
        except LogFile.PhraseNotFoundError:
            # Try going up higher in the log file
            index = self._buffer.rfind(string, len(self.header), index)
            if index == -1: return None
            self._buffer.seek(index)
            content = self._buffer.read(length)
            self._last_iteration = LogFile.Iteration(content, self)
        return self._last_iteration

    def get_iteration(self, number):
        startline = LogFile.Iteration.startline
        tomatch = (startline + '%8d') % number
        length = self.get_iteration_content_length()
        first_iteration = self.get_first_iteration()
        start = length*(number - first_iteration['number']) # + len(self.header)
        index = self._buffer.find(
            tomatch.encode('utf-8'),
            start,
            self._buffer.size(),
        )
        if index == -1: return None
        
        self._buffer.seek(index)
        content = self._buffer.read(length)
        return LogFile.Iteration(content, self)

    
    def get_iterations(self, start : int = 0, stop=None, step : int = 1):
        # This function is as well-optimized as I can get it to be
        # It's still a big bottleneck, but just read fewer iterations from
        # log files and you'll be fine...

        num_iters = self.get_number_of_iterations()
        if stop is None: stop = num_iters
        toget = np.arange(start, stop, step)

        first_iteration = self.get_first_iteration()
        iterations = []
        for number in toget:
            try:
                iteration = self.get_iteration(first_iteration['number'] + number)
            except LogFile.PhraseNotFoundError:
                break
            if iteration is None: raise Exception("Failed to find iteration %d" % (first_iteration['number'] + number))
            print("Got iteration %d" % iteration['number'])
            iterations += [iteration]

        return iterations

    class PhraseNotFoundError(Exception): pass



    class Iteration(starsmashertools.helpers.readonlydict.ReadOnlyDict, object):
        startline = ' output: end of iteration '
        endline = 'indx'

        # If you want to scrape more information, try using
        # https://www.getthedata.com/character-count
        # or any other online character counter and copy+paste an iteration line
        # starting at startline and ending at endline
        # It also helps to check src/output.f
        positions = {
            'number' : [25, 41],
            'time' : [46, 58],
            'system box xmin' : [ 73,  84],
            'system box xmax' : [ 89, 99],
            'system box ymin' : [100, 125],
            'system box ymax' : [130, 140],
            'system box zmin' : [141, 166],
            'system box zmax' : [171, 181],
            'W' : [277, 288],
            'T' : [290, 301],
            'U' : [303, 313],
            'Etot' : [324, 335],
            'Stot' : [340, 351],
            'vcm' : [355, 366],
            'Jtot' : [371, 381],
            'avr neighbors' : [400, 405],
            'sig neighbors' : [409, 415],
            'min neighbors' : [419, 423],
            'max neighbors' : [428, 433],
            'rhomin' : [465, 476],
            'rhomax' : [539, 550],
            'pmin' : [613, 624],
            'pmax' : [687, 698],
            'umin' : [761, 772],
            'umax' : [835, 846],
            'hmin' : [909, 920],
            'hmax' : [983, 994],
        }

        def __init__(self, contents, logfile):
            self.logfile = logfile
            obj = {}
            contents = contents.decode('utf-8')
            lines = contents.split('\n')
            
            for key, (a, b) in LogFile.Iteration.positions.items():
                obj[key] = float(contents[a:b])

            for key in ['number', 'min neighbors', 'max neighbors']:
                obj[key] = int(obj[key])


            
            # The method below is a bit too slow. Replacing in favor of exact
            # matching by positions
            
            """
            l = lines[0].replace(LogFile.Iteration.startline,'').replace('time=','')
            l = l.split()
            obj['number'] = int(l[0])
            obj['time'] = float(l[1])

            l1 = lines[1].replace('system box=', '').split()
            l2 = lines[2].split()
            l3 = lines[3].split()
            obj['system box'] = np.asarray([
                [l1[0], l1[-1]],
                [l2[0], l2[-1]],
                [l3[0], l3[-1]],
            ], dtype=object).astype(float)

            # This is really helpful: https://regex101.com/
            # Get all "something=" statements
            matches = re.findall(r'[\s][\w]*[=][\s]*[+-]?[\d]*[.]?[\d]*[eE]?[+-]?[\d]*', "\n".join(lines[4:-2]))
            for match in matches:
                key, val = match.split('=')
                obj[key.strip()] = float(val)
            """
            # The last 2 lines should always be "dts=..." and "indx..."
            if 'dts=' not in lines[-2]:
                raise LogFile.PhraseNotFoundError("The second-to-last line in the content was not 'dts=...' in '%s'. Perhaps the log file has been cutoff?" % (str(self.logfile.path)))
            if 'indx' not in lines[-1]:
                raise LogFile.PhraseNotFoundError("The last line in the content was not 'indx...' in '%s'. Perhaps the log file has been cutoff?" % (str(self.logfile.path)))
            values = lines[-2].split('=')[1].split()
            obj['dts'] = np.asarray(values, dtype=object).astype(float)
            values = lines[-1].split()[1:]
            obj['indx'] = np.asarray(values, dtype=object).astype(int)
            #"""
            
            super(LogFile.Iteration, self).__init__(obj)
            


