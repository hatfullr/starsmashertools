import starsmashertools.preferences
from starsmashertools.preferences import Pref
import starsmashertools.helpers.argumentenforcer
from starsmashertools.helpers.apidecorator import api
from glob import glob
import numpy as np
import mmap
import collections
import contextlib


@api
@starsmashertools.helpers.argumentenforcer.enforcetypes
@starsmashertools.preferences.use
def find(
        directory : str,
        pattern : str = Pref('pattern', 'log*.sph'),
        throw_error : bool = False,
):
    import starsmashertools.helpers.path
    
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
    class PhraseNotFoundError(Exception):
        def __init__(self, message, *args, **kwargs):
            super(LogFile.PhraseNotFoundError, self).__init__(
                "'%s'" % str(message), *args, **kwargs
            )
    
    @api
    def __init__(self, path, simulation):
        import starsmashertools.helpers.path
        import starsmashertools.lib.simulation

        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'path' : [str],
            'simulation' : [starsmashertools.lib.simulation.Simulation],
        })
        
        self.path = starsmashertools.helpers.path.realpath(path)
        self.simulation = simulation
        self._header = None
        
        self._dts = None
        self._first_iteration = None
        self._last_iteration = None
        self._iteration_content_length = None

        #self._buffer = None

    def __eq__(self, other):
        import starsmashertools.helpers.file
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'other' : LogFile,
        })
        return starsmashertools.helpers.file.compare(self.path, other.path)

    """
    @property
    def buffer(self):
        if self._buffer is not None: return self._buffer
        import starsmashertools.helpers.path
        import starsmashertools.helpers.file
        with starsmashertools.helpers.file.open(self.path, 'rb', lock = False) as f:
            try:
                self._buffer = mmap.mmap(
                    f.fileno(), 0,
                    flags=mmap.MAP_POPULATE,
                    access=mmap.ACCESS_READ,
                )
            except:
                self._buffer = mmap.mmap(
                    f.fileno(), 0,
                    access=mmap.ACCESS_READ
                )
        return self._buffer
    """
    
    @property
    def header(self):
        if self._header is None: self.read_header()
        return self._header

    @contextlib.contextmanager
    def get_buffer(self):
        import starsmashertools.helpers.path
        import starsmashertools.helpers.file
        
        with starsmashertools.helpers.file.open(
                self.path, 'rb', lock = False
        ) as f:
            try:
                yield mmap.mmap(
                    f.fileno(), 0,
                    flags=mmap.MAP_POPULATE,
                    access=mmap.ACCESS_READ,
                )
            except:
                yield mmap.mmap(
                    f.fileno(), 0,
                    access=mmap.ACCESS_READ,
                )
    
    def read_header(self):
        self._header = b""
        with self.get_buffer() as buffer:
            for line in iter(buffer.readline, b""):
                if b'output: end of iteration' in line: break
                self._header += line
        self._header = self._header.decode('utf-8')

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get(
            self,
            phrase : str,
    ):
        if phrase not in self.header:
            raise LogFile.PhraseNotFoundError("Failed to find '%s' in '%s'" % (phrase, self.path))
        i0 = self.header.index(phrase) + len(phrase)
        i1 = i0 + self.header[i0:].index('\n')
        return self.header[i0:i1]

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def has_output_files(
            self,
            paths : list | tuple | np.ndarray,
    ):
        r"""
        Return a list of boolean values of the same length as ``paths``, where
        an element is ``True`` if it is included in this LogFile.

        Parameters
        ----------
        paths : list, tuple, :class:`numpy.ndarray`
            The file paths to check.

        Returns
        -------
        list
            A list of boolean values with the same length as ``paths``\, where
            elements are ``True`` if this log file contains the output file at
            the corresponding path in ``paths``\. Otherwise, that element is
            ``False``\.
        """
        import starsmashertools.helpers.path
        
        first_file = self.get_first_output_file(throw_error=False)
        last_file = self.get_last_output_file(throw_error=False)

        if first_file is None and last_file is None:
            # No output files written in this log file
            return [False]*len(paths)
        if None in [first_file, last_file]:
            raise Exception("Exactly one of first_file or last_file were None. This should never be possible.")
            
        
        # This is only going to work with out*.sph files
        def get_filenum(path):
            basename = starsmashertools.helpers.path.basename(path)
            if 'out' not in basename:
                raise Exception("has_output_files only works with files of type out*.sph")
            return int(basename.replace('out','').replace('.sph',''))
        
        start_num = get_filenum(first_file)
        stop_num = get_filenum(last_file)

        nums = [get_filenum(path) for path in paths]
        return [num >= start_num and num <= stop_num for num in nums]
    
    # Find the time that this log file stopped at
    @api
    def get_stop_time(self):
        string = 'time='
        bstring = string.encode('utf-8')
        with self.get_buffer() as buffer:
            index = buffer.rfind(bstring, len(self.header), buffer.size())
            if index == -1:
                raise LogFile.PhraseNotFoundError(string)
            index += len(bstring)
            end_idx = buffer.find(b'\n', index, buffer.size())
            buffer.seek(index)
            ret = float(buffer.read(end_idx - index).decode('utf-8').strip())
        
        return ret

    @api
    def get_start_time(self):
        string = 'time='
        bstring = string.encode('utf-8')
        with self.get_buffer() as buffer:
            index = buffer.find(bstring, len(self.header), buffer.size())
            if index == -1:
                raise LogFile.PhraseNotFoundError(string)
            index += len(bstring)
            end_idx = buffer.find(b'\n', index, buffer.size())
            buffer.seek(index)
            ret = float(buffer.read(end_idx - index).decode('utf-8').strip())
        
        return ret

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_first_output_file(
            self,
            throw_error : bool = True,
    ):
        string = ' duout: writing file '
        bstring = string.encode('utf-8')
        string2 = 'at t='
        bstring2 = string2.encode('utf-8')
        with self.get_buffer() as buffer:
            index = buffer.find(bstring, len(self.header), buffer.size())
            ret = None
            if index != -1:
                index += len(bstring)
                buffer.seek(index)
                end_idx = buffer.find(bstring2)
                ret = buffer.read(end_idx - index).decode('utf-8').strip()
        
        if throw_error:
            raise LogFile.PhraseNotFoundError(string)
        return ret

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_last_output_file(
            self,
            throw_error : bool = True,
    ):
        string = ' duout: writing file '
        bstring = string.encode('utf-8')
        string2 = 'at t='
        bstring2 = string2.encode('utf-8')
        with self.get_buffer() as buffer:
            index = buffer.rfind(bstring, len(self.header), buffer.size())
            ret = None
            if index != -1:
                index += len(bstring)
                buffer.seek(index)
                end_idx = buffer.find(bstring2, index, buffer.size())
                ret = buffer.read(end_idx - index).decode('utf-8').strip()
        
        if throw_error:
            raise LogFile.PhraseNotFoundError(string)
        return ret

    @api
    def get_number_of_iterations(self):
        first = self.get_first_iteration()
        last = self.get_last_iteration()
        if None not in [first, last]:
            return last['iteration'] - first['iteration']
        return 0

    @api
    def get_iteration_content_length(self):
        if self._iteration_content_length is not None: return self._iteration_content_length
        
        startline = LogFile.Iteration.startline.encode('utf-8')
        endline = LogFile.Iteration.endline.encode('utf-8')

        with self.get_buffer() as buffer:
            stop = buffer.size()

            # The iteration lines are always the same length
            index = buffer.find(startline, len(self.header), stop)
            if index == -1: return None

            _start = index

            index2 = buffer.find(endline, index, stop)
            index3 = buffer.find(b'\n', index2, stop)
        
        self._iteration_content_length = index3 - index
        return self._iteration_content_length

    @api
    def get_first_iteration(self):
        if self._first_iteration is not None: return self._first_iteration
        
        string = LogFile.Iteration.startline.encode('utf-8')
        string2 = LogFile.Iteration.endline.encode('utf-8')
        with self.get_buffer() as buffer:
            size = buffer.size()
            index = buffer.find(string, len(self.header), size)
            if index == -1: return None

            length = self.get_iteration_content_length()
            buffer.seek(index)
            content = buffer.read(length)

        try:
            self._first_iteration = LogFile.Iteration(content, self)
        except:
            self._first_iteration = None
        return self._first_iteration

    @api
    def get_last_iteration(self):
        if self._last_iteration is not None: return self._last_iteration
        
        string = LogFile.Iteration.startline.encode('utf-8')
        string2 = LogFile.Iteration.endline.encode('utf-8')
        with self.get_buffer() as buffer:
            size = buffer.size()
            index = buffer.rfind(string, len(self.header), size)
            if index == -1: return None

            length = self.get_iteration_content_length()
            buffer.seek(index)
            content = buffer.read(length)
            try:
                self._last_iteration = LogFile.Iteration(content, self)
            except:
                # Try going up higher in the log file
                index = buffer.rfind(string, len(self.header), index)
                #print("Trying higher up")
                if index == -1: return None
                buffer.seek(index)
                content = buffer.read(length)
                self._last_iteration = LogFile.Iteration(content, self)
        
        return self._last_iteration

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_iteration(self, number : int):
        startline = LogFile.Iteration.startline
        tomatch = (startline + '%8d') % number
        length = self.get_iteration_content_length()
        first_iteration = self.get_first_iteration()
        start = length*(number - first_iteration['iteration']) # + len(self.header)
        with self.get_buffer() as buffer:
            index = buffer.find(
                tomatch.encode('utf-8'),
                start,
                buffer.size(),
            )
            if index == -1: return None

            buffer.seek(index)
            content = buffer.read(length)
        
        return LogFile.Iteration(content, self)

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_iterations(
            self,
            start : int = 0,
            stop : int | type(None) = None,
            step : int = 1,
    ):
        # This function is as well-optimized as I can get it to be
        # It's still a big bottleneck, but just read fewer iterations from
        # log files and you'll be fine...

        num_iters = self.get_number_of_iterations()
        if stop is None: stop = num_iters
        toget = np.arange(start, stop + 1, step)

        startline = LogFile.Iteration.startline
        length = self.get_iteration_content_length()
        first_iteration = self.get_first_iteration()
        start = len(self.header)
        with self.get_buffer() as buffer:
            buffer.seek(start)
            end = buffer.size()

            toget += first_iteration['iteration']

            for number in toget:
                tomatch = (startline + '%8d') % number
                index = buffer.find(
                    tomatch.encode('utf-8'),
                    start,
                    end,
                )
                if index == -1:
                    raise Exception("Failed to find iteration %d" % (first_iteration['iteration'] + number))

                # Get the content of the iteration
                buffer.seek(index)
                content = buffer.read(length)

                try:
                    yield LogFile.Iteration(content, self)
                except LogFile.PhraseNotFoundError:
                    break

                start = buffer.tell()

                # This code scales worse than above because it has to read the log
                # file at the front and back before seeking ahead. We do the front-
                # back seeking above before the loop to save time.
                #try:
                #    yield self.get_iteration(number)
                #except LogFile.PhraseNotFoundError:
                #    break
                #if iteration is None: raise Exception("Failed to find iteration %d" % (first_iteration['iteration'] + number))

    






    class Iteration(starsmashertools.helpers.readonlydict.ReadOnlyDict, object):
        # TODO: Replace the keys below with some form of smart searching through
        # the StarSmasher source code for exactly how the log files are written.
        startline = ' output: end of iteration '
        endline = 'indx'

        # First value: the type of value we expect
        # Second value: the string to search for in the content
        # Third value: the length of the content to read to convert to the type
        #              given in the first value.
        # Fourth value: the offset from the first value to read
        # Check src/output.f and src/tstep.f in StarSmasher for exact strings to
        # search for. The keys need to be in the same order as they are written
        # in the log files.
        
        keys = collections.OrderedDict()
        keys['iteration'] = [int, ' output: end of iteration ', 8, 0]
        keys['time'] = [float, '       time=', 12, 0]
        keys['system box xmin'] = [float, '   system box= ', 10, 0]
        keys['system box xmax'] = [float, keys['system box xmin'][1], 10, keys['system box xmin'][2] + keys['system box xmin'][3] + len('< x <')]
        keys['system box ymin'] = [float, keys['system box xmin'][1], 10, keys['system box xmax'][2] + keys['system box xmax'][3] + 1 + len('               ')] # +1 for newline
        keys['system box ymax'] = [float, keys['system box xmin'][1], 10, keys['system box ymin'][2] + keys['system box ymin'][3] + len('< y <')]
        keys['system box zmin'] = [float, keys['system box xmin'][1], 10, keys['system box ymax'][2] + keys['system box ymax'][3] + 1 + len('               ')] # +1 for newline
        keys['system box zmax'] = [float, keys['system box xmin'][1], 10, keys['system box zmin'][2] + keys['system box zmin'][3] + len('< z <')]
        keys['W'] = [float, '   energies: W=', 10, 0]
        keys['T'] = [float, ' T=', 10, 0]
        keys['U'] = [float, ' U=', 10, 0]
        keys['Etot'] = [float, '     Etot=', 10, 0]
        keys['Stot'] = [float, ' Stot=', 10, 0]
        keys['vcm'] = [float, ' vcm=', 10, 0]
        keys['Jtot'] = [float, ' Jtot=', 10, 0]
        keys['avr neighbors'] = [float, '   neighbors: avr=', 4, 0]
        keys['sig neighbors'] = [float, keys['avr neighbors'][1], 4, keys['avr neighbors'][2] + keys['avr neighbors'][3] + len(' sig=')]
        keys['min neighbors'] = [int, keys['sig neighbors'][1], 4, keys['sig neighbors'][2] + keys['sig neighbors'][3] + len('  min=')]
        keys['max neighbors'] = [int, keys['min neighbors'][1], 5, keys['min neighbors'][2] + keys['min neighbors'][3] + len(' max=')]
        keys['rhomin'] = [float, '   density: rhomin=', 10, 0]
        keys['rhomax'] = [float, '            rhomax=', 10, 0]
        keys['pmin'] = [float, '    pressure: pmin=', 10, 0]
        keys['pmax'] = [float, '              pmax=', 10, 0]
        keys['amin'] = [float, '  in entropy: amin=', 10, 0]
        keys['amax'] = [float, '              amax=', 10, 0]
        keys['umin'] = [float, '   in energy: umin=', 10, 0]
        keys['umax'] = [float, '              umax=', 10, 0]
        keys['hmin'] = [float, '   smoothing: hmin=', 10, 0]
        keys['hmax'] = [float, '              hmax=', 10, 0]

        # Here we store position information for quick lookups
        positions = {}
        specials = {}


        def __init__(self, contents, logfile):
            starsmashertools.helpers.argumentenforcer.enforcetypes({
                'contents' : [bytes],
                'logfile' : [LogFile],
            })
            
            self.logfile = logfile
            contents = contents.decode('utf-8')
            lines = contents.split('\n')

            path = self.logfile.path
            # Add this logfile to the dictionary of lookup positions
            if path not in LogFile.Iteration.positions.keys():
                LogFile.Iteration.positions[path] = {}
                # Locate all the absolute positions of the current keys
                for key, (_type, search_string, length, offset) in LogFile.Iteration.keys.items():
                    if search_string in contents:
                        start = contents.index(search_string) + len(search_string) + offset
                        stop = start + length
                        LogFile.Iteration.positions[path][key] = [_type, slice(start, stop, 1)]

                LogFile.Iteration.specials[path] = {}
                for _type, key, search_string in [[float,'dts','dts='], [int,'indx','indx']]:
                    if search_string not in contents:
                        raise LogFile.PhraseNotFoundError("'%s' in '%s'. Perhaps the log file has been cutoff?" % (search_string, path))
                    
                    start = contents.index(search_string) + len(search_string)
                    stop = start + 10
                    LogFile.Iteration.specials[path][key] = [_type, []]
                    if key == 'dts': imax = 7
                    else: imax = 6
                    for i in range(imax):
                        LogFile.Iteration.specials[path][key][1] += [slice(start, stop, 1)]
                        start += 10
                        stop += 10

            obj = {}
            for key, (_type, _slice) in LogFile.Iteration.positions[path].items():
                # This means a fortran format error
                if contents[_slice] == '*'*len(contents[_slice]):
                    obj[key] = None
                    continue
                    
                try:
                    obj[key] = _type(contents[_slice])
                except Exception as e:
                    message = "Failed to read key '%s' from log file '%s'.\nTried to read positions %d-%d and got '%s'.\nIteration contents:\n%s" % (key, path, _slice.start, _slice.stop, contents[_slice], contents)
                    raise Exception(message) from e

            for key, (_type, arr) in LogFile.Iteration.specials[path].items():
                try:
                    obj[key] = np.asarray([contents[a] for a in arr], dtype=object).astype(_type)
                except Exception as e:
                    message = "Failed to read key '%s' from log file '%s'.\nIteration contents:\n%s" % (key, path, contents)
                    raise Exception(message) from e
            
            super(LogFile.Iteration, self).__init__(obj)
