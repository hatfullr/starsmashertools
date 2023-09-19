import starsmashertools.preferences as preferences
import starsmashertools.helpers.path
import starsmashertools.helpers.file
import numpy as np
import time
import multiprocessing
import starsmashertools.lib.simulation
import starsmashertools.helpers.multiprocessing
import collections
import mmap
import starsmashertools.helpers.stacktrace
import copy as _copy

class Output(dict, object):
    modes = [
        'raw',
        'cgs',
    ]
    
    def __init__(self, path, simulation, mode='raw'):
        if mode not in Output.modes:
            s = ", ".join(["'"+str(m)+"'" for m in Output.modes])
            raise ValueError("Keyword argument 'mode' must be one of %s, not '%s'" % (s, str(mode)))
        self.path = starsmashertools.helpers.path.realpath(path)
        self.simulation = simulation
        self.mode = mode
        self._isRead = {
            'header' : False,
            'data' : False,
        }
        super(Output, self).__init__()

        self._cache = _copy.copy(preferences.get_default('Output', 'cache'))
        # If no cache is defined in preferences
        if self._cache is None: self._cache = {}

        self._mask = None
    
    def __str__(self): return "Output('%s')" % starsmashertools.helpers.path.basename(self.path)

    def __repr__(self): return str(self)

    # Use this to make sure that the file has been fully read
    def _ensure_read(self):
        if False in self._isRead.values():
            self.read(return_headers=not self._isRead['header'], return_data=not self._isRead['data'])

    def __eq__(self, other):
        return starsmashertools.helpers.file.compare(self.path, other.path)

    def __getitem__(self, item):
        if item in self._cache.keys(): return self.from_cache(item)
        if item not in self.keys(ensure_read=False):
            self._ensure_read()
        ret = super(Output, self).__getitem__(item)

        if isinstance(ret, np.ndarray):
            # Try to mask the output data if we have a mask
            if self._mask is not None:
                if not isinstance(self._mask, np.ndarray):
                    raise TypeError("Property 'Output._mask' must be of type 'np.ndarray', not '%s'" % type(self._mask).__name__)
                try:
                    ret = ret[self._mask]
                except IndexError as e:
                    if 'boolean index did not match indexed array' in str(e):
                        pass
        
        if self.mode == 'cgs':
            if item in self.simulation.units.keys():
                ret = _copy.copy(ret) * self.simulation.units[item]
            elif hasattr(self.simulation.units, item):
                ret = _copy.copy(ret) * getattr(self.simulation.units, item)
        return ret

    def keys(self, *args, **kwargs):
        if 'ensure_read' in kwargs.keys():
            if kwargs.pop('ensure_read'): self._ensure_read()
        else: self._ensure_read()
        ret = super(Output, self).keys(*args, **kwargs)
        obj = {key:None for key in ret}
        for key in self._cache.keys(): obj[key] = None
        return obj.keys()

    def values(self, *args, **kwargs):
        if 'ensure_read' in kwargs.keys():
            if kwargs.pop('ensure_read'): self._ensure_read()
        else: self._ensure_read()
        keys = self._cache.keys()
        for key in self.keys():
            if key in keys:
                self[key] = self.from_cache(key)
        return super(Output, self).values(*args, **kwargs)

    def items(self, *args, **kwargs):
        if 'ensure_read' in kwargs.keys():
            if kwargs.pop('ensure_read'): self._ensure_read()
        else: self._ensure_read()
        keys = self.keys()
        for key in self._cache.keys():
            if key not in keys:
                self[key] = self.from_cache(key)
        return super(Output, self).items(*args, **kwargs)

    def copy_from(self, obj):
        for key in self._isRead.keys():
            self._isRead[key] = True
        
        for key, val in obj.items():
            self[key] = val
    
    def read(self, return_headers=True, return_data=True, **kwargs):
        if self._isRead['header']: return_headers = False
        if self._isRead['data']: return_data = False
        
        obj = self.simulation.reader.read(
            self.path,
            return_headers=return_headers,
            return_data=return_data,
            **kwargs
        )
        
        header, data = None, None
        if return_headers and return_data:
            header, data = obj
        elif not return_headers and return_data:
            data = obj
        elif return_headers and not return_data:
            header = obj

        if header is not None:
            for key, val in header.items():
                self[key] = val
        
        if data is not None:
            for key, val in data.items():
                self[key] = val

        if return_headers: self._isRead['header'] = True
        if return_data: self._isRead['data'] = True

    def from_cache(self, key):
        if key in self._cache.keys():
            if callable(self._cache[key]):
                self._cache[key] = self._cache[key](self)
        return self._cache[key]

    def mask(self, mask):
        self._mask = mask

    def unmask(self):
        self._mask = None

        


# Asynchronous output file reading
class OutputIterator(object):
    def __init__(self, filenames, simulation, onFlush=[], max_buffer_size=None, verbose=False, asynchronous=True):
        if max_buffer_size is None: max_buffer_size = preferences.get_default('OutputIterator', 'max buffer size')
        self.max_buffer_size = max_buffer_size
        self.onFlush = onFlush
        self.simulation = simulation
        self.asynchronous = asynchronous

        for m in self.onFlush:
            if not callable(m):
                raise TypeError("Callbacks in keyword 'onFlush' must be callable, but received '%s'" % str(m))

        # Make sure that the filenames are the true paths
        self.filenames = filenames
        self.verbose = verbose
        self._break = False

        self._process = None

        self._index = -1
        self._buffer_index = -1

    def __str__(self):
        if len(self.filenames) == 0: return "OutputIterator()"
        bname1 = starsmashertools.helpers.path.basename(self.filenames[0])
        if len(self.filenames) == 1: return "OutputIterator('%s')" % bname1
        bname2 = starsmashertools.helpers.path.basename(self.filenames[-1])
        return "OutputIterator('%s' ... '%s')" % (bname1, bname2)

    def __repr__(self): return str(self)

    def __contains__(self, item):
        if isinstance(item, Output):
            for filename in self.filenames:
                if starsmashertools.helpers.file.compare(item.path, filename):
                    return True
        elif isinstance(item, str):
            if not starsmashertools.helpers.path.isfile(item): return False
            for filename in self.filenames:
                if starsmashertools.helpers.file.compare(item, filename):
                    return True
        return False

    def __len__(self): return len(self.filenames)

    def __iter__(self): return self

    def __next__(self):
        if len(self.filenames) == 0: self.stop()
        
        self._buffer_index += 1

        if self._buffer_index >= self.max_buffer_size:
            # Wait for the process to be finished before continuing
            if self.asynchronous:
                if self._process is not None and self._process.is_alive(): self._process.join()
                self._process = starsmashertools.helpers.multiprocessing.Process(
                    target=self.call_flush_methods,
                    daemon=True,
                )
                self._process.start()
            else:
                self.call_flush_methods()
            self._buffer_index = -1

        if self._break: self.stop()

        self._index += 1
        if self._index < len(self.filenames):
            return self.get(self.filenames[self._index])
        else:
            self.stop()


    def next(self, *args, **kwargs): return self.__next__(self, *args, **kwargs)

    def stop(self):
        # Ensure that we do the flush methods whenever we stop iterating
        self.call_flush_methods()
        raise StopIteration

    def call_flush_methods(self):
        for m in self.onFlush:
            r = m()
            if isinstance(r, str) and r == 'break':
                self._break = True
                break
    
    def get(self, filename):
        o = Output(filename, self.simulation)
        o.read(verbose=self.verbose)
        return o









class Reader(object):
    formats = {
        'integer' : 'i4',
        'real*8'  : 'f8',
        'real'    : 'f4',
    }

    EOL = 'f8'
    
    def __init__(self, simulation):
        header_format, header_names, data_format, data_names = Reader.get_output_format(simulation)
        
        self._dtype = {
            'data' : np.dtype([(name, fmt) for name, fmt in zip(data_names, data_format.split(","))]),
            'header' : np.dtype([(name, fmt) for name, fmt in zip(header_names, header_format.split(","))]),
        }

        self._stride = {
            'header' : sum([header_format.count(str(num))*num for num in [1, 2, 4, 6, 8]]),
            'data' : sum([data_format.count(str(num))*num for num in [1, 2, 4, 6, 8]]),
        }

        self._EOL_size = sum([Reader.EOL.count(str(num))*num for num in [1, 2, 4, 6, 8]])


    def read(self, filename, return_headers=True, return_data=True, verbose=False):
        if verbose: print(filename)

        if True not in [return_headers, return_data]:
            raise ValueError("One of 'return_headers' or 'return_data' must be True")

        def do(*args, **kwargs):
            exc = None
            try:
                ret = np.ndarray(*args, **kwargs)
                if ret.dtype.names is not None:
                    for name in ret.dtype.names:
                        if np.any(np.abs(ret[name]) > 1e100):
                            raise Reader.UnexpectedFileFormatError
                else:
                    if np.abs(ret) < 1e-100 or np.abs(ret) > 1e100:
                        raise Reader.UnexpectedFileFormatError
            except Exception as e:
                f.close()
                raise Exception("This Output might have been written by a different simulation. Make sure you use the correct simulation when creating an Output object, as different simulation directories have different reading and writing methods in their source directories.") from e
                
            return ret


        f = starsmashertools.helpers.file.open(filename, 'rb')

        # This speeds up reading significantly.
        buffer = mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)

        header = do(
            buffer=buffer,
            shape=1,
            dtype=self._dtype['header'],
            offset=4,
            strides=self._stride['header'],
        )

        ntot = header['ntot'][0]

        # Check for corrupted files

        filesize = starsmashertools.helpers.file.getsize(filename)
        
        ntot_check = do(
            buffer=buffer,
            shape=1,
            dtype='<i4',
            offset=filesize - 8,
            strides=8,
        )[0]
        if ntot != ntot_check:
            f.close()
            raise Reader.CorruptedFileError(filename)

        if return_headers and not return_data:
            f.close()
            return header

        # There are 'ntot' particles to read
        data = do(
            buffer=buffer,
            shape=ntot,
            dtype=self._dtype['data'],
            offset=self._stride['header'] + self._EOL_size + 4,
            strides=self._stride['data'] + self._EOL_size,
        )
        
        f.close()

        # Now organize the data into Pythonic structures
        new_data = {}
        for name in data[0].dtype.names:
            new_data[name] = np.array(data[name])
        new_header = {}
        for name in header[0].dtype.names:
            new_header[name] = np.array(header[name])[0]
        if return_headers: return new_data, new_header
        else: return new_data
            
        
    # This method returns instructions on how to read the data files
    @staticmethod
    def get_output_format(simulation):
        # Expected data types
        data_types = ('integer', 'real', 'logical', 'character')
        
        src = starsmashertools.helpers.path.get_src(simulation.directory)

        if src is None:
            raise Reader.SourceDirectoryNotFoundError(simulation)
        
        # Find the output.f file
        writerfile = starsmashertools.helpers.path.join(src, "output.f")
        
        listening = False
        
        # Read the output.f file
        subroutine_text = ""
        f = starsmashertools.helpers.file.open(writerfile, 'r')
        for line in f:
            if len(line.strip()) == 0 or line[0] in starsmashertools.helpers.file.fortran_comment_characters:
                continue
            if '!' in line: line = line[:line.index('!')] + "\n"
            if len(line.strip()) == 0: continue
            
            ls = line.strip()
            
            if ls == 'subroutine dump(iu)':
                listening = True

            if listening:
                subroutine_text += line
                if ls == 'end': break
        f.close()

        # Use the subroutine text to get the variable names and their types
        vtypes = {}
        for line in subroutine_text.split("\n"):
            if len(line.strip()) == 0 or (line[0] in starsmashertools.helpers.file.fortran_comment_characters):
                continue
            if '!' in line: line = line[:line.index('!')] + "\n"
            if len(line.strip()) == 0: continue
            
            ls = line.strip()
            if len(ls) > len('include') and ls[:len('include')] == 'include':
                # On 'include' lines, we find the file that is being included
                dname = starsmashertools.helpers.path.dirname(writerfile)
                fname = starsmashertools.helpers.path.join(dname, ls.replace('include','').replace('"','').replace("'", '').strip())
                f = starsmashertools.helpers.file.open(fname, 'r')
                for key, val in starsmashertools.helpers.string.get_fortran_variable_types(f.read(), data_types).items():
                    if key not in vtypes.keys(): vtypes[key] = val
                    else: vtypes[key] += val
                
                f.close()
        
        for key, val in starsmashertools.helpers.string.get_fortran_variable_types(subroutine_text, data_types).items():
            if key not in vtypes.keys(): vtypes[key] = val
            else: vtypes[key] += val

        
        # We have all the variables and their types now, so let's check which variables
        # are written to the output file header
        header_names = []
        listening = False
        for line in subroutine_text.split("\n"):
            if len(line.strip()) == 0 or line[0] in starsmashertools.helpers.file.fortran_comment_characters:
                continue
            if '!' in line: line = line[:line.index('!')] + "\n"
            if len(line.strip()) == 0: continue
            
            ls = line.strip().replace(' ','')
            if 'write(iu)' in ls:
                ls = ls[len('write(iu)'):]
                if ls[-1] == ',': ls = ls[:-1]
                header_names += ls.split(",")
                listening = True
                continue

            if listening:
                # When it isn't a line continuation, we stop
                # https://gcc.gnu.org/onlinedocs/gcc-3.4.6/g77/Continuation-Line.html
                isContinuation = line[5] not in [' ', '0']
                if not isContinuation: break
                l = line[6:].strip()
                if l[-1] == ',': l = l[:-1]
                header_names += l.split(",")

        
        # We know what the header is now, so we get the data types associated with it
        header_formats = collections.OrderedDict()
        for name in header_names:
            for _type, names in vtypes.items():
                if name in names:
                    if _type not in Reader.formats:
                        raise NotImplementedError("Fortran type '%s' is not included in Reader.formats in output.py but is found in '%s'" % (_type, writerfile))
                    header_formats[name] = Reader.formats[_type]
                    break


        # To save us from actually interpreting the Fortran code, we make
        # assumptions about the format of output.f.
        if 'if(ncooling.eq.0)' not in subroutine_text.replace(' ',''):
            raise Exception("Unexpected format in '%s'" % writerfile)

        data_names = []
        listening1 = False
        listening2 = False
        for line in subroutine_text.split("\n"):
            if len(line.strip()) == 0 or line[0] in starsmashertools.helpers.file.fortran_comment_characters: continue
            if '!' in line: line = line[:line.index('!')] + "\n"
            if len(line.strip()) == 0: continue
            
            ls = line.strip().replace(' ','')
            if not listening1:
                if 'if(ncooling.eq.' in ls:
                    idx = ls.index('if(ncooling.eq.') + len('if(ncooling.eq.')
                    if simulation['ncooling'] == int(ls[idx]):
                        listening1 = True
                        continue
            elif not listening2:
                if len(ls) > len('write(iu)') and ls[:len('write(iu)')] == 'write(iu)':
                    listening2 = True
                    l = ls[len('write(iu)'):]
                    if l[-1] == ',': l = l[:-1]
                    data_names = l.split(',')
            else:
                # Each line should be either a line continuation or a comment line, otherwise stop
                # https://gcc.gnu.org/onlinedocs/gcc-3.4.6/g77/Continuation-Line.html
                isContinuation = line[5] not in [' ', '0']
                if not isContinuation: break
                
                line = line[6:]
                ls = line.strip().replace(' ','')
                if ls[-1] == ',': ls = ls[:-1]
                data_names += ls.split(',')

        for i, name in enumerate(data_names):
            if '(' in name:
                data_names[i] = name[:name.index('(')] 


        # With all the data names known, we obtain their types
        data_formats = collections.OrderedDict()
        for name in data_names:
            for _type, names in vtypes.items():
                if name in names:
                    if _type not in Reader.formats:
                        raise NotImplementedError("Fortran type '%s' is not included in Reader.formats in output.py but is found in '%s'" % (_type, writerfile))
                    data_formats[name] = Reader.formats[_type]
                    break

        header_format = '<' + ','.join(header_formats.values())
        header_names = list(header_formats.keys())
        data_format = '<' + ','.join(data_formats.values())
        data_names = list(data_formats.keys())
        return header_format, header_names, data_format, data_names


    class CorruptedFileError(Exception):
        def __init__(self, message):
            super(Reader.CorruptedFileError, self).__init__(message)

    class UnexpectedFileFormatError(Exception):
        def __init__(self, message=""):
            super(Reader.UnexpectedFileFormatError, self).__init__(message)

    class SourceDirectoryNotFoundError(Exception):
        def __init__(self, simulation, message=""):
            if not message:
                message = "Failed to find the code source directory in simulation '%s'" % simulation.directory
            super(Reader.SourceDirectoryNotFoundError, self).__init__(message)










        
