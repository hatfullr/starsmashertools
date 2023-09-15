import starsmashertools.lib.output
import starsmashertools.helpers.jsonfile
import starsmashertools.helpers.path
import warnings
import collections
import copy
import numpy as np
import starsmashertools.helpers.argumentenforcer
import starsmashertools
import starsmashertools.preferences

# Use this class to automatically save/load data you intend to plot,
# so that you don't need to recalculate data every time you make a plot.
# Give arguments in similar to the following format:
#     PlotData(
#         filenames,
#         [
#             get_x_data,
#             get_y_data,
#         ],
#     )
# where "get_x_data" and "get_y_data" are methods which accept a
# starsmashertools.lib.Output object and returns some data which will
# be accepted by whatever method you are using to plot. For example,
# if I were making a line plot with some generic x and y data, I would
# do:
#     def get_x(output):
#         return output['x']
#     def get_y(output):
#         return output['y']
#     p = PlotData(
#             ["out0000.sph"],
#             [get_x, get_y],
#             checkpointfile='xy.pdc.gz',
#     )
#     ax.plot(*p, color='k')
#     plt.show()
# This would get the x and y positions of the particles in out0000.sph
# and plot them on the axis. The first time it runs it will read
# out0000.sph and write the particle x and y data to a file called
# "xy.pdc.gz" ("pdc" stands for "Plot Data Checkpoint"). Each subsequent
# plot that is made this way will read from 'xy.pdc.gz' instead of
# out0000.sph when obtaining the plotting data.
#
# The second argument contains the methods which will be used to obtain
# data from the output files in the first argument. The results of each
# of those methods will be saved and can be used in a plotting method
# as the data arguments.

class PlotData(list, object):
    def __init__(
            self,
            filenames,
            methods,
            simulation=None,
            checkpointfile="pdc.json.gz",
            readfiles=None,
            writefile=None,
            read=True,
            write=True,
            overwrite=False,
            asynchronous=True,
            verbose=None,
            onFlush=[],
            max_buffer_size=None,
    ):
        if filenames is None and write:
            raise ValueError("Argument 'filenames' cannot be 'None' if keyword argument 'write' is 'True'")
        
        if not read and not write:
            raise ValueError("One of arguments 'read' or 'write' must be 'True', but both are 'False'")
        if readfiles is not None and not read:
            raise Exception("Keyword argument 'read' must be 'True' if keyword argument 'readfiles' is not None")
        if writefile is not None and not write:
            raise Exception("Keyword argument 'write' must be 'True' if keyword argument 'writefile' is not None")
        
        if writefile is None and write: writefile = checkpointfile
        if readfiles is None and read: readfiles = checkpointfile

        if write and starsmashertools.helpers.path.isfile(writefile) and not overwrite:
            raise Exception("Keyword argument 'overwrite' is 'False' and the writefile already exists: '%s'" % writefile)

        obj = PDCFile(writefile)

        # Read files and combine them into obj
        if read:
            if not isinstance(readfiles, str) and hasattr(readfiles, "__iter__"):
                read_obj = self.read_files(readfiles)
            elif isinstance(readfiles, str):
                read_obj = self.read(readfiles)

            if read_obj is not None: obj.combine(read_obj, overwrite=True)


        all_method_names = [m.__name__ if callable(m) else m for m in methods]
            

        # Fill in any missing information in obj
        

        
        # filenames can be None if all the user wants to do is read PDCFiles but also
        # they don't want to add any additional information to the main PDCFile
        if filenames is not None:
            # The user wants to add info to the main PDCFile
            
            if isinstance(filenames, str): filenames = [filenames]
            if not hasattr(filenames, "__iter__"):
                raise TypeError("Argument 'filenames' must be either a string or an iterable object")
            if isinstance(filenames, starsmashertools.lib.output.OutputIterator):
                raise TypeError("Argument 'filenames' cannot be type 'starsmashertools.lib.output.OutputIterator")
            if not all([isinstance(filename, str) for filename in filenames]):
                raise TypeError("All elements in argument 'filenames' must be strings")

            # Narrow the filenames to only be those which are missing.
            callable_methods = [m for m in methods if callable(m)]
            callable_method_names = [m.__name__ for m in callable_methods]
            filenames = obj.get_missing(filenames, methods=callable_method_names)

            # If there's missing files
            if filenames:
                # Create an iterator
                if write: onFlush = [obj.save] + onFlush
                iterator = starsmashertools.lib.output.OutputIterator(
                    filenames,
                    simulation,
                    asynchronous = asynchronous,
                    verbose = verbose,
                    onFlush = onFlush,
                    max_buffer_size = max_buffer_size,
                )
                for i, output in enumerate(iterator):
                    result = obj.add(output, methods, overwrite=False)
                    if result == 'break':
                        break

        # If we are to write a file, then create a copy of the current object and write it to disk.
        if write and obj.stale:
            if starsmashertools.helpers.path.isfile(writefile):
                existing_obj = PDCFile(filename = writefile)
                existing_obj.load()
                existing_obj.combine(obj, overwrite=overwrite)
                if existing_obj.stale: existing_obj.save()
                write_obj = copy.copy(existing_obj)
            else:
                write_obj = copy.copy(obj)
                write_obj.filename = writefile
                write_obj.save()
            obj = write_obj

            
        
        
        """
        # First step is to read all the given readfiles and combine them into one object
        obj = JSONObject(writefile)
        if read:
            read_obj = None
            if not isinstance(readfiles, str) and hasattr(readfiles, "__iter__"):
                read_obj = self.read_files(readfiles)
            elif isinstance(readfiles, str):
                read_obj = self.read(readfiles)
            
            if read_obj is not None: obj.combine(read_obj, overwrite=True)
        
        # Now that we have a singular object from which we can read all existing data,
        # retrieve the appropriate data from the single object

        # Figure out the method names that we are looking for in the data. string entries
        # are allowed only if we are in read-only mode (write = False).
        #method_names = self.get_method_names(methods, allow_strings=not write)

        # Get the callable method names. These are the ones we will write with if needed
        callable_methods = [m for m in methods if callable(m)]
        callable_method_names = [m.__name__ for m in callable_methods]
        
        # Build an iterator of all the filenames that are missing from the current object
        # If filenames is None then it means we are reading only, and not writing
        if filenames is not None:
            iterator = None
            if isinstance(filenames, starsmashertools.lib.output.OutputIterator):
                iterator = copy.copy(filenames)
                filenames = iterator.filenames

            missing = {}
            for filename in filenames:
                if not obj.contains(filename):
                    missing[filename] = callable_method_names
                else:
                    mnames = obj.get_method_names(filename)
                    for m in callable_method_names:
                        if m not in mnames:
                            if filename not in missing.keys():
                                missing[filename] = []
                            missing[filename] += [m]
        
            if missing:
                kwargs = {}
                if write: kwargs['onFlush'] = [obj.save]
                if iterator is not None:
                    kwargs['onFlush'] += iterator.onFlush
                    if onFlush is not None: kwargs['onFlush'] += onFlush
                    if max_buffer_size is not None:
                        kwargs['max_buffer_size'] = max_buffer_size
                    else:
                        kwargs['max_buffer_size'] = iterator.max_buffer_size
                    if verbose is None:
                        kwargs['verbose'] = iterator.verbose
                    else:
                        kwargs['verbose'] = verbose
                    kwargs['asynchronous'] = iterator.asynchronous
                else:
                    if verbose is not None: kwargs['verbose'] = verbose
                    if onFlush is not None: kwargs['onFlush'] = onFlush
                    if max_buffer_size is not None:
                        kwargs['max_buffer_size'] = max_buffer_size
                    kwargs['asynchronous'] = asynchronous
                iterator = starsmashertools.lib.output.OutputIterator(list(missing.keys()), simulation, **kwargs)

                # Step through the iterator, obtaining any data needed for missing methods
                if write:
                    broke = False
                    broke_index = None
                    for i, output in enumerate(iterator):
                        methods_to_evaluate = copy.copy(callable_methods)
                        if obj.contains(output.path):
                            for m, mname in zip(callable_methods, callable_method_names):
                                if not callable(m) or mname in obj.get_method_names(output.path):
                                    methods_to_evaluate.remove(m)
                        # Obtain the values by calling each of the methods
                        values = []
                        for m in methods_to_evaluate:
                            value = m(output)
                            if isinstance(value, str) and value == 'break':
                                broke = True
                                broke_index = i
                                break
                            values += [value]
                        if broke: break
                        obj.add(output.path, methods_to_evaluate, values)
                    if broke:
                        # Fill in the rest of the data with 'break'
                        for filename in iterator.filenames[broke_index:]:
                            obj.add(filename, methods, ['break' for i in range(len(methods))])
        
        # If we are to write a file, then create a copy of the current object and write it to disk.
        if write and obj.stale:
            if starsmashertools.helpers.path.isfile(writefile):
                existing_obj = JSONObject(writefile)
                existing_obj.load()
                existing_obj.combine(obj, overwrite=overwrite)
                if existing_obj.stale: existing_obj.save()
                write_obj = copy.copy(existing_obj)
            else:
                write_obj = copy.copy(obj)
                write_obj.filename = writefile
                write_obj.save()
            obj = write_obj
        """

        #print(obj)
        #quit()
        
        # Now convert the object to give us what we want
        #method_names = self.get_method_names(methods, allow_strings=True)
        result = self.convert(obj, method_names = all_method_names, simulation = simulation)
        super(PlotData, self).__init__(result.values())


    def convert(self, jsonobj, method_names = None, simulation = None):
        # Convert the given JSONObject to give us what we want
        if not isinstance(jsonobj, PDCFile):#JSONObject):
            raise TypeError("Argument 'jsonobj' must be of type 'PDCFile', not '%s'" % type(jsonobj).__name__)
        #print(method_names)
        if method_names is None:
            method_names = []
            for filename, obj in jsonobj.items():
                for key in obj.keys():
                    if key not in method_names: method_names += [key]

        #print(method_names)
        #quit()
                    
        result = collections.OrderedDict()
        for m in method_names:
            result[m] = []

        keys = result.keys()
        # Retrieve all the information from all simulations
        bname = None
        if simulation is not None: bname = starsmashertools.helpers.path.basename(simulation.directory)
        for filename, o in jsonobj.items():
            if simulation is not None and bname not in filename: continue
            for name, value in o.items():
                if name in keys and not (isinstance(value, str) and value == 'break'):
                    result[name] += [value]
        return result
        

    def read(self, filename):
        if starsmashertools.helpers.path.isfile(filename):
            obj = PDCFile(filename=filename) #JSONObject(filename)
            obj.load()
            return obj
        return None

    # read all the given readfiles and combine them into one object
    def read_files(self, filenames):
        objects = []
        modtimes = []
        for filename in filenames:
            if starsmashertools.helpers.path.isfile(filename):
                obj = self.read(filename)
                objects += [obj]
                modtimes += [starsmashertools.helpers.path.getmtime(obj.filename)]
        if not objects: return None
        # Combine all the objects into a single file, starting with the file which was written
        # the longest time ago. Each consecutive file overwrites that data so that the result
        # is the most up-to-date object we can get.
        modtimes = np.asarray(modtimes)
        # This sorts the times from lowest value to highest, or long-ago to recent, since the
        # time returned from mtime is time since epoch.
        idx = np.argsort(modtimes)
        final_object = PDCFile() #JSONObject()
        for i in idx:
            final_object.combine(objects[i], overwrite=True)
        return final_object

    # Given a list of methods, return the names of those methods
    def get_method_names(self, methods, allow_strings=False):
        if isinstance(methods, str): return str
        if not hasattr(methods, '__iter__'):
            if not callable(methods):
                raise TypeError("Argument 'methods' must be callable if it is not given as an iterable or str")
            return methods.__name__
        else:
            method_names = []
            for m in methods:
                if callable(m):
                    method_names += [m.__name__]
                else:
                    if isinstance(m, str):
                        if not allow_strings:
                            raise TypeError("Found a non-string member of argument 'methods' when keyword argument 'allow_strings' was 'False'")
                        
                        method_names += [m]
                    else:
                        raise TypeError("Argument 'methods' must be a list of either callable objects or objects of type 'str'. Received object of type '%s'" % type(m).__name__)
            return method_names
        
        





# A PDC file, or "Plot Data Checkpoint" file, contains information typically
# used for plotting StarSmasher data. It is an ordered dictionary whose keys
# are paths to StarSmasher output files and whose values are another
# dictionary of type PDCData.
class PDCFile(collections.OrderedDict, object):
    def __init__(self, filename=None):
        if filename is None:
            filename = starsmashertools.preferences.get_default('PDCFile', 'filename', throw_error=True)
        self.filename = starsmashertools.helpers.path.realpath(filename)
        super(PDCFile, self).__init__()
        # Used for filename comparisons
        self._simplified_filenames = {}
        self.stale = False

    @property
    def loaded(self): return hasattr(self, "_loaded") and self._loaded
        
    def __setitem__(self, key, value):
        # Ensure proper (strict) usage
        if not isinstance(key, str):
            raise KeyError("Only keys of type 'str' can be added to a PDCFile but received type '%s'" % type(key).__name__)
        if not isinstance(value, PDCData):
            raise ValueError("Only values of type PDCData can be added to a PDCFile but received type '%s'" % type(value).__name__)
        simplified = self._simplify_filename(key)
        self._simplified_filenames[simplified] = key
        
        super(PDCFile, self).__setitem__(key, value)
        # Mark that this data needs to be saved whenever one of its values are modified
        self.stale = True

    def __getitem__(self, key):
        try:
            match = self._get_matching_filename(key)
            if match is not None: key = match
        except: pass
        return super(PDCFile, self).__getitem__(key)

    def pop(self, key, *args, **kwargs):
        simplified = self._simplify_filename(key)
        if simplified in self._simplified_filenames.keys():
            key = self._simplified_filenames.pop(simplified)
        super(PDCFile, self).pop(key, *args, **kwargs)
        self.stale = True

    def _simplify_filename(self, filename):
        basename = starsmashertools.helpers.path.basename(filename)
        dirname = starsmashertools.helpers.path.basename(starsmashertools.helpers.path.dirname(filename))
        return starsmashertools.helpers.path.join(dirname, basename)

    def _get_matching_filename(self, filename):
        simplified = self._simplify_filename(filename)
        if simplified in self._simplified_filenames.keys():
            return self._simplified_filenames[simplified]

    def contains(self, filename):
        return self._get_matching_filename(filename) is not None

    # You can only add an Output object to PDCFiles. The methods in
    # 'methods' will each be called using 'output' as input.
    def add(self, output, methods, overwrite=False):
        if not isinstance(output, starsmashertools.lib.output.Output):
            raise TypeError("Argument 'output' must be of type 'starsmashertools.lib.output.Output', not '%s'" % type(output).__name__)
        if not hasattr(methods, '__iter__') or isinstance(methods, str):
            raise ValueError("Argument 'methods' must be a non-str iterable")
        
        for method in methods:
            if not callable(method):
                raise ValueError("Method is not callable: '%s'" % str(method))
        
        if self.contains(output.path):
            if not overwrite:
                self._append(output, methods)
                return
        
        values = [method(output) for method in methods]
        self[output.path] = PDCData(methods, values)

    # When the given output object is already included in a PDCFile, this function
    # appends output from methods in 'methods' that are missing in the PDCFile.
    def _append(self, output, methods):
        # Save time by not checking the inputs
        #if not self.contains(output.path):
        #    raise KeyError("No key matching '%s'" % output.path)
        #if not hasattr(methods, "__iter__") or isinstance(methods, str):
        #    raise TypeError("Argument 'methods' must be an iterable, non-string object")
        item = self[output.path]
        _methods = item.keys()
        for method in methods:
            #if not callable(method):
            #    raise ValueError("Element in 'methods' is not callable: '%s'" % str(method))
            if method.__name__ not in _methods:
                item[method.__name__] = method(output)
        self[output.path] = item

    # Remove the given filenames from the dictionary
    def remove_filenames(self, filenames):
        for filename in filenames:
            self.pop(filename, None)

    # Remove the given methods from all filenames in the dictionary
    def remove_methods(self, methods):
        for key, data in self.items():
            for method in methods:
                data.pop(method, None)
    
    def load(self):
        if not starsmashertools.helpers.path.isfile(self.filename):
            raise FileNotFoundError(self.filename)
        obj = starsmashertools.helpers.jsonfile.load(self.filename)
        for filename, data in obj.items():
            self[filename] = PDCData(data.keys(), data.values())
        self.stale = False
        self._loaded = True

    def save(self, overwrite=False):
        if starsmashertools.helpers.path.isfile(self.filename):
            if overwrite:
                warnings.warn("File '%s' exists and will be appended/overwritten" % self.filename, stacklevel=2)
            else:
                raise Exception("File '%s' exists but keyword 'overwrite' is False" % self.filename)
        starsmashertools.helpers.jsonfile.save(self.filename, self)
        self.stale = False

    # Combine the contents of this object and other. If 'overwrite' is True then the
    # contents of 'other' will supercede the contents of this object.
    def combine(self, other, overwrite=False):
        if not isinstance(other, PDCFile):
            raise TypeError("Argument 'other' must be of type PDCFile, not '%s'" % type(other).__name__)
        
        if not self.loaded: self.load()
        if not other.loaded: other.load()
        
        for filename, data in other.items():
            if self.contains(filename):
                self[filename].combine(data, overwrite=overwrite)
            else: self[filename] = data

    # Return a list of all the methods in the PDCFile
    def get_methods(self):
        methods = []
        for data in self.values():
            for key in data.keys():
                if key not in methods: methods += [key]
        return methods
        

    # Given a list of files, return a list of filenames that either don't exist
    # in this PDCFile or have missing methods. If keyword 'methods' is None then
    # we will check for consistency in the methods, such that a file is deemed
    # "missing" if it is missing a method from the list of all methods. If keyword
    # 'methods' is not None, then we will only check for files that don't have
    # one of the methods given.
    def get_missing(self, filenames, methods=None):
        if not hasattr(filenames, "__iter__") or isinstance(filenames, str):
            filenames = [filenames]
        
        if methods is None: methods = self.get_methods()
        
        missing = []
        keys = self.keys()
        for filename in filenames:
            if not self.contains(filename):
                missing += [filename]
                continue
            _methods = self[filename].keys()
            for m in methods:
                if m not in _methods:
                    missing += [filename]
                    break
        return missing
                
            
    
        

# A data structure used by the PDCFile class. The keys of the dictionary are
# callable method names and the values are the actual values of any builtin
# (picklable) type.
class PDCData(collections.OrderedDict, object):
    allowed_types = [
        type(None),
        bool,
        float,
        int,
        str,
    ]
    
    def __init__(self, methods, values):
        super(PDCData, self).__init__()
        methods = PDCData.methods_to_str(methods)
        for method, value in zip(methods, values):
            self[method] = value

    def __copy__(self):
        return self.__deepcopy__()

    def __deepcopy__(self, *args, **kwargs):
        result = self.__class__.__new__(self.__class__)
        result.__dict__.update(self.__dict__)
        for key, val in self.items(): result[key] = val
        return result

    @staticmethod
    def from_json(pdcdata):
        return PDCData(pdcdata.keys(), pdcdata.values())

    @staticmethod
    def methods_to_str(methods):
        wasiter = False
        if not hasattr(methods, '__iter__') or isinstance(methods, str):
            methods = [methods]
            wasiter = True
        for i, method in enumerate(methods):
            if not isinstance(method, str) and callable(method):
                methods[i] = method.__name__
        if wasiter: return methods[0]
        return methods

    def convert_type(self, value):
        if hasattr(value, '__iter__') and not isinstance(value, str):
            return [self.convert_type(v) for v in value]

        # Convert numpy types
        if type(value).__module__ == np.__name__:
            value = value.item()
        
        if not isinstance(value, tuple(PDCData.allowed_types)):
            raise ValueError("Values in PDCData objects must be one of types "+", ".join(["'%s'" % type(t).__name__ for t in PDCData.allowed_types])+", not '%s'" % type(value).__name__)

        return value
        
    
    def __setitem__(self, key, value):
        # Ensure proper (strict) usage
        key = PDCData.methods_to_str(key)
        if not isinstance(key, str):
            raise KeyError("Keys for PDCData objects must be type 'str', not type '%s'" % type(key).__name__)
        value = self.convert_type(value)
        return super(PDCData, self).__setitem__(key, value)

    def contains(self, method):
        return method in self.keys()

    # Combine the contents of this object and other. If 'overwrite' is True then the
    # contents of 'other' will supercede the contents of this object.
    def combine(self, other, overwrite=False):
        if not isinstance(other, PDCData):
            raise TypeError("Argument 'other' must be of type PDCData, not '%s'" % type(other).__name__)
        methods, values = self.keys(), self.values()
        for method, value in other.items():
            if value is None: continue

            # If we already have this method, the we shouldn't change our value unless
            # the value is None or if overwrite is True.
            if method in methods:
                if self[method] is not None and not overwrite: continue
            self[method] = value
                

# An object that contains easy-to-understand logic about how the data is stored in
# the json files
class JSONObject(collections.OrderedDict, object):
    def __init__(self, filename=None):
        if filename is not None:
            filename = starsmashertools.helpers.path.realpath(filename)

        self.filename = filename
        super(JSONObject, self).__init__()
        self.stale = False
        self._key_basenames = {}

    def __setitem__(self, *args, **kwargs):
        self.stale = True
        return super(JSONObject, self).__setitem__(*args, **kwargs)

    def __getitem__(self, item):
        # Get item by simulation directory basename
        try:
            match = self._get_matching_item(item)
            if match is not None: item = match
        except:
            pass
        return super(JSONObject, self).__getitem__(item)

    def _get_simdir(self, path):
        return starsmashertools.helpers.path.basename(starsmashertools.helpers.path.dirname(path))

    # Use the "key_basenames" to find matching keys, where a "key_basename" is, e.g. for a filename like
    # '/here/stuff/things/simulation/out0000.sph', the "key_basename" is 'simulation/out0000.sph'. If
    # there are multiple simulations with the same directory name, i.e. 'simulation', then this method
    # won't work. If you want to extend this method, consider implementing some way to search back
    # through the simulation path. That is, if there is another file located at
    # '/here/there/things/simulation/out0000.sph', then the "key_basenames" should be set as
    # 'stuff/things/simulation/out0000.sph' and 'there/things/simulation/out0000.sph'.
    def _get_matching_item(self, filename):
        if filename in self.keys(): return filename
        key_basename = self._get_key_basename(filename)
        if key_basename in self._key_basenames.keys():
            return self._key_basenames[key_basename]
        return None

    def _get_key_basename(self, filename):
        basename = starsmashertools.helpers.path.basename(filename)
        dirname = starsmashertools.helpers.path.basename(starsmashertools.helpers.path.dirname(filename))
        return starsmashertools.helpers.path.join(dirname, basename)

    def _add_key_basename(self, filename):
        key_basename = self._get_key_basename(filename)
        if key_basename not in self._key_basenames.keys():
            self._key_basenames[key_basename] = filename

    def contains(self, filename):
        return self._get_matching_item(filename) is not None

    def get_method_names(self, filename):
        if not self.contains(filename): return []
        return list(self[filename].keys())

    def get_all_method_names(self):
        method_names = []
        for obj in self.values():
            for method_name in obj.keys():
                if method_name not in method_names: method_names += [method_name]
        return method_names
    
    def get_missing(self, filenames):
        keys = list(self.keys())
        missing = []
        notmissing = []
        for filename in filenames:
            if filename in keys: notmissing += [filename]
            else: missing += [filename]
        if len(notmissing) == 0: return missing
            
        # Obtain all the method names
        all_method_names = self.get_all_method_names()

        actually_missing = copy.copy(missing)
        actually_notmissing = copy.copy(notmissing)
        for filename in notmissing:
            methods = self.get_method_names(filename)
            for method in all_method_names:
                if method not in methods:
                    actually_notmissing.remove(filename)
                    actually_missing += [filename]
                    break

        return actually_missing
    
            
    """    
        method_names = []
        for filename in filenames:
            for m in self.get_method_names(filename):
                if m not in method_names: method_names += [m]
        
        missing = []
        for filename in filenames:
            if not self.contains(filename):
                missing += [filename]
            else:
                mnames = self.get_method_names(filename)
                for m in method_names:
                    if m not in mnames:
                        missing += [filename]
                        break
        return missing
    """

    def add(self, filename, methods, values):
        names = self.get_method_names(filename)
        if names:
            for m, v in zip(methods, values):
                name = m.__name__
                if name not in names:
                    self[filename][name] = v
        else:
            self[filename] = {m.__name__ : v for m,v in zip(methods, values)}
        self._add_key_basename(filename)

    def save(self):
        if self.filename is None:
            raise Exception("Cannot save a JSONObject whose filename is 'None'")
        if starsmashertools.helpers.path.isfile(self.filename):
            warnings.warn("File '%s' exists and will be appended/overwritten." % self.filename, stacklevel=2)
            
        starsmashertools.helpers.jsonfile.save(self.filename, self)
        #savename = self.filename+".temp"
        #starsmashertools.helpers.jsonfile.save(savename, self)
        #starsmashertools.helpers.path.rename(savename, self.filename)
        self.stale = False

    def load(self):
        if self.filename is None:
            raise Exception("Cannot load a JSONObject whose filename is 'None'")
        if not starsmashertools.helpers.path.isfile(self.filename):
            raise FileNotFoundError(self.filename)
        obj = starsmashertools.helpers.jsonfile.load(self.filename)
        for filename, val in obj.items():
            if self.contains(filename):
                for name, value in val.items():
                    self[filename][name] = value
            else:
                self[filename] = val
            self._add_key_basename(filename)
        self.stale = False

    # Combine the contents of this object and other. If 'overwrite' is True then the
    # contents of 'other' will supercede the contents of this object.
    def combine(self, other, overwrite=False):
        if not isinstance(other, JSONObject):
            raise ValueError("Argument 'other' must be of type 'PlotData', not '%s'" % type(other).__name__)
        
        filenames = self.keys()
        
        for filename, val in other.items():
            if self.contains(filename):
                keys = self[filename].keys()
                for name, value in val.items():
                    if name in keys:
                        if overwrite:
                            self[filename][name] = value
                    else:
                        self[filename][name] = value
            else:
                self[filename] = val








@starsmashertools.helpers.argumentenforcer.enforcetypes
def make_pdc(
        filename : str,
        simulations : list,
        methods : list,
        parallel : bool = False,
        # The following keywords are manually set in the below code
        simulation = None,
        read = True,
        write = True,
        # Keywords to pass to the PlotData class
        **kwargs
):
    # Ensure correct types in input
    for i, simulation in enumerate(simulations):
        if isinstance(simulation, str):
            simulations[i] = starsmashertools.get_simulation(simulation)
        elif (not isinstance(simulation, starsmashertools.lib.simulation.Simulation) and
              not issubclass(simulation, starsmashertools.lib.simulation.Simulation)):
            raise TypeError("A simulation in the input list is type '%s', but must be either 'str' or 'starsmashertools.lib.simulation.Simulation'" % type(simulation).__name__)

    for method in methods:
        if not callable(method):
            raise TypeError("A method in the input list is not callable: '%s'" % str(method))

    if 'writefile' in kwargs.keys():
        raise KeyError("Invalid keyword 'writefile'")
    if 'checkpointfile' in kwargs.keys():
        raise KeyError("Invalid keyword 'checkpointfile'")
    
    kwargs['checkpointfile'] = filename

    def do_serial(filenames, simulation, **kwargs):
        starsmashertools.mpl.plotdata.PlotData(
            filenames,
            methods,
            simulation=simulation,
            read=True,
            write=True,
            **kwargs
        )
        return kwargs.get('writefile', kwargs.get('checkpointfile', None))
    
    def do_parallel(filenames, simulation, **kwargs):
        pool = multiprocessing.Pool()
        nprocs = pool._processes
        procs = [None]*nprocs

        chunks = np.array_split(filenames, nprocs)
        for p in range(nprocs):
            kwargs['writefile'] = 'pdc_thread%d.json.gz' % p
            procs[p] = pool.apply_async(
                do_serial,
                args = (
                    chunks[p],
                    simulation,
                ),
                kw = kwargs,
            )
        writefiles = []
        for p in range(nprocs):
            writefiles += [procs[p].get()]
            
        pool.close()
        pool.join()

        # Combine the PDC files
        obj = JSONObject(filename)
        for writefile in writefiles:
            new_obj = JSONObject(writefile)
            new_obj.load()
            obj.combine(new_obj)

        # Save the PDC
        obj.save()

        # Delete the threads' PDC files
        for writefile in writefiles:
            os.remove(writefile)
    
    obj = JSONObject(filename)
    
    for simulation in simulations:
        # Collect all the output files that are missing from the PDCs
        outputfiles = obj.get_missing(simulation.get_outputfiles())
        if len(outputfiles) == 0: continue
        
        outputfiles = np.asarray(outputfiles, dtype=str)

        if not parallel: do_serial(outputfiles, simulation, **kwargs)
        else: do_parallel(outputfiles, simulation, **kwargs)
            
