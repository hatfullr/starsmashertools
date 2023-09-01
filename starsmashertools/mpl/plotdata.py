import starsmashertools.lib.output
import starsmashertools.helpers.jsonfile
import starsmashertools.helpers.path
import warnings
import collections
import copy
import numpy as np

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
                    kwargs['max_buffer_size'] = iterator.max_buffer_size
                    kwargs['verbose'] = iterator.verbose
                    kwargs['asynchronous'] = iterator.asynchronous
                else:
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
                existing_obj = JSONObject(writefile).load()
                existing_obj.combine(obj, overwrite=overwrite)
                if existing_obj.stale: existing_obj.save()
                write_obj = copy.copy(existing_obj)
            else:
                write_obj = copy.copy(obj)
                write_obj.filename = writefile
                write_obj.save()
            obj = write_obj
        

        # Now convert the object to give us what we want
        method_names = self.get_method_names(methods, allow_strings=True)
        result = self.convert(obj, method_names = method_names, simulation = simulation)
        
        super(PlotData, self).__init__(result.values())


    def convert(self, jsonobj, method_names = None, simulation = None):
        # Convert the given JSONObject to give us what we want
        if not isinstance(jsonobj, JSONObject):
            raise TypeError("Argument 'jsonobj' must be of type 'JSONObject', not '%s'" % type(jsonobj).__name__)
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
            obj = JSONObject(filename)
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
        final_object = JSONObject()
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

    def get_missing(self, filenames):
        # Obtain all the method names available
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
