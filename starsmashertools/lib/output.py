import starsmashertools.preferences
from starsmashertools.preferences import Pref
import starsmashertools.helpers.argumentenforcer
from starsmashertools.helpers.apidecorator import api
import numpy as np
import typing
import copy
import collections
import itertools
import time
import mmap
import re
import itertools
import struct
import warnings

try:
    import matplotlib.axes
    has_matplotlib = True
except ImportError:
    has_matplotlib = False

@starsmashertools.preferences.use
class Output(dict, object):
    r"""
    A container for StarSmasher binary output data, usually appearing as
    ``out*.sph`` in simulation directories.
    """
    
    modes = [
        'raw',
        'cgs',
    ]
    @api
    def __init__(
            self,
            path,
            simulation,
            mode='raw',
    ):
        r"""
        Parameters
        ----------
        path : str
            The file path to a ``StarSmasher`` binary output data file.
        
        simulation : :class:`~.simulation.Simulation`
            The :class:`~.simulation.Simulation` object that this output file 
            belongs to.
        
        Other Parameters
        ----------------
        mode : str, default = 'raw'
            Affects the units of the data. Deprecated.
        """
        import starsmashertools.helpers.string
        
        if mode not in Output.modes:
            s = starsmashertools.helpers.string.list_to_string(Output.modes, join='or')
            raise ValueError("Keyword argument 'mode' must be one of %s, not '%s'" % (s, str(mode)))
        self._path = path
        self.simulation = simulation
        self.mode = mode
        self._isRead = {
            'header' : False,
            'data' : False,
        }
        super(Output, self).__init__()

        self._cache = None
        self._clear_cache()

        self._mask = None
        self._header = None
        self.data = None

        self._original_data = None

    @property
    def path(self):
        import starsmashertools.helpers.path
        return starsmashertools.helpers.path.realpath(self._path)

    @staticmethod
    def _get_path_and_simulation(obj):
        import starsmashertools.helpers.path
        import starsmashertools
        simulation = starsmashertools.get_simulation(obj['simulation'])
        path = starsmashertools.helpers.path.join(
            obj['path'],
            simulation.directory,
        )
        return path, simulation

    def _get_relpath(self):
        import starsmashertools.helpers.path
        return starsmashertools.helpers.path.relpath(
            self.path,
            start = self.simulation.directory,
        )

    #"""
    # Used when pickling
    def __getstate__(self):
        import starsmashertools.helpers.path
        return {
            'path' : self._path,
            'simulation' : self.simulation.directory,
            'mode' : self.mode,
            'mask' : self._mask,
        }
    
    # Used when pickling
    def __setstate__(self, data):
        import starsmashertools
        self.__init__(
            data['path'],
            starsmashertools.get_simulation(data['simulation']),
            mode = data['mode'],
        )
        if data['mask'] is not None:
            self.mask(data['mask'])
    #"""
    
    def __str__(self):
        import starsmashertools.helpers.path
        string = self.__class__.__name__ + "(%s)"
        return string % ("'%s'" % starsmashertools.helpers.path.basename(self._path))

    def __repr__(self): return str(self)

    @api
    def __getitem__(self, item):
        """
        If the file has not yet been read, read it. Then return the requested
        item.
        """
        if item in self.keys(ensure_read=False):
            if item in self._cache.keys():
                return self.from_cache(item)
        else:
            self._ensure_read()
        ret = super(Output, self).__getitem__(item)
        
        if self.mode == 'cgs':
            if item in self.simulation.units.keys():
                ret = copy.copy(ret) * self.simulation.units[item]
            elif hasattr(self.simulation.units, item):
                ret = copy.copy(ret) * getattr(self.simulation.units, item)
        return ret

    def __eq__(self, other):
        import starsmashertools.helpers.file
        import starsmashertools.helpers.path
        if isinstance(other, str):
            if starsmashertools.helpers.path.exists(other):
                return starsmashertools.helpers.file.compare(self.path, other)
            return False
        if not isinstance(other, Output): return False
        return starsmashertools.helpers.file.compare(self.path, other.path)

    def __hash__(self, *args, **kwargs):
        import starsmashertools.helpers.path
        return self.path.__hash__(*args, **kwargs)

    def __copy__(self,*args,**kwargs):
        ret = Output(self._path, self.simulation, mode=self.mode)
        ret.copy_from(self)
        return ret
    def __deepcopy__(self,*args,**kwargs):
        return self.__copy__(*args, **kwargs)

    # Use this to make sure that the file has been fully read
    def _ensure_read(self):
        if False in self._isRead.values():
            self.read(return_headers=not self._isRead['header'], return_data=not self._isRead['data'])

    def _clear_cache(self):
        self._cache = copy.deepcopy(self.preferences.get('cache'))
        # If no cache is defined in preferences
        if self._cache is None: self._cache = {}

    @property
    def header(self):
        if not self._isRead['header']:
            self.read(return_headers=True, return_data=False)
        return self._header

    @property
    def is_masked(self): return self._mask is not None
        
    @api
    def keys(self, *args, ensure_read : bool = True, **kwargs):
        if ensure_read: self._ensure_read()
        return itertools.chain(
            super(Output, self).keys(*args, **kwargs),
            self._cache.keys(),
        )
    
    @api
    def values(self, *args, ensure_read : bool = True, **kwargs):
        if ensure_read: self._ensure_read()
        keys = self._cache.keys()
        for key in self.keys():
            if key in keys:
                self[key] = self.from_cache(key)
        return super(Output, self).values(*args, **kwargs)
    
    @api
    def items(self, *args, ensure_read : bool = True, **kwargs):
        if ensure_read: self._ensure_read()
        keys = self.keys()
        for key in self._cache.keys():
            if key not in keys:
                self[key] = self.from_cache(key)
        return super(Output, self).items(*args, **kwargs)
    
    @api
    def copy_from(self, output : 'starsmashertools.lib.output.Output'):
        self._mask = copy.deepcopy(output._mask)
        
        for key in self._isRead.keys():
            self._isRead[key] = True
        
        for key, val in output.items():
            self[key] = copy.deepcopy(val)
    
    def read(self, return_headers=True, return_data=True, **kwargs):
        if self._isRead['header']: return_headers = False
        if self._isRead['data']: return_data = False

        self._header = None
        self.data = None

        read_header = return_headers
        read_data = return_data

        if read_header or read_data:
            obj = self.simulation.reader.read(
                self.path,
                return_headers=read_header,
                return_data=read_data,
                **kwargs
            )

            if read_header and read_data:
                self.data, self._header = obj
            elif not read_header and read_data:
                self.data = obj
            elif read_header and not read_data:
                self._header = obj
        
        if self._header is not None:
            for key, val in self._header.items():
                self[key] = val

        if self.data is not None:
            for key, val in self.data.items():
                self[key] = val

        if read_header: self._isRead['header'] = True
        if read_data: self._isRead['data'] = True
    
    @api
    def from_cache(self, key : str):
        if key in self._cache.keys():
            if callable(self._cache[key]):
                self._cache[key] = self._cache[key](self)
        return self._cache[key]

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def mask(self, mask : np.ndarray | list | tuple):
        if self.is_masked:
            raise Exception("Cannot apply more than 1 mask to an Output object")
        
        if not isinstance(mask, np.ndarray):
            mask = np.asarray(mask)
        if mask.dtype.type == np.bool_ and len(mask) != self['ntot']:
            raise Exception("The given mask does not contain particle IDs but its length (%d) != the number of particles (%d) in the output file '%s'" % (len(mask), self['ntot'], str(self.path)))

        self._clear_cache()

        self._original_data = {}
        for key, val in self.items():
            self._original_data[key] = copy.deepcopy(val)

        for key, val in self.items():
            if isinstance(val, np.ndarray):
                self[key] = val[mask]
        
        self._mask = mask

    @api
    def unmask(self):
        self._mask = None
        # We should really store the cached values that we obtained while being
        # masked so that we don't have to recalculate them, but doing so would
        # be difficult. So we just make it simple and clear the cache.
        self._clear_cache()

        for key, val in self._original_data.items():
            self[key] = val
        self._original_data = None

    @api
    def get_core_particles(self):
        """
        Return the IDs of any core particles present in this output file. A core
        particle is identified in ``StarSmasher`` as having the unique property
        of zero specific internal energy. This is because core particles act 
        only gravitationally on other particles.

        Returns
        -------
        :class:`numpy.ndarray`
            Integer IDs of the core particles in this output file, if there are
            any. Note that the IDs are zero'th indexed, unlike the particle IDs
            in ``StarSmasher``, which start at index 1 instead of 0.
        """
        return self['ID'][self['u'] == 0]

    @api
    def get_non_core_particles(self):
        """
        Similar to :meth:`~.get_core_particles`\, but instead returns the IDs of
        the particles which are not identified as being core particles.
        
        Returns
        -------
        :class:`numpy.ndarray`
            Non-core-particle IDs.
        """
        return self['ID'][self['u'] != 0]

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def condense(
            self,
            identifier : str,
            func : str | typing.Callable | type(None) = None,
            args : list | tuple = (),
            kwargs : dict = {},
            mask : np.ndarray | list | tuple | type(None) = None,
            overwrite : bool = False,
            max_stored : int = Pref('condense.max stored', 10000),
    ):
        """
        Perform an operation as defined by a string or function using this
        :class:`~.Output` object. The result is saved to the 
        :class:`~.simulation.Simulation` archive for quick access in the
        future.
        
        Values stored in the simulation archive will be erased when "max stored
        condense results" in :mod:`starsmashertools.preferences` is reached,
        per output file. If this value is changed, the stored results will be 
        updated automatically the next time this function is called, but only 
        for this output object. The identifiers which were accessed most 
        recently are kept.
        
        It is good practice to ensure the value returned by your ``func`` 
        argument is small, so that many of them can be stored in the simulation
        archive. You could also use this method just to store a large object 
        from a single output file, but it is generally not recommended because
        it significantly increases the overhead of retrieving values from the 
        simulation archive.
        
        Parameters
        ----------
        identifier : str
           The name to use when storing the result of ``func`` in the simulation
           archive. If ``overwrite = False`` then the value which is stored in 
           the simulation archive with key ``identifier`` will be returned and 
           no calculations will be done.

        Other Parameters
        ----------------
        func : str, :class:`typing._CallableType`\, None, default = None
           If `None`\, no calculations will be done and the simulation archive
           will be searched for ``identifier``\. If no key is found matching
           ``identifier`` for this output file, a :py:class:`KeyError` will be
           raised. Keyword ``overwrite`` is ignored in this case.
           
           If type :py:class:`str` is given, it will be given as input to an 
           :py:func:`exec` call. In this case, variables will be accessible as
           the names of the keys in this :class:`~.Output` object, such as 
           ``'x'\, ``'rho'``\, etc., including keys which are in 
           :mod:`starsmashertools.preferences`\. The string must set a
           variable called ``'result'``\, the contents of which will be saved in
           the simulation archive.
        
           If type :class:`typing._CallableType` is given, it must be a function
           which accepts a single input that is this :class:`~.Output` object 
           and it must return an object that can be serialized, such as those
           in :attr:`~.helpers.jsonfile.serialization_methods`\. This is so
           that it can be saved in the simulation's archive.
        
        args : list, tuple, default = ()
           Positional arguments passed directly to ``func`` when it is of type
           :class:`typing._CallableType`\.

        kwargs : dict, default = {}
           Keyword arguments passed directly to ``func`` when it is of type
           :class:`typing._CallableType`\.
        
        mask : :class:`numpy.ndarray`\, list, tuple, None, default = None
           If not `None`\, the mask will be applied to this :class:`~.Output`
           object using :meth:`~.mask`\.

        overwrite : bool, default = False
           Ignored if ``func`` is `None`\.

           If `False` and the simulation archive already contains the given
           ``identifier`` then the archived value is returned. Otherwise, the
           result of ``func`` is added to the simulation archive.
           
           If `True`\, then the result of ``func`` is added to the simulation 
           archive if it doesn't already exists, and replaces the value in the 
           archive if it does.

        max_stored : int, default = ``Pref('condense.max stored', 10000)``
           The maximum number of results from Output.condense that is allowed to
           be stored in the simulation archives per output.

        Returns
        -------
        The value returned depends on the contents of positional argument 
        ``func``\, and/or the contents of the simulation archive.
        """
        archive_key = 'Output.condense'
        archive_value = {}
        if archive_key in self.simulation.archive.keys():
            archive_value = self.simulation.archive[archive_key].value

        relpath = self._get_relpath()
        # Make sure this output file exists in the archived values
        archive_value[relpath] = archive_value.get(relpath, {})
        
        if len(archive_value[relpath].keys()) > max_stored:
            # Reduce the number of stored values
            keys = list(archive_value[relpath].keys())
            values = []
            for key in keys:
                values += [archive_value[relpath][key]]
            
            # Sort keys by oldest-to-newest access times
            keys = [x for _, x in sorted(zip(values, keys), key=lambda pair:pair[0]['access time'])]

            ndel = len(archive_value[relpath].keys()) - max_stored
            keys = keys[ndel:]
            values = values[ndel:]

            archive_value[relpath] = {key:val for key, val in zip(keys, values)}
        
        
        if func is None:
            # Find the pre-existing identifier or throw an error if it doesn't
            # exist
            if identifier not in archive_value[relpath].keys():
                raise KeyError("Identifier '%s' not found in the simulation archive for %s" % (identifier, self))
        else:
            # Run calculations
            if identifier not in archive_value[relpath].keys() or overwrite:
                if mask is not None: self.mask(mask)

                if isinstance(func, str):
                    variables = {key:val for key, val in self.items()}
                    loc = {}
                    exec(func, variables, loc)
                    if 'result' not in loc.keys():
                        raise SyntaxError("Argument 'func' is of type 'str' but it does not set variable 'result'")
                    archive_value[relpath][identifier] = {
                        'value' : loc['result'],
                    }
                else: # It's a function
                    archive_value[relpath][identifier] = {
                        'value' : func(self, *args, **kwargs),
                    }

                if mask is not None: self.unmask()
            # Otherwise, we don't need to run any calculations. Just access the
            # value in the simulation archive.
        
        archive_value[relpath][identifier]['access time'] = time.time()
        # Save to the archive
        self.simulation.archive.add(
            archive_key,
            archive_value,
            mtime = None,
            origin = None,
        )
        return archive_value[relpath][identifier]['value']

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def rotate(
            self,
            keys : list | tuple = [('x', 'y', 'z'), ('vx', 'vy', 'vz'), ('vxdot', 'vydot', 'vzdot')],
            xangle : float | int = 0.,
            yangle : float | int = 0.,
            zangle : float | int = 0.,
    ):
        """
        Rotate the particles using an Euler rotation.

        An Euler rotation can be understood as follows. Imagine the x, y, and z
        axes as dowels. First the z-axis dowel is rotated `zangle` degrees
        clockwise, then the y-axis dowel is rotated `yangle` degrees clockwise,
        and finally the x-axis dowel is rotated `xangle` degrees clockwise.

        Parameters
        ----------
        keys : list, tuple, :class:`numpy.ndarray`\, default = ``[('x', 'y', 'z'), ('vx', 'vy', 'vz'), ('vxdot', 'vydot', 'vzdot')]``
            The particle quantities to rotate. You should carefully specify all
            vector particle quantities (those with x, y, and z components).

        xangle : float, default = 0.
            The x component of an Euler rotation in degrees. Either `xangle` or
            `yangle` can be thought of as equivalent to polar angle "theta", but
            not both.
        
        yangle : float, default = 0.
            The y component of an Euler rotation in degrees. Either `xangle` or
            `yangle` can be thought of as equivalent to polar angle 
            :math:`\\theta`\.

        zangle : float, default = 0.
            The z component of an Euler rotation in degrees. This can be thought
            of as equivalent to azimuthal angle :math:`\\phi`\.
        """
        import starsmashertools.math
        
        for xkey, ykey, zkey in keys:
            x, y, z = self[xkey], self[ykey], self[zkey]
            self[xkey], self[ykey], self[zkey] = starsmashertools.math.rotate(
                x, y, z,
                xangle = xangle,
                yangle = yangle,
                zangle = zangle,
            )

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_extents(
            self,
            radial : bool = False,
            r : list | tuple | np.ndarray | type(None) = None,
    ):
        """
        Get the x, y, and z bounds of the simulation, where the minima are found
        as ``min(x - 2*h)`` and maxima ``max(x + 2*h)`` for the x-axis and 
        similarly for the y and z axes.

        Parameters
        ----------
        radial : bool, default = False
            If True, returns a
            :class:`~.helpers.extents.RadialExtents` instead of a
            :class:`~.helpers.extents.Extents`\. Use this if you
            want the extents of a spherically symmetric simulation, such as a
            :class:`~.relaxation.Relaxation`\.

        r : list, tuple, :class:`numpy.ndarray`\, None, default = None
            The radii of the particle kernels. If `None` then the kernels are
            assumed to have radii 2h.

        Returns
        -------
        :class:`~.helpers.extents.Extents` or :class:`~.helpers.extents.RadialExtents`
        """
        import starsmashertools.helpers.extents

        units = self.simulation.units
        
        if r is None: r = 2 * self['hp'] * float(units['hp'])
        r = np.asarray(r)
        
        x = self['x'] * float(units['x'])
        y = self['y'] * float(units['y'])
        z = self['z'] * float(units['z'])
        
        if radial:
            return starsmashertools.helpers.extents.RadialExtents(self)
        
        return starsmashertools.helpers.extents.Extents(
            xmin = starsmashertools.lib.units.Unit(np.amin(x - r), units.length.label),
            xmax = starsmashertools.lib.units.Unit(np.amax(x + r), units.length.label),
            ymin = starsmashertools.lib.units.Unit(np.amin(y - r), units.length.label),
            ymax = starsmashertools.lib.units.Unit(np.amax(y + r), units.length.label),
            zmin = starsmashertools.lib.units.Unit(np.amin(z - r), units.length.label),
            zmax = starsmashertools.lib.units.Unit(np.amax(z + r), units.length.label),
        )
        

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_formatted_string(
            self,
            format_sheet : str = Pref('get_formatted_string.format sheet'),
    ):
        """
        Convert an output file to a (mostly) human-readable string format. 
        This will only display the values in the header of the output file and 
        it uses one of the format sheets located in the ``format_sheets`` 
        directory.

        Parameters
        ----------
        output : :class:`~.Output`
            The output file to convert to a string.

        Other Parameters
        ----------------
        format_sheet : str, None, default = ``Pref('get_formatted_string.format sheet')``
            The format sheet to use when converting.

        Returns
        -------
        str
            The formatted string.
        """
        import starsmashertools.helpers.formatter
        return starsmashertools.helpers.formatter.Formatter(
            format_sheet,
        ).format_output(self)

    if has_matplotlib:
        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def plot(
                self,
                ax : matplotlib.axes.Axes,
                x : str = 'x',
                y : str = 'y',
                **kwargs
        ):
            """
            Create a scatter plot on the given Matplotlib axes showing the
            particle values.
            
            Parameters
            ----------
            ax : matplotlib.axes.Axes
                The axis to make the plot on.

            x : str, default = 'x'
                The output file key to plot on the x-axis.

            y : str, default = 'y'
                The output file key to plot on the y-axis.

            Other Parameters
            ----------------
            kwargs
                Keyword arguments are passed directly to 
                :class:`starsmashertools.mpl.artists.OutputPlot` .
            
            Returns
            -------
            :class:`starsmashertools.mpl.artists.OutputPlot`
                A Matplotlib ``Artist``\.

            See Also
            --------
            :mod:`starsmashertools.preferences` ('Plotting'), 
            :class:`starsmashertools.mpl.artists.OutputPlot`
            """
            import starsmashertools.mpl.artists
            ret = starsmashertools.mpl.artists.OutputPlot(
                ax, x, y, **kwargs
            )
            ret.set_output(self)
            return ret
















































    


        









# Asynchronous output file reading
@starsmashertools.preferences.use
class OutputIterator(object):
    """
    An iterator which can be used to iterate through :class:`~.Output` objects
    efficiently.
    """
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(
            self,
            filenames : list | tuple | np.ndarray,
            simulation,
            onFlush : list | tuple | np.ndarray = [],
            max_buffer_size : int = Pref('max buffer size', 100),
            asynchronous : bool = True,
            **kwargs,
    ):
        """
        Constructor.
        
        Parameters
        ----------
        filenames : list, tuple, :class:`numpy.ndarray`
            The list of file names to iterate through.

        simulation : :class:`~starsmashertools.lib.simulation.Simulation`
            The simulation that the given file names belong to.

        onFlush : list, tuple, :class:`numpy.ndarray`\, default = []
            A list of functions to be called in the :meth:`~.flush` method. If 
            one of the methods returns ``'break'`` then iteration is stopped and
            no other ``onFlush`` functions are called.

        max_buffer_size : int, default = ``Pref('max buffer size', 100)``
            The maximum size of the buffer for reading Output ahead-of-time.
        
        asynchronous : bool, default = True
            If `True`\, Output objects are read asynchronously in a separate
            process than the main thread. Otherwise, Output objects are read on
            an "as needed" basis in serial.
        
        Other Parameters
        ----------------
        **kwargs
            Other optional keyword arguments that are passed to the
            :meth:`~.Output.read` function.
        """
        import starsmashertools.lib.simulation
        
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'simulation' : [starsmashertools.lib.simulation.Simulation],
        })
        self.max_buffer_size = max_buffer_size
        self.onFlush = onFlush
        self.simulation = simulation
        self.asynchronous = asynchronous
        self.kwargs = kwargs
        
        for m in self.onFlush:
            if not callable(m):
                raise TypeError("Callbacks in keyword 'onFlush' must be callable, but received '%s'" % str(m))

        # Make sure that the filenames are the true paths
        self.filenames = filenames
        self._break = False

        self._process = None

        self._index = -1
        self._buffer_index = -1

    def __str__(self):
        import starsmashertools.helpers.path
        string = self.__class__.__name__ + "(%s)"
        if len(self.filenames) == 0: return string % ""
        bname1 = starsmashertools.helpers.path.basename(self.filenames[0])
        if len(self.filenames) == 1: return string % ("'%s'" % bname1)
        bname2 = starsmashertools.helpers.path.basename(self.filenames[-1])
        return string % ("'%s' ... '%s'" % (bname1, bname2))

    def __repr__(self): return str(self)

    def __contains__(self, item):
        import starsmashertools.helpers.path
        import starsmashertools.helpers.file
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
        import starsmashertools.helpers.asynchronous
        if len(self.filenames) == 0: self.stop()
        
        self._buffer_index += 1

        if self._buffer_index >= self.max_buffer_size:
            # Wait for the process to be finished before continuing
            if self.asynchronous:
                if self._process is not None and self._process.is_alive(): self._process.join()
                self._process = starsmashertools.helpers.asynchronous.Process(
                    target = self.flush,
                    daemon=True,
                )
                self._process.start()
            else:
                self.flush()
            self._buffer_index = -1

        if self._break: self.stop()

        self._index += 1
        if self._index < len(self.filenames):
            return self.get(self.filenames[self._index])
        else:
            self.stop()
    
    def next(self, *args, **kwargs): return self.__next__(*args, **kwargs)

    def stop(self):
        # Ensure that we do the flush methods whenever we stop iterating
        self.flush()
        raise StopIteration

    def flush(self):
        for m in self.onFlush:
            r = m()
            if isinstance(r, str) and r == 'break':
                self._break = True
                break
    
    def get(self, filename):
        o = Output(filename, self.simulation)
        o.read(**self.kwargs)
        return o

    def tolist(self):
        return [Output(filename, self.simulation) for filename in self.filenames]







class ParticleIterator(OutputIterator, object):
    """
    Similar to an :class:`~.OutputIterator`, but instead of iterating through 
    all the particles in each Output file, we iterate through a list of
    particles instead.
    """
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(
            self,
            particle_IDs : int | list | tuple | np.ndarray, # 0th index
            *args,
            **kwargs
    ):
        if not hasattr(particle_IDs, '__iter__'): particle_IDs = [particle_IDs]
        super(ParticleIterator, self).__init__(*args, **kwargs)
        self.particle_IDs = particle_IDs

    def get(self, filename):
        import starsmashertools.helpers.file
        import starsmashertools.helpers.readonlydict
        
        if len(self.particle_IDs) == 0: return {}
        # Calculate the file position for the quantity we want to read
        header_stride = self.simulation.reader._stride['header']
        header_dtype = self.simulation.reader._dtype['header']
        
        data_stride = self.simulation.reader._stride['data']
        dtype = self.simulation.reader._dtype['data']

        sort_indices = np.argsort(self.particle_IDs)
        IDs = np.asarray(self.particle_IDs)[sort_indices]
        # Each 4 is a Fortran record marker
        positions = [4 + header_stride + 4 + 4 + (data_stride + 4+4) * ID for ID in IDs]

        # Calling Reader.read also ensures that the proper Fortran write
        # statements have been located ahead of time
        header = self.simulation.reader.read(
            filename,
            return_headers = True,
            return_data = False,
        )
        
        with starsmashertools.helpers.file.open(
                filename, 'rb', lock = False, verbose = False,
        ) as f:
            buffer = mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
        
        _buffer = bytearray(len(IDs) * (data_stride + 4+4))
        for i, (ID, position) in enumerate(zip(IDs, positions)):
            pos = i * (data_stride+4+4)
            _buffer[pos : pos + data_stride] = buffer[position : position + data_stride]
        d = np.ndarray(
            buffer=_buffer,
            shape=len(self.particle_IDs),
            dtype=dtype,
            strides=data_stride + 4 + 4,
        )[np.argsort(sort_indices)]
        return starsmashertools.helpers.readonlydict.ReadOnlyDict(
            {name:d[name].flatten() for name in d.dtype.names} | \
            {'t': header['t'], 'ID' : self.particle_IDs}
        )

                
                




































    



@starsmashertools.preferences.use
class Reader(object):
    def __init__(
            self,
            simulation,
    ):
        self.simulation = simulation

        self._endian = None
        self._dtype = None
        self._stride = None
        self._formats = {
            'header' : {
                'format' : None,
                'variables' : None,
            },
            'data' : {
                'format' : None,
                'variables' : None,
            },
        }

    def get_endian(self):
        if self._endian is None:
            for s in ['header','data']:
                if self._formats[s]['format'] is None: continue
                if self._formats[s]['format'].startswith('<'):
                    self._endian = 'little'
                    break
                elif self._formats[s]['format'].startswith('>'):
                    self._endian = 'big'
                    break
            else: return None
        return self._endian
    
    def read_from_header(self, key : str, filename : str):
        import starsmashertools.helpers.file
        
        if self._formats['header']['variables'] is None:
            self.update_format(filename)
        
        variables = self._formats['header']['variables']
        names = [v.name for v in variables]
        if key not in names:
            raise KeyError("No key '%s' in the dtypes" % key)
        variable = variables[names.index(key)]
        
        with starsmashertools.helpers.file.open(
                filename, 'rb', lock = False, verbose = False,
        ) as f:
            f.seek(4 + sum([v.size for v in variables[:names.index(key)]]))
            return variable.get_value(f.read(variable.size), self.get_endian())

    def read(
            self,
            filename : str,
            return_headers : bool = True,
            return_data : bool = True,
            verbose : bool = False,
    ):
        import starsmashertools.helpers.file
        import starsmashertools.helpers.path
        
        if verbose: print(filename)

        if True not in [return_headers, return_data]:
            raise ValueError("One of 'return_headers' or 'return_data' must be True")
        
        if None in [self._dtype, self._stride]:
            self.update_format(filename)
        
        endian = self.get_endian()
        
        try:
            with starsmashertools.helpers.file.open(
                    filename, 'rb', lock = False, verbose = False,
            ) as f:
                # Check how many rows of data the file has
                # The last thing written to the file must always be variable
                # "ntot"
                f.seek(-4, 2)
                recl = int.from_bytes(f.read(4), byteorder = endian)
                f.seek(-(4 + recl), 2)
                size = f.tell() - 4
                ntot = int.from_bytes(f.read(recl), byteorder = endian)
                nlines = (size - (self._stride['header'] + 8)) // (self._stride['data'] + 8)
                # Compare the number of lines to the written number of particles
                # at the end of the file. This number must always be written as
                # a regular Fortran integer
                if nlines != ntot:
                    raise Reader.CorruptedFileError("nlines != ntot (%d != %d). This Output might have been written by a different simulation. Make sure you use the correct simulation when creating an Output object, as different simulation directories have different reading and writing methods in their source directories: '%s'" % (nlines, ntot, filename))

                if return_headers:
                    f.seek(4)
                    header = np.ndarray(
                        buffer = f.read(self._stride['header']),
                        shape = 1,
                        dtype = self._dtype['header'],
                        offset = 0, # start-of-record
                        strides = self._stride['header'],
                    )
                
                
                if return_data:
                    # Go past the header's trailing record marker, to the start
                    # of the first data row's record marker
                    data = np.ndarray(
                        # mmap speeds up reading significantly, but is about 2x
                        # slower when reading only the header.
                        buffer = mmap.mmap(
                            f.fileno(), 0, access=mmap.ACCESS_READ,
                        ),
                        shape = ntot,
                        dtype = self._dtype['data'],
                        # Start at the beginning of the first record, beyond the
                        # record marker for the first line of data.
                        offset = 4 + self._stride['header'] + 4 + 4,
                        # Skip the record markers at the end of each row, and at
                        # the beginning of each next row.
                        strides = self._stride['data'] + 4 + 4,
                    )
        except Exception as e:
            if not isinstance(e, Reader.CorruptedFileError):
                raise Reader.CorruptedFileError("This Output might have been written by a different simulation. Make sure you use the correct simulation when creating an Output object, as different simulation directories have different reading and writing methods in their source directories: '%s'" % filename) from e
            raise

        # Convert to read-only dictionaries
        if return_headers:
            header = starsmashertools.helpers.readonlydict.ReadOnlyDict(
                {name:np.array(header[name])[0] for name in header[0].dtype.names}
            )
        if return_data:
            data = starsmashertools.helpers.readonlydict.ReadOnlyDict(
                {name:np.array(data[name]) for name in data[0].dtype.names} | \
                {'ID': np.arange(ntot, dtype=int)} # Always include an 'ID' key
            )
        
        if return_headers and not return_data: return header
        if not return_headers and return_data: return data
        if return_headers and return_data: return header, data
        
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def set_write_statements(
            self,
            header : str | type(None) = None,
            data : str | type(None) = None,
    ):
        r"""
        Manually set the header and data write statements from the StarSmasher
        Fortran source code. This is helpful in case there are multiple write
        statements of the same formatting and you want to distinguish which of
        them actually gets used in the simulation.

        Parameters
        ----------
        header : str, None, default = None
            The Fortran write statement used to create the headers. If `None`\,
            this parameter is ignored.

        data : str, None, default = None
            The Fortran write statement used to create the data rows. If 
            `None`\, this parameter is ignored.
        """
        import starsmashertools.helpers.fortran
        import starsmashertools.helpers.path

        src = starsmashertools.helpers.path.get_src(self.simulation.directory)
        if src is None:
            raise Reader.SourceDirectoryNotFoundError(self.simulation)

        ws_vars = []
        write_statements = []
        for path in starsmashertools.helpers.path.find_files(src):
            for statement, variables in starsmashertools.helpers.fortran.FortranFile(
                    path
            ).get_write_statements():
                ws_vars += [variables]
                write_statements += [statement]
        
        for s,d in zip(['header','data'], [header, data]):
            if d is None: continue
            statement = copy.deepcopy(d)
            if not starsmashertools.helpers.fortran.is_stripped(d):
                statement = starsmashertools.helpers.fortran.strip(statement)
            if statement not in write_statements:
                raise Exception("Failed to find an identical write statement in the StarSmasher source code as:\n%s" % statement)
            self._formats[s]['variables'] = ws_vars[write_statements.index(statement)]
    
    
    def update_format(self, filename : str):
        r"""
        Search the StarSmasher source directory for all fortran "write" 
        statements and check to see if each write statement fits the formatting
        of the out*.sph file. Raises an exception when more than one valid 
        format is located.

        Note that strings are not permitted in the output files.
        """
        import starsmashertools.helpers.path
        import starsmashertools.helpers.file
        import starsmashertools.helpers.fortran
        
        src = starsmashertools.helpers.path.get_src(self.simulation.directory)
        if src is None:
            raise Reader.SourceDirectoryNotFoundError(self.simulation)

        # Figure out the length of the header and body. No need to lock the file
        # because starsmashertools never edits StarSmasher outputs.
        endian = self.get_endian()
        if endian is None:
            endian = starsmashertools.helpers.fortran.get_endian_from_file(
                filename,
            )
        with starsmashertools.helpers.file.open(
                filename, 'rb', lock = False, verbose = False
        ) as f:
            recl = int.from_bytes(f.read(4), byteorder = endian)
            f.seek(0)
            header = f.read(recl + 8)
            start = f.tell()
            recl = int.from_bytes(f.read(4), byteorder = endian)
            f.seek(start)
            dataline = f.read(recl + 8)

        # Now we know the header contents and the contents of the first line of
        # data in the file. We try to locate the write statement that was
        # responsible for writing the header and each row of data. Note that it
        # is possible for there for be more than one valid write statement, but
        # we are not doing a full translation of the fortran code into python,
        # so we can't know which one is correct.
        if None in [s['variables'] for s in self._formats.values()]:
            ws_vars = []
            write_statements = []
            for path in starsmashertools.helpers.path.find_files(src):
                f = starsmashertools.helpers.fortran.FortranFile(path)
                for statement, variables in f.get_write_statements():
                    ws_vars += [variables]
                    write_statements += [statement]
        
        for s,byte_data in zip(['header','data'],[header,dataline]):
            if self._formats[s]['variables'] is not None: continue
            indices = []
            for i, variables in enumerate(ws_vars):
                attempt = starsmashertools.helpers.fortran.get_numpy_format(
                    byte_data, variables, endian = endian,
                )
                if attempt is not None: indices += [i]
            if len(indices) == 0:
                raise Exception("Found no possible write statements in the StarSmasher source code that correspond with %s information in file '%s'" % (s,filename))
            if len(indices) > 1:
                p1 = "Found more than one possible write statements in the StarSmasher source code that correspond with %s information in file '%s'. Please choose one of the following write statements and manually set it for this reader, using \"simulation.reader.set_write_statements(%s = ...)\":" % (s,filename,s)
                p1 += '\n' + '\n\n'.join([write_statements[i] for i in indices])
                raise Exception(p1)
            self.set_write_statements(**{s:write_statements[indices[0]]})
        
        for s,byte_data in zip(['header','data'],[header,dataline]):
            self._formats[s]['format'] = starsmashertools.helpers.fortran.get_numpy_format(
                byte_data,
                self._formats[s]['variables'],
                endian = endian,
            )
        
        
        header_fmt = self._formats['header']['format']
        data_fmt = self._formats['data']['format']
        header_names = [v.name for v in self._formats['header']['variables']]
        data_names = [v.name for v in self._formats['data']['variables']]

        self._dtype = {
            key:np.dtype([
                (name,f) for name,f in zip(names, fmt.split(','))
            ]) for key,names,fmt in zip(
                ['header','data'],
                [header_names,data_names],
                [header_fmt,data_fmt],
            )
        }
        self._stride = {
            key:sum([
                v.size for v in self._formats[key]['variables']
            ]) for key in ['header', 'data']
        }
    
    class CorruptedFileError(Exception):
        def __init__(self, message):
            super(Reader.CorruptedFileError, self).__init__(message)

    class SourceDirectoryNotFoundError(Exception):
        def __init__(self, simulation, message=""):
            if not message:
                message = "Failed to find the code source directory in simulation '%s'" % simulation.directory
            super(Reader.SourceDirectoryNotFoundError, self).__init__(message)

