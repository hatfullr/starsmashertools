import starsmashertools.preferences
from starsmashertools.preferences import Pref
import starsmashertools.helpers.argumentenforcer
from starsmashertools.helpers.apidecorator import api
import numpy as np
import typing

try:
    import matplotlib.axes
    has_matplotlib = True
except ImportError:
    has_matplotlib = False

@starsmashertools.preferences.use
class Output(dict, object):
    """
    A container for StarSmasher binary output data, usually appearing as
    ``out*.sph`` in simulation directories.

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
        import starsmashertools.helpers.path
        import starsmashertools.helpers.string
        
        if mode not in Output.modes:
            s = starsmashertools.helpers.string.list_to_string(Output.modes, join='or')
            raise ValueError("Keyword argument 'mode' must be one of %s, not '%s'" % (s, str(mode)))
        self.path = starsmashertools.helpers.path.realpath(path)
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
            starsmashertools.helpers.path.realpath(self.path),
            start = self.simulation.directory,
        )

    #"""
    # Used when pickling
    def __getstate__(self):
        import starsmashertools.helpers.path
        return {
            'path' : self.path,
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
        return string % ("'%s'" % starsmashertools.helpers.path.basename(self.path))

    def __repr__(self): return str(self)

    @api
    def __getitem__(self, item):
        import copy
        
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
        return starsmashertools.helpers.file.compare(self.path, other.path)

    def __hash__(self, *args, **kwargs):
        import starsmashertools.helpers.path
        return self.path.__hash__(*args, **kwargs)

    def __copy__(self,*args,**kwargs):
        ret = Output(self.path, self.simulation, mode=self.mode)
        ret.copy_from(self)
        return ret
    def __deepcopy__(self,*args,**kwargs):
        return self.__copy__(*args, **kwargs)

    # Use this to make sure that the file has been fully read
    def _ensure_read(self):
        if False in self._isRead.values():
            self.read(return_headers=not self._isRead['header'], return_data=not self._isRead['data'])

    def _clear_cache(self):
        import copy
        self._cache = copy.copy(self.preferences.get('cache'))
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
        import itertools
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
    def copy_from(self, obj):
        import copy
        self._mask = copy.deepcopy(obj._mask)
        
        for key in self._isRead.keys():
            self._isRead[key] = True
        
        for key, val in obj.items():
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
    def from_cache(self, key):
        if key in self._cache.keys():
            if callable(self._cache[key]):
                self._cache[key] = self._cache[key](self)
        return self._cache[key]

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def mask(self, mask : np.ndarray | list | tuple):
        import copy
        
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
        return np.where(self['u'] == 0)[0]

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
        return np.where(self['u'] != 0)[0]

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
        import time
            
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
        import copy
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
    Similar to an OutputIterator, but instead of iterating through all the
    particles in each Output file, we iterate through a list of particles
    instead. This involves a different kind of Reader.
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
        import mmap
        
        if len(self.particle_IDs) == 0: return {}
        # Calculate the file position for the quantity we want to read
        header_stride = self.simulation.reader._stride['header']
        header_dtype = self.simulation.reader._dtype['header']
        
        EOL_size = self.simulation.reader._EOL_size
        data_stride = self.simulation.reader._stride['data'] + EOL_size
        dtype = self.simulation.reader._dtype['data']

        sort_indices = np.argsort(self.particle_IDs)
        IDs = np.asarray(self.particle_IDs)[sort_indices]
        positions = [header_stride + EOL_size + data_stride * ID for ID in IDs]
        
        _buffer = bytearray(len(IDs) * data_stride)
        with starsmashertools.helpers.file.open(
                filename, 'rb', lock = False, verbose = False,
        ) as f:
            buffer = mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
        # Grab the headers so we can record the times
        try:
            header = self.simulation.reader._read(
                buffer=buffer,
                shape=1,
                dtype=header_dtype,
                offset=4,
                strides=header_stride,
            )
        except Exception as e:
            raise Reader.CorruptedFileError("This Output might have been written by a different simulation. Make sure you use the correct simulation when creating an Output object, as different simulation directories have different reading and writing methods in their source directories. %s" % filename) from e
        
        for i, (ID, position) in enumerate(zip(IDs, positions)):
            pos = i * data_stride
            _buffer[pos : pos + data_stride] = buffer[position : position + data_stride]
        try:
            d = self.simulation.reader._read(
                buffer=_buffer,
                shape=len(self.particle_IDs),
                dtype=dtype,
                offset=4,
                strides=data_stride,
            )[np.argsort(sort_indices)]
        except Exception as e:
            raise Reader.CorruptedFileError("This Output might have been written by a different simulation. Make sure you use the correct simulation when creating an Output object, as different simulation directories have different reading and writing methods in their source directories. %s" % filename) from e
        
        data = {name:d[name].flatten() for name in d.dtype.names}
        data['t'] = header['t'][0]
        data['ID'] = self.particle_IDs
        data = starsmashertools.helpers.readonlydict.ReadOnlyDict(data)
        return data
                
                




































    




class Reader(object):
    formats = {
        'integer' : 'i4',
        'real*8'  : 'f8',
        'real'    : 'f4',
    }

    EOL = 'f8'
    
    def __init__(self, simulation):
        self.simulation = simulation
        header_format, header_names, data_format, data_names = Reader.get_output_format(self.simulation)
        
        self._dtype = {
            'data' : np.dtype([(name, fmt) for name, fmt in zip(data_names, data_format.split(","))]),
            'header' : np.dtype([(name, fmt) for name, fmt in zip(header_names, header_format.split(","))]),
        }

        self._stride = {
            'header' : sum([header_format.count(str(num))*num for num in [1, 2, 4, 6, 8]]),
            'data' : sum([data_format.count(str(num))*num for num in [1, 2, 4, 6, 8]]),
        }

        self._EOL_size = sum([Reader.EOL.count(str(num))*num for num in [1, 2, 4, 6, 8]])

    def read_from_header(self, key : str, filename : str):
        import starsmashertools.helpers.file
        names = self._dtype['header'].names
        if key not in names:
            raise KeyError("No key '%s' in the dtypes" % key)

        index = names.index(key)
        size = self._dtype['header'][index].itemsize
        others = names[:index]
        location = 4 + sum([self._dtype['header'][name].itemsize for name in others])
        
        with starsmashertools.helpers.file.open(
                filename, 'rb', lock = False, verbose = False,
        ) as f:
            f.seek(location)
            content = f.read(size)
        return self._read(
            buffer = content,
            shape = 1,
            dtype = self._dtype['header'][key],
            offset = 0,
            strides = size,
        )[0]
        
    def _read(self, *args, **kwargs):
        ret = np.ndarray(*args, **kwargs)
        if ret.dtype.names is not None:
            for name in ret.dtype.names:
                finite = np.isfinite(ret[name])
                if not np.any(finite): continue
                if np.any(np.abs(ret[name][finite]) > 1e100):
                    raise Reader.UnexpectedFileFormatError
        else:
            finite = np.isfinite(ret)
            if np.any(finite):
                if np.abs(ret[finite]) < 1e-100 or np.abs(ret[finite]) > 1e100:
                    raise Reader.UnexpectedFileFormatError
        return ret

    def _read_header(self, buffer):
        import starsmashertools.helpers.readonlydict
        
        header = self._read(
            buffer=buffer,
            shape=1,
            dtype=self._dtype['header'],
            offset=4,
            strides=self._stride['header'],
        )
        
        new_header = {}
        for name in header[0].dtype.names:
            new_header[name] = np.array(header[name])[0]
        header = starsmashertools.helpers.readonlydict.ReadOnlyDict(new_header)
        return header

    def _read_data(self, buffer, ntot):
        import starsmashertools.helpers.readonlydict
        # There are 'ntot' particles to read
        data = self._read(
            buffer=buffer,
            shape=ntot,
            dtype=self._dtype['data'],
            offset=self._stride['header'] + self._EOL_size + 4,
            strides=self._stride['data'] + self._EOL_size,
        )
        
        # Now organize the data into Pythonic structures
        new_data = {}
        for name in data[0].dtype.names:
            new_data[name] = np.array(data[name])
        return starsmashertools.helpers.readonlydict.ReadOnlyDict(new_data)

    def read(
            self,
            filename,
            return_headers=True,
            return_data=True,
            verbose=False,
    ):
        import starsmashertools.helpers.file
        import starsmashertools.helpers.path
        import mmap
        
        if verbose: print(filename)
        
        if True not in [return_headers, return_data]:
            raise ValueError("One of 'return_headers' or 'return_data' must be True")
        try:
            with starsmashertools.helpers.file.open(
                    filename, 'rb', lock = False, verbose = False,
            ) as f:
                # We always need to check ntot at beginning and end of file
                f.seek(4) # Garbage in front (Fortran issue?)
                ntot_buffer = f.read(8)

                if return_data:
                    # This speeds up reading significantly when reading data,
                    # but it's about 2x slower when reading the header only.
                    buffer = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                else: # Return header only
                    f.seek(0)
                    buffer = f.read(self._stride['header'] + 4)

                f.seek(-8, 2) # Go to 8th byte before the end
                ntot_check_buffer = f.read(8) # Specifying 8 here goes faster
            
            ntot = self._read(
                buffer = ntot_buffer,
                shape = 1,
                dtype = '<i4',
                offset = 0,
                strides = 8,
            )[0]
            
            ntot_check = self._read(
                buffer = ntot_check_buffer,
                shape = 1,
                dtype = '<i4',
                offset = 0,
                strides = 8,
            )[0]
        except Exception as e:
            raise Reader.CorruptedFileError("This Output might have been written by a different simulation. Make sure you use the correct simulation when creating an Output object, as different simulation directories have different reading and writing methods in their source directories: '%s'" % filename) from e
        
        if ntot != ntot_check:
            raise Reader.CorruptedFileError("ntot != ntot_check. This Output might have been written by a different simulation. Make sure you use the correct simulation when creating an Output object, as different simulation directories have different reading and writing methods in their source directories: '%s'" % filename)
        
        try:
            if return_headers:
                header = self._read_header(buffer)
            
            if return_headers and not return_data: return header
            # return_data == True here
            data = self._read_data(buffer, ntot)
            
            if return_headers: return data, header
            else: return data # If return_headers == False
        except Exception as e:
            raise Reader.CorruptedFileError("This Output might have been written by a different simulation. Make sure you use the correct simulation when creating an Output object, as different simulation directories have different reading and writing methods in their source directories: '%s'" % filename) from e
        
    # This method returns instructions on how to read the data files
    @staticmethod
    def get_output_format(simulation):
        import starsmashertools.helpers.path
        import starsmashertools.helpers.file
        import starsmashertools.helpers.string
        import collections
        
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
        with starsmashertools.helpers.file.open(
                writerfile, 'r', lock = False, verbose = False,
        ) as f:
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
                with starsmashertools.helpers.file.open(
                        fname, 'r', lock = False, verbose = False,
                ) as f:
                    for key, val in starsmashertools.helpers.string.get_fortran_variable_types(f.read(), data_types).items():
                        if key not in vtypes.keys(): vtypes[key] = val
                        else: vtypes[key] += val
        
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










        
