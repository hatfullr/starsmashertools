import matplotlib.image
import starsmashertools.lib.output
import numpy as np
import starsmashertools.helpers.argumentenforcer
import starsmashertools.lib.kernels
import starsmashertools.lib.units
import starsmashertools.helpers.asynchronous
import multiprocessing
import warnings
import traceback
import time

try:
    import starsmashertools.helpers.gpujob
    from numba import cuda
    import math
    has_gpus = True
except ImportError:
    has_gpus = False

class ColumnDensity(matplotlib.image.AxesImage, object):
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def __init__(
            self,
            output : starsmashertools.lib.output.Output,
            key_or_array : str | list | tuple | np.ndarray,
            *args,
            max_resolution : list | tuple | np.ndarray | type(None) = None,
            kernel_function : str | starsmashertools.lib.kernels._BaseKernel | type(None) = None,
            units : str | dict | starsmashertools.lib.units.Units | type(None) = None,
            view_units : dict | starsmashertools.lib.units.Units | type(None) = None,
            log : bool = False,
            parallel : bool = True,
            gpus : bool = True,
            interpolation : str | type(None) = 'none',
            origin : str = 'lower',
            **kwargs
    ):
        r"""
        Create a column density plot.

        Parameters
        ----------
        output : :class:`~.lib.output.Output`
            The StarSmasher output object to use.

        key_or_array : str, list, tuple, :class:`numpy.ndarray`
            If a :py:class:`str`\, it must correspond to one of the particle
            arrays in ``output``\. Otherwise, it must be an array of values with
            the same length as the number of particles in ``output``\. Units
            will not be used if an array is given.

        *args
            Other positional arguments are passed directly to 
            :meth:`matplotlib.image.AxesImage.__init__`\.

        Other Parameters
        ----------------
        max_resolution : list, tuple, :class:`numpy.ndarray`\, None, default = None
            The maximum allowed image resolution. If an array is given, it must 
            be length 2, where the first element is the ``x`` resolution and the
            second is the ``y`` resolution. If ``None``\, then the image will
            always have a resolution equal to the number of pixels in the axes.

        kernel_function : str, :class:`~.lib.kernels._BaseKernel`\, None, default = None
            The kernel function to use while integrating. If ``None``\, then
            ``key_or_array`` is considered to be constant throughout each
            particle's kernel. If a :py:class:`str` is given, it will be 
            converted into a :class:`~.lib.kernels._BaseKernel` type using
            :func:`~.lib.kernels.get_by_name`\.

        units : str, dict, :class:`~.lib.units.Units`\, None, default = None
            If ``'none'`` then no units will be used. If ``None``\, the units of
            the simulation that ``output`` comes from will be used. If a dict is
            given, it can contain key ``'A'`` to set the units of the image 
            values. If ``key_or_array`` is a :py:class:`str` then the units of
            the image values will be set to ``units[key_or_array]`` if there is
            no key ``'A'`` in ``units``\.

        view_units : dict, :class:`~.lib.units.Units`\, None, default = None
            The ``units`` keyword argument is used to convert from StarSmasher
            code units to cgs, and ``view_units`` sets the units that are shown
            on the plot. If ``None``\, then ``units`` will be shown. If a dict 
            is given, it can contain key ``'A'`` to set the units of the image 
            values. If ``key_or_array`` is a :py:class:`str` then the units of
            the image values will be set to ``view_units[key_or_array]`` if 
            there is no key ``'A'`` in ``view_units``\.

        interpolation : str, None, default = 'none'
            See :meth:`matplotlib.image.AxesImage.__init__`\.

        log : bool, default = False
            If `True`\, the image will be created in log10.

        parallel : bool, default = True
            If `True`\, calculations will be performed using multiprocessing.

        gpus : bool, default = True
            If `True`\, the value of ``parallel`` is ignored and calculations 
            are instead performed using Numba CUDA.

        **kwargs
            Other keyword arguments are passed directly to 
            :meth:`matplotlib.image.AxesImage.__init__`\.
        
        See Also
        --------
        :meth:`matplotlib.image.AxesImage.__init__`
        """
        if units is None: units = output.simulation.units
        if isinstance(kernel_function, str):
            kernel_function = starsmashertools.lib.kernels.get_by_name(
                kernel_function
            )()
        elif kernel_function is None:
            kernel_function = starsmashertools.lib.kernels.get_by_nkernel(
                output.simulation['nkernel'],
            )()

        if not kernel_function.integrated:
            kernel_function.set_integrated(True)

        kernel_function.interpolate_table = False

        self.output = output
        self.key_or_array = key_or_array
        self.max_resolution = max_resolution
        self.kernel_function = kernel_function
        self.units = units
        self.view_units = view_units
        self.log = log
        self.gpus = gpus
        self.parallel = parallel
        self._args = args
        self._kwargs = kwargs

        if self.gpus and not has_gpus:
            raise ValueError("Keyword argument 'gpus' is True, but either no GPUs were detected or the Numba installation was not detected")
        
        properties = self.get_image_properties()
        x = properties['x']
        y = properties['y']
        r = properties['r']

        vmin = kwargs.pop('vmin', None)
        vmax = kwargs.pop('vmax', None)
        
        kwargs['extent'] = kwargs.get(
            'extent',
            [np.amin(x - r), np.amax(x + r),
             np.amin(y - r), np.amax(y + r)],
        )
        
        super(ColumnDensity, self).__init__(
            *args,
            interpolation = interpolation,
            origin = origin,
            **kwargs
        )

        self.set_array(np.full((1, 1), np.nan))

        if 'alpha' in kwargs: self.set_alpha(kwargs['alpha'])
        
        if self.get_clip_path() is None:
            # image does not already have clipping set, clip to Axes patch
            self.set_clip_path(self.axes.patch)
        self._scale_norm(
            kwargs.get('norm', None),
            vmin, vmax,
        )
        if 'url' in kwargs: self.set_url(kwargs['url'])

        # update ax.dataLim, and, if autoscaling, set viewLim
        # to tightly fit the image, regardless of dataLim.
        #self.set_extent(self.get_extent())
        
        self.calculate()
        
        self.axes.add_image(self)

    def get_image_properties(self):
        import copy
        if not hasattr(self, '_axes'): resolution = None
        else: resolution = self.get_resolution()
        
        x = copy.deepcopy(self.output['x'])
        y = copy.deepcopy(self.output['y'])
        h = copy.deepcopy(self.output['hp'])
        m = copy.deepcopy(self.output['am'])
        rho = copy.deepcopy(self.output['rho'])
        
        # The column density quantity to show
        if isinstance(self.key_or_array, str):
            A = copy.deepcopy(self.output[self.key_or_array])
        else: A = copy.deepcopy(self.key_or_array)

        if self.units is not None:
            if 'x' in self.units: x *= float(self.units['x'])
            if 'y' in self.units: y *= float(self.units['y'])
            if 'hp' in self.units: h *= float(self.units['hp'])
            if 'am' in self.units: m *= float(self.units['am'])
            if 'rho' in self.units: rho *= float(self.units['rho'])
            if 'A' in self.units: A *= float(self.units['A'])
            else:
                if isinstance(self.key_or_array, str):
                    A *= float(self.units[self.key_or_array])
        
        A *= m / rho
        r = self.kernel_function.compact_support * h
        
        if resolution is None:
            dx, dy, xmin, xmax, ymin, ymax = None, None, None, None, None, None
        else:
            xmin = np.amin(x - r)
            xmax = np.amax(x + r)
            ymin = np.amin(y - r)
            ymax = np.amax(y + r)
            #xmin, xmax, ymin, ymax = self.get_extent()
            #xmin, xmax = self.axes.get_xlim()
            #ymin, ymax = self.axes.get_ylim()
            dx = (xmax - xmin) / resolution[0]
            dy = (ymax - ymin) / resolution[1]

        physical_extent = np.asarray([xmin, xmax, ymin, ymax])
        view_extent = copy.deepcopy(physical_extent)
        if self.view_units is not None:
            if 'x' in self.view_units:
                if view_extent[0] is not None:
                    view_extent[0] /= self.view_units['x']
                if view_extent[1] is not None:
                    view_extent[1] /= self.view_units['x']
            if 'y' in self.view_units:
                if view_extent[2] is not None:
                    view_extent[2] /= self.view_units['y']
                if view_extent[3] is not None:
                    view_extent[3] /= self.view_units['y']
        
        return {
            'ntot' : self.output['ntot'],
            'resolution' : resolution,
            'physical extent' : physical_extent,
            'view extent' : view_extent,
            'dx' : dx,
            'dy' : dy,
            'x' : x,
            'y' : y,
            'h' : h,
            'r' : r,
            'A' : A,
        }
        

    def calculate(self):
        if self.gpus: image, view_extent = self.get_image_gpus()
        else: image, view_extent = self.get_image_cpus()
        
        if self.log:
            warnings.filterwarnings(action = 'ignore')
            image = np.log10(image)
            warnings.resetwarnings()
        else: image[image == 0] = np.nan
        
        self.set_extent(view_extent)
        self.set_array(image)


    def get_image_gpus(self):
        @cuda.jit('float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], int32, float64, float64, float64, float64, int32, int32, float64[:]')
        def kernel(image, x, y, h, r, A, N, xmin, ymin, dx, dy, Nx, Ny, kernel_table):
            i = cuda.grid(1)
            if i < N:
                xi = x[i]
                yi = y[i]
                ri = r[i]
                ri2 = ri * ri
                Ai = A[i]
                invh2i = 1 / h[i]**2
                
                ctab = float(len(kernel_table) - 1)
                ctabinvri = ctab / ri # same as / (compact_support*h)
                
                imin = max(int(math.floor((xi - ri - xmin) / dx)), 0)
                imax = min(int(math.ceil((xi + ri - xmin) / dx)), Nx)
                jmin = max(int(math.floor((yi - ri - ymin) / dy)), 0)
                jmax = min(int(math.ceil((yi + ri - ymin) / dy)), Ny)

                for ii in range(imin, imax):
                    xpos = xmin + ii * dx
                    deltax2 = (xpos - xi)**2
                    for jj in range(jmin, jmax):
                        ypos = ymin + jj * dy
                        r2 = deltax2 + (ypos - yi)**2
                        if r2 >= ri2: continue
                        index = int(ctabinvri * math.sqrt(r2))
                        cuda.atomic.add(
                            image,
                            jj + ii * Ny,
                            Ai * kernel_table[index] * invh2i,
                        )
        
        properties = self.get_image_properties()
        resolution = properties['resolution']
        xmin, xmax, ymin, ymax = properties['physical extent']
        
        image = np.zeros(int(np.prod(resolution)))
        with starsmashertools.helpers.gpujob.GPUJob(
                [
                    properties['x'], properties['y'], properties['h'],
                    properties['r'], properties['A'], properties['ntot'],
                    xmin, ymin, properties['dx'], properties['dy'],
                    resolution[0], resolution[1],
                    self.kernel_function.table,
                ],
                [image],
                kernel = kernel,
        ) as result:
            image = result
        
        image = np.asarray(image).reshape(resolution).T
        return image, properties['view extent']

    
        
    def get_image_cpus(self):
        properties = self.get_image_properties()
        ntot = properties['ntot']
        resolution = properties['resolution']
        xmin, xmax, ymin, ymax = properties['physical extent']
        dx = properties['dx']
        dy = properties['dy']
        r = properties['r']
        x = properties['x']
        y = properties['y']
        h = properties['h']
        r = properties['r']
        A = properties['A']
        
        #@profile
        def get(image, resolution, x, y, h, r, A, kernel_function, finished_queue, error_queue):
            try:
                # Load the kernel function's table into memory
                wint = kernel_function.table
                ctab = float(len(wint) - 1) / kernel_function.compact_support
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ijminmax_arr = np.column_stack((
                        # imin
                        np.maximum(
                            np.floor((x - r - xmin) / dx).astype(int),
                            np.zeros(len(x), dtype = int),
                        ),
                        # imax
                        np.minimum(
                            np.ceil((x + r - xmin) / dx).astype(int),
                            np.full(len(x), resolution[0], dtype = int),
                        ),
                        # jmin
                        np.maximum(
                            np.floor((y - r - ymin) / dy).astype(int),
                            np.zeros(len(x), dtype = int),
                        ),
                        # jmax
                        np.minimum(
                            np.ceil((y + r - ymin) / dy).astype(int),
                            np.full(len(x), resolution[1], dtype = int),
                        ),
                    ))

                # Use only imin,imax,jmin,jmax arguments which result in a non-empty slice
                mask = (ijminmax_arr[:,0] < ijminmax_arr[:,1]) & (ijminmax_arr[:,2] < ijminmax_arr[:,3])

                if not mask.any():
                    return

                # Ignore out-of-bounds particles
                mask = mask & (((x+r > xmin) & (x-r < xmax)) | ((y+r > ymin) & (y-r < ymin)))
                
                x = x[mask]
                y = y[mask]
                h = h[mask]
                r = r[mask]
                A = A[mask]
                
                xpos = np.linspace(xmin, xmax, resolution[0] + 1)[:-1] + 0.5*dx
                ypos = np.linspace(ymin, ymax, resolution[1] + 1)[:-1] + 0.5*dy
                
                ctabinvh = ctab / h
                invh2 = 1. / h**2
                size2 = r**2
                
                has_lock = hasattr(image, 'get_lock')
                
                dx2 = (xpos[:,None] - x)**2
                idx_xs = dx2 < size2
                for i, (idx_x, dx2_x) in enumerate(zip(idx_xs,dx2)):
                    if not idx_x.any(): continue
                    dx2_x = dx2_x[idx_x]
                    
                    dy2 = (ypos[:,None]-y[idx_x])**2
                    idx_ys = dy2 < size2[idx_x]
                    for j, (idx_y, dy2_y) in enumerate(zip(idx_ys,dy2)):
                        if not idx_y.any(): continue
                        dr2 = dx2_x[idx_y] + dy2_y[idx_y]
                        
                        idx_r = dr2 < size2[idx_x][idx_y]
                        if not idx_r.any(): continue

                        indices = (ctabinvh[idx_x][idx_y][idx_r] * np.sqrt(dr2[idx_r])).astype(int, copy = False)

                        try: image.acquire()
                        except: pass
                        image[j + i * resolution[1]] += np.dot(A[idx_x][idx_y][idx_r], wint[indices] * invh2[idx_x][idx_y][idx_r])
                        try: image.release()
                        except: pass
                finished_queue.put(None)
            except:
                error_queue.put(traceback.format_exc())

        manager = multiprocessing.Manager()
        error_queue = manager.Queue()
        finished_queue = manager.Queue()
        
        if self.parallel:
            image = multiprocessing.Array('d', [0]*int(np.prod(resolution)), lock = True)
            
            nprocs = starsmashertools.helpers.asynchronous.max_processes
            chunks = np.array_split(np.arange(ntot), nprocs)
            processes = [starsmashertools.helpers.asynchronous.Process(
                target = get,
                args = (
                    image,
                    resolution,
                    x[chunk],
                    y[chunk],
                    h[chunk],
                    r[chunk],
                    A[chunk],
                    self.kernel_function,
                    finished_queue,
                    error_queue,
                ),
                daemon = True,
            ) for chunk in chunks]
            
            try:
                for process in processes: process.start()

                ndone = 0
                while ndone < len(processes):
                    if not error_queue.empty():
                        print(error_queue.get())
                        for process in processes: process.terminate()
                        quit(1)

                    if not finished_queue.empty():
                        finished_queue.get()
                        ndone += 1
                        continue

                    time.sleep(1.e-4)
                for process in processes:
                    process.join()
                    process.terminate()
            except:
                for process in processes: process.terminate()
                raise
        else:
            image = np.zeros(int(np.prod(resolution)))
            get(
                image,
                resolution,
                x,
                y,
                h,
                r,
                A,
                self.kernel_function,
                finished_queue,
                error_queue,
            )
            if not error_queue.empty():
                print(error_queue.get())
                quit(1)
            
        image = np.asarray(image).reshape(resolution).T
        return image, properties['view extent']
        
    
    def get_resolution(self):
        window_extent = self.axes.get_window_extent(
            renderer = self.axes.get_figure().canvas.get_renderer(),
        )
        resolution = [int(window_extent.width), int(window_extent.height)]
        if self.max_resolution is not None:
            resolution[0] = min(resolution[0], self.max_resolution[0])
            resolution[1] = min(resolution[1], self.max_resolution[1])
        return resolution
    
    def draw(self, *args, **kwargs):
        # Check if we need to recalculate the image
        if self.get_array().shape != self.get_resolution():
            self.calculate()
        return super(ColumnDensity, self).draw(*args, **kwargs)


    
    
