import numpy as np
from starsmashertools.helpers.apidecorator import api
import starsmashertools.helpers.argumentenforcer
import starsmashertools.lib.output
import starsmashertools.helpers.readonlydict

# Important note:
#    I did try writing GPU code for this, but it turns out that serial CPU code
#    is fastest in this case. The reason is likely because particle information
#    gets stored on low-level caches. When written in parallel, each process
#    needs to access the particle information which floods the read head with
#    too many instructions and so many process just sit and wait. On each of the
#    image rays we do 1 big reduction operation to get the interacting particles
#    and then calculate the optical depth of the ray as needed. I never saw the
#    number of interacting particles exceed ~1% the total number of particles.
#    That small fraction of particles sits nicely on the lowest cache (L3?), so
#    sequential access is very fast.
#
#    The alternative in GPU form involves an N^2 loop on every thread, which
#    loses big time to serial CPU. There may be some neat tricks I haven't had
#    time to implement which could speed things up, such as multiprocessing with
#    shared arrays or perhaps shared GPU arrays (see https://numba.readthedocs.io/en/stable/cuda/memory.html#shared-memory-and-thread-synchronization)
#    or perhaps using constant memory, which usually has the fastest access
#    times. NumPy already uses SIMD operations for the particle reductions on
#    each ray, so I doubt that can be sped up much. The main bottleneck is the
#    get_flux function.


class FluxFinder(object):
    """
    A class for working with radiative flux emitted from a simulation. Paper in
    progress.
    """

    required_keys = [
        'x', 'y', 'z', 'radius', 'hp', 
        'tau', 'flux', 'ID',
    ]
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(
            self,
            output : starsmashertools.lib.output.Output,
            resolution : list | tuple = (200, 200),
            tau_ray_max : float | int = 20.,
            viewing_angle : list | tuple = (0., 0., 0.),
            include_in_rotation : list | tuple = [],
            extent : list | tuple | np.ndarray | type(None) = None,
            flux_weighted_averages : list | tuple = [],
    ):
        import copy
        
        # Make copies of the outputs so that we can rotate them etc. without
        # worrying about messing them up for other functions
        self._original_output = copy.deepcopy(output)
        self.output = self._original_output

        # Input parameters
        self.resolution = resolution
        self.tau_ray_max = tau_ray_max
        self.viewing_angle = viewing_angle
        self.include_in_rotation = include_in_rotation
        self.extent = extent
        self.flux_weighted_averages = flux_weighted_averages
        self.contributing_particle_IDs = []

        self._contributors = None
        self.result = None

    def check_outputs(self):
        """
        Check if the output object has all the required keys.

        Returns
        -------
        bool
            True if all required keys are present and False otherwise.
        """
        for key in FluxFinder.required_keys:
            if key not in self.output.keys():
                return False
        return True

    def prepare_output(self):
        """
        Perform all necessary transformations to the particle data in
        preparation for :func:`~.get`.
        """
        import inspect
        import copy
        import starsmashertools.preferences
        import warnings
        
        # Start with a clean copy of the data
        self.output = copy.deepcopy(self._original_output)

        self._contributors = np.full(len(self.output['x']), False)

        # Rotate the particles
        # Check the default value of each one's rotate keys
        signature = inspect.signature(self.output.rotate)
        default_keys = signature.parameters['keys'].default
        self.output.rotate(
            keys = default_keys + list(self.include_in_rotation),
            # Negative values b/c it's what the user expects
            xangle = -self.viewing_angle[0],
            yangle = -self.viewing_angle[1],
            zangle = -self.viewing_angle[2],
        )

        # Set the domain
        xmin = np.nanmin(self.output['x'] - 2 * self.output['hp'])
        xmax = np.nanmax(self.output['x'] + 2 * self.output['hp'])
        ymin = np.nanmin(self.output['y'] - 2 * self.output['hp'])
        ymax = np.nanmax(self.output['y'] + 2 * self.output['hp'])

        center = np.array([0.5*(xmin + xmax), 0.5*(ymin + ymax)])

        max_coord = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
        xmax = max_coord
        ymax = max_coord
        xmin = -max_coord
        ymin = -max_coord

        extent = [xmin, xmax, ymin, ymax]
        
        if self.extent is None: self.extent = extent
        else:
            for i, (current, new) in enumerate(zip(self.extent, extent)):
                if current is None: self.extent[i] = new

        if self.output.simulation['ncooling'] in [2, 3]:
            T_dust_min, T_dust = starsmashertools.preferences.get_default(
                'FluxFinder', 'dust temperature range',
                throw_error = True,
            )
            kappa_dust = starsmashertools.preferences.get_default(
                'FluxFinder', 'dust opacity',
                throw_error = True,
            )
            tau_particle_cutoff = starsmashertools.preferences.get_default(
                'FluxFinder', 'tau particle cutoff',
                throw_error = True,
            )
            c_speed = 2.99792458e10
            a_const = 7.565767e-15
            
            self.output.mask(self.output['tau'] >= tau_particle_cutoff)
            
            units = self.output.simulation.units
            
            m = self.output['am']
            h = self.output['hp']
            rho = self.output['rho'] * float(units['rho'])
            u = self.output['u'] * float(units['u'])
            uraddot_emerg = self.output['dEemergdt'] * float(units['dEemergdt'])
            uraddot_diff = self.output['dEdiffdt'] * float(units['dEdiffdt'])
            kappa = self.output['popacity'] * float(units['popacity'])
            tau = self.output['tau']
            temp = self.output['temperatures']
            
            flux = np.zeros(m.shape)

            rloc = np.full(tau.shape, np.nan)

            idx = np.logical_and(
                np.logical_and(temp < T_dust, temp > T_dust_min),
                kappa < kappa_dust,
            )
            if np.any(idx):
                kappa[idx] = kappa_dust


            idx = uraddot_emerg < uraddot_diff
            if np.any(idx):
                rloc[idx] = 2 * h[idx]
                rho[idx] = m[idx]*float(units.mass) / (rloc[idx]*float(units.length))**3
                tau[idx] = rho[idx] * kappa[idx] * rloc[idx] * float(units.length)
                flux_em = uraddot_emerg[idx] / (4 * np.pi * (rloc[idx]*float(units.length))**2)
                flux[idx] = flux_em
            if np.any(~idx):
                rloc[~idx] = (m[~idx]*float(units.mass) / (4/3.*np.pi * rho[~idx]))**(0.33333) / float(units.length)
                tau[~idx] = rho[~idx] * kappa[~idx] * rloc[~idx] * float(units.length)
                flux_em = uraddot_emerg[~idx]/(4 * np.pi * (rloc[~idx] * float(units.length))**2)
                ad = 4 * 3.1415 * (rloc[~idx]*float(units.length))**2
                
                warnings.filterwarnings(action = 'ignore')
                dd = c_speed / (kappa[~idx] * rho[~idx])
                warnings.resetwarnings()
                
                tdiffloc = kappa[~idx] * rho[~idx] * (rloc[~idx] * float(units.length))**2 / c_speed
                gradt = a_const * temp[~idx]**4 / (rloc[~idx] * float(units.length))
                warnings.filterwarnings(action = 'ignore')
                dedt = ad/3.*dd*gradt/(m[~idx] * float(units.mass))
                warnings.resetwarnings()
                
                flux[~idx]=dedt*m[~idx]*float(units.mass)/(4 * np.pi * (rloc[~idx]*float(units.length))**2)
                flux[~idx] = np.minimum(flux[~idx], flux_em)
                
            self.output['opacity'] = kappa
            self.output['radius'] = rloc
            self.output['rho'] = rho
            self.output['tau'] = tau
            self.output['flux'] = flux

    def get_flux(
            self,
            z : np.ndarray,
            r2 : np.ndarray,
            drprime2 : np.ndarray,
            kapparho : np.ndarray,
            flux : np.ndarray,
            IDs : np.ndarray,
            tau_ray_max : float | int,
            flux_weighted_averages = [],
    ):
        total_flux = 0.
        weighted_averages = [0.] * len(flux_weighted_averages)
        
        # Obtain the interacting particles' depths
        dz = np.sqrt(r2 - drprime2)
        front = z + dz
        back = z - dz
        
        # Sort the particles by their fronts, in order of farthest-away
        # particles to closest wrt observer
        idx_sorted = np.argsort(front)
        
        if len(front) > 1:
            # Work on all the interacting particles except the surface particle
            # to get the total flux
            _fronts = np.full(len(front), np.nan)
            for i in range(len(front) - 1):
                # Get the "back side" of each particle. Note that the ray must
                # start at the surface of particle i
                _fronts[i + 1:] = front[idx_sorted[i]] # Surface of particle i
                _max = np.maximum(
                    back[idx_sorted[i+1:]], # Back-side of interacting particles
                    _fronts[i + 1:],        # Surface of particle i
                )

                # Get the distances that the ray travels through the kernels of
                # the interacting particles
                ds = front[idx_sorted[i + 1:]] - _max
                
                # Obtain the total optical depth of the ray. We do a dot product
                # here because each contribution to tau is kappa*rho*ds, and
                # thus tau_ray = kappa_0*rho_0*ds_0 + kappa_1*rho_1*ds_1 + ...,
                # which is the same as vector1 * vector2 where vector1 =
                # (kappa_0*rho_0, kappa_1*rho_1, ... ) and vector2 =
                # (ds_0, ds_1, ... ). The benefit is that NumPy's dot product is
                # faster than its sum operation for vectors that can fit in the
                # L3 cache. Inside of this function our vector lengths are all
                # very small (<1000 elements at most).
                # (https://stackoverflow.com/a/75556529/4954083)
                tau_ray = np.dot(kapparho[idx_sorted[i + 1:]], ds)
                
                if tau_ray <= tau_ray_max:
                    if tau_ray < 0: raise Exception("Bad tau_ray")
                    exptau = np.exp(-tau_ray)
                    f = flux[idx_sorted[i]] * exptau

                    for _i, val in enumerate(flux_weighted_averages):
                        weighted_averages[_i] += val[idx_sorted[i]] * f
                    
                    total_flux += f

                    self._contributors[IDs[idx_sorted[i]]] = True
                    #cID = IDs[idx_sorted[i]]
                    #self.contributing_particle_IDs = list(set(
                    #    self.contributing_particle_IDs + [cID]
                    #))
                    #if cID not in self.contributing_particle_IDs:
                    #    self.contributing_particle_IDs += [cID]
        
        # Add the surface particle's flux
        f = flux[idx_sorted[-1]]
        for _i, val in enumerate(flux_weighted_averages):
            weighted_averages[_i] += val[idx_sorted[-1]] * f
        total_flux += f
        self._contributors[IDs[idx_sorted[-1]]] = True
        #cID = IDs[idx_sorted[-1]]
        #self.contributing_particle_IDs = list(set(
        #    self.contributing_particle_IDs + [cID]
        #))
        #if cID not in self.contributing_particle_IDs:
        #    self.contributing_particle_IDs += [cID]
        
        return total_flux, weighted_averages
    
    @api
    def get(self):
        """
        Obtain the radiative flux using the current settings.

        Returns
        -------
        np.ndarray
            A 2D array of flux values 
        """
        self.result = None
        self.prepare_output()
        
        flux = np.zeros(np.asarray(self.resolution) + 1)
        
        xmin, xmax, ymin, ymax = self.extent
        
        dx = (xmax - xmin) / self.resolution[0]
        dy = (ymax - ymin) / self.resolution[1]
        
        x = self.output['x']
        y = self.output['y']
        z = self.output['z']
        rloc = self.output['radius']
        pflux = self.output['flux']
        tau = self.output['tau']
        
        i_array = np.arange(self.resolution[0])
        j_array = np.arange(self.resolution[1])
        
        xlocs = xmin + dx * i_array
        ylocs = ymin + dy * j_array
        
        deltax2_array = (xlocs[:,None] - x)**2
        
        rloc2 = rloc * rloc

        kapparho = tau / rloc

        _IDs = np.arange(len(kapparho)) # Not the actual particle IDs
        IDs = self.output['ID'] # The actual particle IDs
        
        weighted_averages = []
        for key in self.flux_weighted_averages:
            weighted_averages += [np.full(flux.shape, np.nan)]
        
        for i, ii in enumerate(i_array):
            # Determine which particles will interact with the cells in this
            # column (x position)
            interacting_x = deltax2_array[i] < rloc2
            
            if not np.any(interacting_x): continue
            
            interacting_IDs = _IDs[interacting_x]
            y_interacting = y[interacting_x]
            
            deltax2 = deltax2_array[i][interacting_x]
            
            deltay2_array = (ylocs[:,None] - y_interacting)**2
            
            for j, jj in enumerate(j_array):
                drprime2 = deltax2 + deltay2_array[j]
                
                interacting_xy = drprime2 < rloc2[interacting_IDs]
                if not np.any(interacting_xy): continue

                drprime2 = drprime2[interacting_xy]
                interacting = interacting_IDs[interacting_xy]
                
                flux[ii, jj], averages = self.get_flux(
                    z[interacting],
                    rloc2[interacting],
                    drprime2,
                    kapparho[interacting],
                    pflux[interacting],
                    IDs[interacting],
                    self.tau_ray_max,
                    flux_weighted_averages = [arr[interacting] for arr in self.flux_weighted_averages],
                )
                
                for _i, val in enumerate(averages):
                    weighted_averages[_i][ii, jj] = val

        # Store a list of all the contributing particles
        self.contributing_particle_IDs = np.where(self._contributors)[0]
        
        # Complete the weighted averages
        idx = np.logical_and(flux > 0, np.isfinite(flux))
        if np.any(idx):
            for i in range(len(weighted_averages)):
                weighted_averages[i][idx] /= flux[idx]
                weighted_averages[i][~idx] = 0

        return FluxFinder.Result(
            flux,
            weighted_averages,
            fluxfinder = self,
        )

    
        


    class Result(starsmashertools.helpers.readonlydict.ReadOnlyDict, object):
        class FileFormatError(Exception, object): pass
        
        def __init__(
                self,
                flux : np.ndarray,
                weighted_averages : np.ndarray,
                fluxfinder = None,
                output_file = None,
                contributing_particle_IDs = None,
                extent = None,
                tau_ray_max = None,
                viewing_angle = None,
        ):
            import copy
            import warnings

            if fluxfinder is not None:
                output_file = copy.deepcopy(fluxfinder.output.path)
                contributing_particle_IDs = copy.deepcopy(fluxfinder.contributing_particle_IDs)
                extent = copy.deepcopy(fluxfinder.extent)
                tau_ray_max = copy.deepcopy(fluxfinder.tau_ray_max)
                viewing_angle = copy.deepcopy(fluxfinder.viewing_angle)
                
            warnings.filterwarnings(action='ignore')

            log_flux = np.log10(flux)
            log_weighted_averages = [np.log10(arr) for arr in weighted_averages]

            super(FluxFinder.Result, self).__init__({
                'output_file' : output_file,
                'flux' : copy.deepcopy(flux),
                'log_flux' : copy.deepcopy(log_flux),
                'weighted_averages' : copy.deepcopy(weighted_averages),
                'log_weighted_averages' : copy.deepcopy(log_weighted_averages),
                'contributing_particle_IDs' : contributing_particle_IDs,
                'extent' : extent,
                'tau_ray_max' : tau_ray_max,
                'viewing_angle' : viewing_angle,
            })
            warnings.resetwarnings()
            
            self.fluxfinder = fluxfinder
            
        def __getattribute__(self, attr):
            if attr != 'keys':
                if attr in self.keys():
                    return self[attr]
            return super(FluxFinder.Result, self).__getattribute__(attr)

        @property
        def shape(self): return self.flux.shape
        
        @starsmashertools.helpers.argumentenforcer.enforcetypes
        @api
        def save(
                self,
                filename : str | type(None) = None,
                data_format : str = '%15.7E',
                log : bool = False,
        ):
            """
            Save to a file.

            Parameters
            ----------
            filename : str, None, default = None
                The file name to save to. If `None` then the file will be saved 
                in the current working directory as the same name as the output
                file's basename with an added suffix '.flux.json'. If the file
                name ends with '.json' then 
                :func:`starsmashertools.helpers.jsonfile.save` will be used to 
                save the data. Similarly, if the file name ends with '.json.gz'
                or '.json.zip' then the file will be compressed according to 
                :func:`starsmashertools.helpers.jsonfile.save`. Otherwise, the
                file will be written in a human-readable format.

            data_format : str, default = '%15.7E'
                The float string formatter to use for array data when writing a
                non-json formatted file.

            log : bool, default = False
                If `True`, the log10 values will be saved. If `False`, the 
                regular values will be saved.
            """
            import starsmashertools.helpers.path
            import starsmashertools.helpers.jsonfile
            import copy


            if filename is None:
                basename = starsmashertools.helpers.path.basename(
                    self.output_file,
                )
                filename = starsmashertools.helpers.path.join(
                    starsmashertools.helpers.path.getcwd(),
                    basename + '.json',
                )

            if (filename.endswith('.json') or
                filename.endswith('.json.gz') or
                filename.endswith('.json.zip')):
                obj = {}
                for key, val in self.items():
                    obj[key] = val
                obj['log'] = log
                starsmashertools.helpers.jsonfile.save(filename, obj)
                return

            def write_array(f, array, fmt, name):
                f.write('%s\n' % name)
                f.write(('vmin: '+data_format+'\n') % get_vmin(array))
                f.write(('vmax: '+data_format+'\n') % get_vmax(array))
                total_fmt = ' '.join([fmt]*array.shape[0])+'\n'
                for i, row in enumerate(array):
                    f.write(total_fmt % tuple(row))

            def get_vmin(array):
                idx = np.isfinite(array)
                if not np.any(idx): return -float('inf')
                return np.nanmin(array[idx])

            def get_vmax(array):
                idx = np.isfinite(array)
                if not np.any(idx): return float('inf')
                return np.nanmax(array[idx])

            flux = self.log_flux if log else self.flux
            weighted_averages = self.log_weighted_averages if log else self.weighted_averages

            # Otherwise write a human-readable file
            labels = ['xmin','xmax','ymin','ymax']
            viewing_labels = ['xangle', 'yangle', 'zangle']
            try:
                with open(filename, 'w') as f:
                    f.write(self.output_file + '\n')
                    f.write('log: ' + str(log) + '\n')
                    for item, label in zip(self.viewing_angle, viewing_labels):
                        f.write((label + ': ' + data_format + '\n') % item)

                    for item, label in zip(self.extent, labels):
                        f.write((label + ': ' + data_format + '\n') % item)
                    f.write('Nx: %d\n' % flux.shape[0])
                    f.write('Ny: %d\n' % flux.shape[1])

                    f.write('\n')
                    write_array(f, flux, data_format, 'Flux')

                    for i, weighted_average in enumerate(weighted_averages):
                        f.write('\n')
                        write_array(
                            f,
                            weighted_average,
                            data_format,
                            'Weighted Average %d' % i,
                        )

                    f.write('\ncontributing_particle_IDs:\n')

                    maxID = max(self.contributing_particle_IDs)
                    length = len(str(maxID))
                    fmt = ' %' + str(length) + 'd'
                    i = 0
                    for ID in self.contributing_particle_IDs:
                        f.write(fmt % ID)
                        i += 1
                        if i == 12:
                            f.write('\n')
                            i = 0
            except:
                if starsmashertools.helpers.path.exists(filename):
                    starsmashertools.helpers.path.remove(filename)
                raise
                
        @staticmethod
        @starsmashertools.helpers.argumentenforcer.enforcetypes
        @api
        def load(
                filename : str,
        ):
            import starsmashertools.helpers.path
            import starsmashertools.helpers.jsonfile
            import starsmashertools.helpers.string
            
            if (filename.endswith('.json') or
                filename.endswith('.json.gz') or
                filename.endswith('.json.zip')):
                obj = starsmashertools.helpers.jsonfile.load(filename)
                return FluxFinder.Result(
                    obj['flux'],
                    obj['weighted_averages'],
                    output_file = obj['output_file'],
                    contributing_particle_IDs = obj['contributing_particle_IDs'],
                    extent = obj['extent'],
                    tau_ray_max = obj['tau_ray_max'],
                    viewing_angle = obj['viewing_angle'],
                )

            def read_array(f, Nx, Ny, log):
                name = f.readline().strip()
                vmin = starsmashertools.helpers.string.parse(
                    f.readline().split(':')[-1].strip(),
                )
                vmax = starsmashertools.helpers.string.parse(
                    f.readline().split(':')[-1].strip(),
                )
                data = np.empty((Nx, Ny), dtype=object)
                for i in range(Ny):
                    data[i] = f.readline().strip().split()
                data = data.astype(float)
                if log: data = 10.**data
                return data, vmin, vmax, name

            labels = ['xmin','xmax','ymin','ymax']
            viewing_labels = ['xangle', 'yangle', 'zangle']
            extent = [None, None, None, None]
            viewing_angle = [None, None, None]
            with open(filename, 'r') as f:
                try:
                    output_file = f.readline().strip()
                    log = starsmashertools.helpers.string.parse(
                        f.readline().split('log:')[-1].strip(),
                    )
                    for i, label in enumerate(viewing_labels):
                        line = f.readline().split(label + ':')[-1].strip()
                        viewing_angle[i] = starsmashertools.helpers.string.parse(line)
                    
                    for i, label in enumerate(labels):
                        line = f.readline().split(label + ':')[-1].strip()
                        extent[i] = starsmashertools.helpers.string.parse(line)
                    Nx = starsmashertools.helpers.string.parse(
                        f.readline().split('Nx:')[-1].strip(),
                    )
                    Ny = starsmashertools.helpers.string.parse(
                        f.readline().split('Ny:')[-1].strip(),
                    )
                except Exception as e:
                    raise FluxFinder.Result.FileFormatError(filename) from e

                newline = f.readline().strip() # newline
                if newline:
                    raise FluxFinder.Result.FileFormatError(filename)

                # flux array
                flux,vmin,vmax,name = read_array(f, Nx, Ny, log)

                weighted_averages = {}
                start = f.tell()
                while True:
                    start = f.tell()
                    try:
                        newline = f.readline()
                        if newline.strip():
                            raise FluxFinder.Result.FileFormatError(filename)
                        data,vmin,vmax,name = read_array(f, Nx, Ny, log)
                        if 'Weighted Average' not in name:
                            raise FluxFinder.Result.FileFormatError(filename)
                        number = starsmashertools.helpers.string.parse(
                            name.split('Weighted Average')[-1].strip(),
                        )
                        weighted_averages[number] = data
                    except Exception as e:
                        if isinstance(e, FluxFinder.Result.FileFormatError):
                            raise(e)
                        break

                # Finally, read the contributing particles
                f.seek(start)
                newline = f.readline().strip()
                if newline:
                    raise FluxFinder.Result.FileFormatError(filename)

                try:
                    f.readline().split('contributing_particle_IDs:')
                except Exception as e:
                    raise FluxFinder.Result.FileFormatError(filename) from e

                contributing_particle_IDs = []
                for line in f:
                    contributing_particle_IDs += line.strip().split()
                contributing_particle_IDs = np.array(
                    contributing_particle_IDs,
                    dtype = object,
                ).astype(int).tolist()
                

            # Now that the file has been read we just need to sort the weighted
            # averages (if there are any)

            IDs = list(weighted_averages.keys())
            order = np.argsort(IDs)
            wa = []
            for ID in order:
                wa += [weighted_averages[ID]]
            weighted_averages = wa
            
            return FluxFinder.Result(
                flux,
                weighted_averages,
                fluxfinder = None,
                output_file = output_file,
                contributing_particle_IDs = contributing_particle_IDs,
                extent = extent,
                tau_ray_max = tau_ray_max,
                viewing_angle = viewing_angle,
            )
