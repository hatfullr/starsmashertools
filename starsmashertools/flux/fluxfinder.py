import numpy as np
from starsmashertools.helpers.apidecorator import api
import starsmashertools.helpers.argumentenforcer
import starsmashertools.lib.output
import starsmashertools.helpers.readonlydict
import matplotlib.axes
import pickle

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
            verbose : bool = False,
    ):
        import copy
        
        # Input parameters
        self.resolution = resolution
        self.tau_ray_max = tau_ray_max
        self.viewing_angle = viewing_angle
        self.include_in_rotation = include_in_rotation
        self.extent = extent
        self.flux_weighted_averages = flux_weighted_averages
        self.verbose = verbose
        
        # Make copies of the outputs so that we can rotate them etc. without
        # worrying about messing them up for other functions
        self._original_output = copy.deepcopy(output)
        self._output = None
        self.output = output # Calls _initialize

    @property
    def output(self): return self._output

    @output.setter
    def output(self, value):
        import copy
        self._original_output = copy.deepcopy(value)
        self._output = copy.deepcopy(self._original_output)
        self._initialize()

    # For pickling
    def __getstate__(self):
        return {
            'output' : pickle.dumps(self._original_output),
            'resolution' : self.resolution,
            'tau_ray_max' : self.tau_ray_max,
            'viewing_angle' : self.viewing_angle,
            'include_in_rotation' : self.include_in_rotation,
            'extent' : self.extent,
            'flux_weighted_averages' : self.flux_weighted_averages,
            'verbose' : self.verbose,
        }
    
    def __setstate__(self, data):
        import copy
        self._original_output = pickle.loads(data['output'])
        self.resolution = data['resolution']
        self.tau_ray_max = data['tau_ray_max']
        self.viewing_angle = data['viewing_angle']
        self.include_in_rotation = data['include_in_rotation']
        self.extent = data['extent']
        self.flux_weighted_averages = data['flux_weighted_averages']
        self.verbose = data['verbose']
        self.output = copy.deepcopy(self._original_output)
        
    def print_progress(self, progress):
        import starsmashertools.helpers.string
        if not self.verbose: return
        print(starsmashertools.helpers.string.get_progress_string(
            progress * 100.,
        ))
    
    def _initialize(self):
        self._contributors = np.full(len(self.output['x']), False)
        self.contributing_particle_IDs = []

        self._setup_units()
        self.prepare_output()
        self._setup_viewing_angle()
        self._setup_extent()

        self.progress = 0.

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

    def _setup_units(self):
        units = self.output.simulation.units
        self._flux_unit = float(units.flux)
        self._temperature_unit = float(units.temperature)
        self._opacity_unit = float(units.opacity)
        self._c = float(units.c)
        self._a = float(units.a)

    def _setup_extent(self):
        import numpy as np
        if self.extent is not None and None not in self.extent: return
        
        x = self.output['x']
        y = self.output['y']
        r = self.output['radius']
        xmin = np.nanmin(x - r)
        xmax = np.nanmax(x + r)
        ymin = np.nanmin(y - r)
        ymax = np.nanmax(y + r)

        """
        center = np.array([0.5*(xmin + xmax), 0.5*(ymin + ymax)])

        max_coord = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax))
        xmax = max_coord
        ymax = max_coord
        xmin = -max_coord
        ymin = -max_coord
        """

        if self.extent is None: self.extent = [None]*4

        # Overwrite only values that are None
        new_extent = [xmin, xmax, ymin, ymax]
        for i, (old, new) in enumerate(zip(self.extent, new_extent)):
            if old is None: self.extent[i] = new

        xmin, xmax, ymin, ymax = self.extent
        
        self.dx = (xmax - xmin) / self.resolution[0] # rsun
        self.dy = (ymax - ymin) / self.resolution[1] # rsun

    def _setup_viewing_angle(self):
        import inspect
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

    def prepare_output(self):
        """
        Perform all necessary transformations to the particle data in
        preparation for :func:`~.get`.
        """
        import copy
        import starsmashertools.preferences
        import warnings

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
            
            self.output.mask(self.output['tau'] >= tau_particle_cutoff)
            
            T_dust_min /= self._temperature_unit # code units
            T_dust /= self._temperature_unit # code units
            kappa_dust /= self._opacity_unit # code units
            
            m = self.output['am'] # msun
            h = self.output['hp'] # rsun
            rho = self.output['rho'] # code units
            u = self.output['u'] # code units
            uraddot_emerg = self.output['dEemergdt'] # code units
            uraddot_diff = self.output['dEdiffdt'] # code units
            if self.output.simulation['ncooling'] == 2:
                uraddotcool = self.output['uraddot'] # code units
            elif self.output.simulation['ncooling'] == 3:
                uraddotcool = self.output['uraddotcool'] # code units
            else:
                raise NotImplementedError("FluxFinder can only work for ncooling = 2 or 3, not ncooling = %d" % self.output.simulation['ncooling'])
            
            temp = self.output['temperatures'] # code units
            flux = np.zeros(m.shape)
            rloc = np.full(m.shape, np.nan)

            # Set rlocs first
            idx = uraddot_emerg < uraddot_diff
            if np.any(idx):
                rloc[idx] = 2. * h[idx]
            rloc[~idx] = pow(m[~idx]/(4/3.*np.pi)/rho[~idx],0.33333) # code units

            # tau = kappa*rho*ds, where ds = radius of particle
            tau = self.output['tau']
            kappa = self.output['popacity'] # code units
            #kappa = tau / (rho * rloc)

            idx = np.logical_and(
                np.logical_and(temp < T_dust, temp > T_dust_min),
                kappa < kappa_dust,
            )
            if np.any(idx):
                kappa[idx] = kappa_dust # code units

            self.flux_watch_idx = np.array([])

            idx = uraddot_emerg < uraddot_diff
            if np.any(idx):
                #"""
                # this particle is considered in "fluffy" approximation
                #rloc[idx]=2.*h[idx] # rsun
                rho[idx]=m[idx]/(4./3. * np.pi * rloc[idx]**3) # code units
                tau[idx]=rho[idx]*kappa[idx]*rloc[idx] # code units
                flux_em = uraddot_emerg[idx]/(4.*np.pi*rloc[idx]**2) # code units
                flux[idx]=flux_em
                #print("Treat fluffy %d %f %f %f %f %f"%(i, kappa[i], temp[i], tau[i], flux[i], rloc[i]))          
                #"""
                
                """
                rloc[idx] = 2 * h[idx] # rsun
                rho[idx] = m[idx] / rloc[idx]**3 # code units
                tau[idx] = rho[idx] * kappa[idx] * rloc[idx] # code units
                flux_em = uraddot_emerg[idx] / (4 * np.pi * rloc[idx]**2) # code units
                flux[idx] = flux_em # code units
                """
            if np.any(~idx):
                #"""
                # this particle is considered in "dense" approximation
                #rloc[~idx]=pow(m[~idx]/(4/3.*np.pi)/rho[~idx],0.33333) # code units
                tau[~idx]=rho[~idx]*kappa[~idx]*rloc[~idx] # code units
                ad      = 4.*np.pi*rloc[~idx]**2 # code units
                ad_cool = 2.*np.pi*rloc[~idx]**2 # code units
                flux_em = uraddot_emerg[~idx]/ad # code units
                dd=c_speed/kappa[~idx]/rho[~idx] # code units
                tdiffloc=kappa[~idx]*rho[~idx]*rloc[~idx]**2/self._c # code units
                gradt=self._a*temp[~idx]**4/rloc[~idx] # code units
                dedt=ad/3.*dd*gradt/m[~idx] # code units
                flux[~idx]=dedt*m[~idx]/ad # code units
                flux_cool=-uraddotcool[~idx]*m[~idx]/ad # code units
                ratio = -uraddotcool[~idx]/dedt*12. # code units

                idx_ratio = ratio <= 6
                #print(sum(idx_ratio), self.output['ntot'], self.output['ntot'] - sum(idx_ratio))
                if np.any(idx_ratio): # all what is cooled is to goes through the outer hemisphere
                    flux[~idx][idx_ratio] = -uraddotcool[~idx][idx_ratio] * m[~idx][idx_ratio] / ad_cool[idx_ratio] # code units
                if np.any(~idx_ratio): # the particle is getting cooled also to the back, but cools at max through the outer hemisphere
                    flux[~idx][~idx_ratio]=dedt[~idx_ratio]*m[idx][~idx_ratio]/ad[~idx_ratio] # code units
                #"""
                
                """
                rloc[~idx] = (m[~idx] / (4/3.*np.pi * rho[~idx]))**(0.33333) # rsun
                tau[~idx] = rho[~idx] * kappa[~idx] * rloc[~idx] # code units
                flux_em = uraddot_emerg[~idx]/(4 * np.pi * rloc[~idx]**2) # code units
                ad = 4 * 3.1415 * rloc[~idx]**2 # code units
                
                warnings.filterwarnings(action = 'ignore')
                dd = self._c / (kappa[~idx] * rho[~idx]) # code units
                warnings.resetwarnings()
                
                tdiffloc = kappa[~idx] * rho[~idx] * rloc[~idx]**2 / self._c # code units
                gradt = self._a * temp[~idx]**4 / rloc[~idx] # code units
                
                warnings.filterwarnings(action = 'ignore')
                dedt = ad/3.*dd*gradt/m[~idx] # code units
                warnings.resetwarnings()
                
                flux[~idx]=dedt*m[~idx]/(4 * np.pi * rloc[~idx]**2) # code units
                """
                
                flux[~idx] = np.minimum(flux[~idx], flux_em) # code units
                
                
                self.flux_watch_idx = np.arange(len(self.output['x']))[~idx]
                
            self.output['opacity'] = kappa # code units
            self.output['radius'] = rloc # rsun
            self.output['rho'] = rho # code units
            self.output['tau'] = tau # code units
            #self.output['flux'] = np.abs(self.output['uraddotcool']) * self.output['am'] / (4 * np.pi * (2*self.output['hp'])**2) # code units
            self.output['flux'] = flux # code units

    def get_particle_flux(
            self,
            ID, # The particle we want the flux from
            interacting_IDs,# The particles interacting on this (xpos, ypos) ray
            xpos, # rsun
            ypos, # rsun
            zpos, # rsun
    ):
        #import warnings

        return self.output['flux'][ID] # code units
        """
        # These IDs might need their fluxes recalculated specifically for each
        # ray
        if ID in self.flux_watch_idx:
            # We will convert to cgs at the end, because that is what
            # StarSmasher does.
        
            Ti = self.output['temperatures'][ID] # code units
            rhoi = self.output['rho'][ID] # code units
            ui = self.output['u'][ID] # code units
            mi = self.output['am'][ID] # code units
            ri = self.output['radius'][ID] # rsun
            kappai = self.output['opacity'][ID] # code units

            # Find the closest particle to the surface of particle i at our
            # current location
            others = interacting_IDs[interacting_IDs != ID]
            xj = self.output['x'][others] # rsun
            yj = self.output['y'][others] # rsun
            zj = self.output['z'][others] # rsun
            dx = xpos - xj # rsun
            dy = ypos - yj # rsun
            dz = zpos - zj # rsun
            dr2 = dx * dx + dy * dy + dz * dz # rsun
            
            rj = self.output['radius'][others] # rsun

            idx = dr2 < rj * rj # rsun
            if np.any(idx):
                # If there aren't any in idx, then there's no nearby particles
                
                nearest_other = np.argsort(dr2)[0]
                nearest = others[nearest_other]
                Tj = self.output['temperatures'][nearest]

                # Note: this if statement has a significant effect
                if Tj >= Ti: return 0. # For heating events, there is no cooling
                
                xi = self.output['x'][ID] # rsun
                yi = self.output['y'][ID] # rsun
                zi = self.output['z'][ID] # rsun

                dxij = xi - xj[nearest_other] # rsun
                dyij = yi - yj[nearest_other] # rsun
                dzij = zi - zj[nearest_other] # rsun

                drij = np.sqrt(dxij * dxij + dyij * dyij + dzij * dzij) # rsun

                Uradi = self._a * Ti**4 # code units
                Uradj = self._a * Tj**4 # code units
                
                # Similar to dense.f, but with an added negative sign, because
                # dEdiffdt gets multiplied by a negative sign later, but we
                # don't do that in this code, so we account for it here.
                gradt = (Uradi - Uradj) / drij # code units

                # Uncomment to get fluxes exactly the same as the originals
                # (sanity check). You also need to comment out
                # "if Tj >= Ti: return 0." above.
                #gradt = self._a * Ti**4 / ri # code units
            else:
                return self.output['flux'][ID] # code units
                
                
            dEemergdt = self.output['dEemergdt'][ID] # code units
            flux_em = dEemergdt/(4 * np.pi * ri**2) # code units

            ad = 4 * 3.1415 * ri**2 # code units
            warnings.filterwarnings(action = 'ignore')
            dd = c_speed / (kappai * rhoi) # code units
            warnings.resetwarnings()
            dedt = ad/3.*dd*gradt/mi # code units

            # This time the flux is being emitted on the ray toward the
            # observer, so the area associated with the emission is the area of
            # a single pixel on the image. This gives the "apparent" flux.
            Fray = dedt * mi / (self.dx * self.dy) # code units

            Fray = min(Fray, flux_em) # code units
            return Fray # code units
        
        # If this isn't a particle to recalculate the temperature gradient for,
        # just return the flux we already had
        return self.output['flux'][ID] # code units
        """
        
    def get_flux(
            self,
            IDs : np.ndarray, # These are the interacting IDs
            drprime2 : np.ndarray, # rsun
            xpos, # rsun
            ypos, # rsun
            z, # rsun
            r2, # rsun
            kapparho, # code units
            flux, # code units
            flux_weighted_averages = [],
    ):
        total_flux = 0.
        weighted_averages = [0.] * len(flux_weighted_averages)
        total_attenuation = 0.
        total_unattenuated = 0.

        # Obtain the interacting particles' depths
        dz = np.sqrt(r2 - drprime2) # rsun
        front = z + dz # rsun
        back = z - dz # rsun
        
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
                _fronts[i + 1:] = front[idx_sorted[i]] # Surface of particle i # rsun
                _max = np.maximum(
                    back[idx_sorted[i+1:]], # Back-side of interacting particles
                    _fronts[i + 1:],        # Surface of particle i
                ) # rsun

                # Get the distances that the ray travels through the kernels of
                # the interacting particles
                ds = front[idx_sorted[i + 1:]] - _max # rsun
                
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
                
                if tau_ray <= self.tau_ray_max:
                    if tau_ray < 0: raise Exception("Bad tau_ray")
                    exptau = np.exp(-tau_ray)
                    # The temperature gradient (if it has been calculated) is
                    # incorrect. Send this particle off to correct its flux.
                    particle_flux = self.get_particle_flux(
                        IDs[idx_sorted[i]],
                        IDs,
                        xpos, # rsun
                        ypos, # rsun
                        front[idx_sorted[i]], # rsun
                    ) # code units
                    particle_flux *= self._flux_units # cgs
                    f = particle_flux * exptau # cgs
                    #f = flux[idx_sorted[i]] * exptau
                    
                    total_flux += f # cgs
                    total_attenuation += particle_flux - f # cgs
                    total_unattenuated += particle_flux # cgs
                    
                    for _i, val in enumerate(flux_weighted_averages):
                        weighted_averages[_i] += val[idx_sorted[i]] * f # cgs

                    self._contributors[IDs[idx_sorted[i]]] = True
        
        # Add the surface particle's flux. The temperature gradient is correct
        # here.
        f = flux[idx_sorted[-1]] * self._flux_units # cgs
        for _i, val in enumerate(flux_weighted_averages):
            weighted_averages[_i] += val[idx_sorted[-1]] * f # cgs
        total_flux += f # cgs
        self._contributors[IDs[idx_sorted[-1]]] = True
        
        return total_flux, weighted_averages, total_attenuation, total_unattenuated, IDs[idx_sorted[-1]]

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get(
            self,
            parallel : bool = False,
    ):
        """
        Obtain the radiative flux using the current settings.

        Other Parameters
        ----------------
        parallel : bool, default = False
            If `True`, the computation is done in parallel on the CPU. The task
            which would be computed at each pixel of the image is delegated to
            multiple processes.

        Returns
        -------
        :class:`~flux.fluxfinder.FluxFinder.Result`
            The result.
        """
        flux = np.zeros(np.asarray(self.resolution) + 1)
        attenuation = np.zeros(flux.shape)
        unattenuated = np.zeros(flux.shape)
        first_particle = np.full(flux.shape, -1, dtype = int)

        x = self.output['x'] # rsun
        y = self.output['y'] # rsun
        z = self.output['z'] # rsun
        rloc = self.output['radius'] # rsun
        pflux = np.zeros(self.output['ntot'])
        if 'flux' in self.output.keys():
            pflux = self.output['flux'] # code units
        tau = self.output['tau'] # code units
        
        i_array = np.arange(self.resolution[0])
        j_array = np.arange(self.resolution[1])
        
        xlocs = self.extent[0] + self.dx * i_array # rsun
        ylocs = self.extent[2] + self.dy * j_array # rsun
        
        deltax2_array = (xlocs[:,None] - x)**2 # rsun
        
        rloc2 = rloc * rloc # rsun

        kapparho = self.output['opacity'] * self.output['rho'] # code units

        _IDs = np.arange(len(kapparho)) # Not the actual particle IDs
        IDs = self.output['ID'] # The actual particle IDs
        
        weighted_averages = []
        for key in self.flux_weighted_averages:
            weighted_averages += [np.full(flux.shape, np.nan)]
            
        total = len(i_array) * len(j_array)
        current = 0

        indices = []
        all_arguments = []
        all_keywords = []
        for i, ii in enumerate(i_array):
            # Determine which particles will interact with the cells in this
            # column (x position)
            interacting_x = deltax2_array[i] < rloc2
            
            if not np.any(interacting_x): continue
            
            interacting_IDs = _IDs[interacting_x]
            y_interacting = y[interacting_x] # rsun
            
            deltax2 = deltax2_array[i][interacting_x] # rsun
            deltay2_array = (ylocs[:,None] - y_interacting)**2 # rsun
            
            for j, jj in enumerate(j_array):
                drprime2 = deltax2 + deltay2_array[j] # rsun
                
                interacting_xy = drprime2 < rloc2[interacting_IDs]
                if not np.any(interacting_xy): continue

                interacting = interacting_IDs[interacting_xy]
                
                args = (
                    _IDs[interacting],
                    drprime2[interacting_xy], # rsun
                    xlocs[ii], # rsun
                    ylocs[jj], # rsun
                    z[interacting], # rsun
                    rloc2[interacting], # rsun
                    kapparho[interacting], # code units
                    pflux[interacting], # code units
                )
                kwargs = {
                    'flux_weighted_averages' : [arr[interacting] for arr in self.flux_weighted_averages],
                }

                indices += [[ii, jj]]
                all_arguments += [args]
                all_keywords += [kwargs]


        increment = 1. / float(len(all_arguments))
        progress = 0.
        self.print_progress(progress)
        if not parallel:
            for (ii, jj), args, kwargs in zip(indices, all_arguments, all_keywords):
                flux[ii, jj], averages, attenuation[ii, jj], unattenuated[ii,jj], first_particle[ii,jj] = self.get_flux(*args, **kwargs)
                for i, val in enumerate(averages):
                    weighted_averages[i][ii, jj] = val
                progress += increment
                self.print_progress(progress)
        else:
            import starsmashertools.helpers.asynchronous
            
            parallel = starsmashertools.helpers.asynchronous.ParallelFunction(
                self.get_flux,
                args = all_arguments,
                kwargs = all_keywords,
                daemon = True,
            )
            
            parallel.do(
                0.25,
                lambda parallel=parallel: self.print_progress(parallel.get_progress()),
            )
            outputs = parallel.get_output()
            for index, result in zip(indices, outputs):
                ii, jj = index
                flux[ii, jj], averages, attenuation[ii, jj], unattenuated[ii,jj], first_particle[ii,jj] = result
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

        self.print_progress(1.)
        
        return FluxFinder.Result(
            flux, # cgs
            weighted_averages, # cgs
            attenuation, # cgs
            unattenuated, # cgs
            first_particle,
            fluxfinder = self,
        )

    
        


    class Result(starsmashertools.helpers.readonlydict.ReadOnlyDict, object):
        plot_keys = [
            'flux',
            'log_flux',
            'attenuation',
            'log_attenuation',
            'unattenuated',
            'log_unattenuated',
        ]
        
        class FileFormatError(Exception, object): pass
        
        def __init__(
                self,
                flux : np.ndarray,
                weighted_averages : np.ndarray,
                attenuation : np.ndarray,
                unattenuated : np.ndarray,
                first_particle : np.ndarray,
                fluxfinder = None,
                output : starsmashertools.lib.output.Output | type(None) = None,
                contributing_particle_IDs = None,
                extent = None,
                tau_ray_max = None,
                viewing_angle = None,
        ):
            import copy
            import warnings

            if fluxfinder is not None:
                #output_file = copy.deepcopy(fluxfinder.output.path)
                output = copy.deepcopy(fluxfinder.output)
                contributing_particle_IDs = copy.deepcopy(fluxfinder.contributing_particle_IDs)
                extent = copy.deepcopy(fluxfinder.extent)
                tau_ray_max = copy.deepcopy(fluxfinder.tau_ray_max)
                viewing_angle = copy.deepcopy(fluxfinder.viewing_angle)
                
            warnings.filterwarnings(action='ignore')

            log_flux = np.log10(flux)
            log_weighted_averages = [np.log10(arr) for arr in weighted_averages]
            log_attenuation = np.log10(attenuation)
            log_unattenuated = np.log10(unattenuated)
            
            super(FluxFinder.Result, self).__init__({
                'output' : output,
                'simulation' : output.simulation.directory if output is not None else None,
                'flux' : copy.deepcopy(flux),
                'log_flux' : copy.deepcopy(log_flux),
                'attenuation' : copy.deepcopy(attenuation),
                'log_attenuation' : copy.deepcopy(log_attenuation),
                'unattenuated' : copy.deepcopy(unattenuated),
                'log_unattenuated' : copy.deepcopy(log_unattenuated),
                'weighted_averages' : copy.deepcopy(weighted_averages),
                'log_weighted_averages' : copy.deepcopy(log_weighted_averages),
                'first_particle' : copy.deepcopy(first_particle),
                'contributing_particle_IDs' : contributing_particle_IDs,
                'extent' : extent,
                'dx' : (extent[1] - extent[0]) / flux.shape[0],
                'dy' : (extent[3] - extent[2]) / flux.shape[1],
                'resolution' : flux.shape,
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

        @staticmethod
        def from_json(obj):
            import starsmashertools.lib.output
            import starsmashertools
            
            output = None
            if obj['output'] is not None and obj['simulation'] is not None:
                simulation = starsmashertools.get_simulation(obj['simulation'])
                output = starsmashertools.lib.output.Output(
                    obj['output'],
                    simulation,
                )
            
            return FluxFinder.Result(
                obj['flux'],
                obj['weighted_averages'],
                obj['attenuation'],
                obj['unattenuated'],
                obj['first_particle'],
                fluxfinder = None,
                output = output,
                contributing_particle_IDs = obj['contributing_particle_IDs'],
                extent = obj['extent'],
                tau_ray_max = obj['tau_ray_max'],
                viewing_angle = obj['viewing_angle'],
            )

        @starsmashertools.helpers.argumentenforcer.enforcetypes
        @api
        def plot(
                self,
                ax : matplotlib.axes.Axes,
                which : str | np.ndarray = 'log_flux',
                interpolation : str = 'none',
                origin : str = 'lower',
                extent : list | tuple | np.ndarray | type(None) = None,
                **kwargs
        ):
            """
            Use Matplotlib's ``imshow`` method on the given axes to create a
            plot.
            
            Parameters
            ----------
            ax : matplotlib.axes.Axes
                The Matplotlib axes to create the image on.
            
            Other Parameters
            ----------------
            which : str, np.ndarray, default = 'log_flux'
                If a ``str`` type is given, it must be one of the keys in
                :py:attr:`~fluxfinder.FluxFinder.Result.plot_keys`. If a
                ``np.ndarray`` is given it must be a 2D array supported by the
                ``imshow`` function.

            interpolation : str, default = 'none'
                Passed directly to ``imshow``. Use 'none' to minimize distortion
                in the image.

            origin : str, default = 'lower'
                Determines the orientation of the image, which depends on how
                data is stored in the flux array.

            extent : list, tuple, np.ndarray, None, default = None
                If `None` then ``Result['extent']`` will be used. Otherwise, it
                should be a Matplotlib extent collection
                (``[xmin, xmax, ymin, ymax]``, for example).

            **kwargs
                Other keyword arguments are passed directly to ``imshow``.

            Returns
            -------
            matplotlib.image.AxesImage
                The return value from ``imshow``.
            """

            starsmashertools.helpers.argumentenforcer.enforcevalues({
                'which' : FluxFinder.Result.plot_keys,
            })

            if isinstance(which, str): data = self[which]
            else: data = which

            if extent is None: extent = self['extent']
            
            return ax.imshow(
                data.T,
                interpolation = interpolation,
                origin = origin,
                extent = extent,
                **kwargs
            )
        
        
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

            obj = {}
            for key, val in self.items():
                if isinstance(val, starsmashertools.lib.output.Output):
                    obj[key] = val.path
                else: obj[key] = val
                
            starsmashertools.helpers.jsonfile.save(filename, obj)
                
        @staticmethod
        @starsmashertools.helpers.argumentenforcer.enforcetypes
        @api
        def load(
                filename : str,
        ):
            import starsmashertools.helpers.path
            import starsmashertools.helpers.jsonfile
            import starsmashertools.helpers.string

            obj = starsmashertools.helpers.jsonfile.load(filename)
            return FluxFinder.Result.from_json(obj)

        @starsmashertools.helpers.argumentenforcer.enforcetypes
        @api
        def get_particles(
                self,
                x : float | int | np.float_ | np.integer | tuple | list | np.ndarray,
                y : float | int | np.float_ | np.integer | tuple | list | np.ndarray,
        ):
            """
            Obtain a list of particles who are considered as "contributing" on
            the given (x, y) position(s) in the image. A particle is
            "contributing" if a ray extended from the image into the screen
            enters its kernel. These particles make up the entirety of the flux
            at the given image positions or pixel indices.

            Parameters
            ----------
            x : float, int, np.float_, np.integer, tuple, list, np.ndarray
                Float types are physical x positions on the image and integer 
                types are pixel indices. If a tuple, list, or np.ndarray is
                given it must have the same shape as parameter `y`.

            y : float, int, np.float_, np.integer, tuple, list, np.ndarray
                Float types are physical y positions on the image and integer 
                types are pixel indices. If a tuple, list, or np.ndarray is
                given it must have the same shape as parameter `x`.
            
            Returns
            -------
            np.ndarray
                A NumPy array of particle indices.
            """
                
            # Check for errors
            x_is_iter = isinstance(x, (list, tuple, np.ndarray))
            y_is_iter = isinstance(y, (list, tuple, np.ndarray))

            if x_is_iter and not y_is_iter: y = [y]*len(x)
            if not x_is_iter and y_is_iter: x = [x]*len(y)
            if not x_is_iter and not y_is_iter:
                x = [x]
                y = [y]
            
            if isinstance(x, np.ndarray) and len(x.shape) > 1:
                raise ValueError("Argument 'x' must be one-dimensional if it is of type np.ndarray, but its shape is %s" % str(x.shape))
            elif isinstance(x, (list, tuple)):
                for _x in x:
                    if not hasattr(_x, '__iter__'): continue
                    raise Exception("Argument 'x' is an iterable of iterables, which is not supported")

            if isinstance(y, np.ndarray) and len(y.shape) > 1:
                raise ValueError("Argument 'y' must be one-dimensional if it is of type np.ndarray, but its shape is %s" % str(x.shape))
            elif isinstance(y, (list, tuple)):
                for _y in y:
                    if not hasattr(_y, '__iter__'): continue
                    raise Exception("Argument 'y' is an iterable of iterables, which is not supported")
            
            if len(x) != len(y):
                raise ValueError("Arguments 'x' and 'y' must have the same lengths: %d != %d" % (len(x), len(y)))
                
            floats = []
            ints = []
            for i, (_x, _y) in enumerate(zip(x, y)):
                if not isinstance(_x, type(_y)):
                    raise TypeError("Each element of arguments 'x' and 'y' must be the same type as each other, but elements %d have types '%s' and '%s'" % (i, type(_x).__name__, type(_y).__name__))

                if isinstance(_x, (float, np.float_)):
                    floats += [[_x, _y]]
                elif isinstance(_x, (int, np.integer)):
                    ints += [[_x, _y]]
                else:
                    raise TypeError("Unsupported type '%s' in input arrays at element %d" % (type(_x).__name__, i))

            floats = np.asarray(floats, dtype=float)
            ints = np.asarray(ints, dtype=int)

            # Convert the integers to floats
            xmin, xmax, ymin, ymax = self['extent']

            xypos = np.full((len(floats) + len(ints), 2), np.nan)
            xypos[:len(floats)] = floats
            xypos[len(floats):][:,0] = xmin + self['dx'] * ints
            xypos[len(floats):][:,1] = ymin + self['dy'] * ints

            result = []
            
            IDs = self['contributing_particle_IDs']
            with starsmashertools.mask(self['output'], IDs) as masked:
                xy = np.column_stack((masked['x'], masked['y']))
                r2 = masked['radius']**2
                for p in enumerate(xypos):
                    drprime2 = np.sum((xy - p)**2, axis=-1)
                    idx = drprime2 < r2
                    if not np.any(idx): continue
                    result += [[IDs[idx]]]

            return np.asarray(result, dtype=int)
