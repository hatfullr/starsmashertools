import starsmashertools.flux.fluxfinder
import numpy as np
import warnings


class IcoFluxFinder(starsmashertools.flux.fluxfinder.FluxFinder, object):
    """
    A type of FluxFinder which uses the "subdivided icosahedron" angles that are
    used in the modified version of StarSmasher written by Hatfull et al. (2024;
    in progress).
    """
    def __init__(self, *args, **kwargs):
        super(IcoFluxFinder, self).__init__(*args, **kwargs)

        if not self.output.simulation['ncooling'] in [2, 3]:
            raise Exception("IcoFluxFinder can only be used for simulations with ncooling == 2 or 3, not '%d'" % self.output.simulation['ncooling'])

        self.tab_qtau = None

    def _setup_angles(self):
        import starsmashertools.helpers.string
        import copy
        import starsmashertools.math
        
        logfiles = self.output.simulation.get_logfiles()
        if not logfiles:
            raise Exception("IcoFluxFinder requires log files in your simulation: '%s'" % str(self.output.simulation))
        
        logfile = self.output.simulation.get_logfiles()[0]
        
        subdivs = self.output.simulation['icosahedron_subdivisions']
        num_vertices = 10*(2**subdivs)**2 + 2
        
        phrase = 'Using icosahedron angles:\n'
        if phrase not in logfile.header:
            raise Exception("Failed to find phrase '%s' in header of logfile '%s'" % (phrase, logfile.path))

        start = logfile.header.find(phrase) + len(phrase) + 1
        index = copy.deepcopy(start)
        for i in range(num_vertices):
            index = logfile.header.find('\n', index) + 1

        lines = logfile.header[start:index - 1].split('\n')
        angles = np.array([l.split() for l in lines], dtype=object).astype(float)
        self._theta = angles[:,0]
        self._phi = angles[:,1]

        self._unit_vertices = np.column_stack((
            np.sin(self._theta) * np.cos(self._phi),
            np.sin(self._theta) * np.sin(self._phi),
            np.cos(self._theta),
        ))
        
        xrot, yrot, zrot = starsmashertools.math.rotate(
            self._unit_vertices[:,0],
            self._unit_vertices[:,1],
            self._unit_vertices[:,2],
            xangle = -self.viewing_angle[0],
            yangle = -self.viewing_angle[1],
            zangle = -self.viewing_angle[2],
        )
        self._unit_vertices[:,0] = xrot
        self._unit_vertices[:,1] = yrot
        self._unit_vertices[:,2] = zrot

    def _initialize_qtau(self):
        import starsmashertools.helpers.path
        import starsmashertools.helpers.file
        src = starsmashertools.helpers.path.get_src(self.output.simulation.directory)
        filename = starsmashertools.helpers.path.join(src, 'cooling', 'qtau.f')
        
        listening = False
        content = []
        with starsmashertools.helpers.file.open(filename, 'r') as f:
            for line in f:
                if line.strip().startswith('data tab_qtau'):
                    listening = True
                    l = line.strip().replace('data tab_qtau /','')
                    l = l.replace(',', '').replace('d0', '')
                    content += l.split()
                    continue
                if not listening: continue

                l = line.strip().replace('$','').replace(',','')
                l = l.replace('d0','').replace('/','')
                content += l.split()
                if '/' in line: break
                
        self.tab_qtau = np.asarray(content, dtype=object).astype(float)

    def _qtau(self, tau):
        tau = np.asarray(tau)
        result = np.full(tau.shape, np.nan)

        idx1 = tau > 5
        idx2 = np.logical_and(
            tau <= 5,
            tau <= 0.01,
        )
        idx3 = np.logical_and(
            np.logical_and(
                tau <= 5,
                tau > 0.01,
            ),
            tau >= 1,
        )
        idx4 = np.logical_and(
            np.logical_and(
                tau <= 5,
                tau > 0.01,
            ),
            tau < 1,
        )

        if np.any(idx1): result[idx1] = 0.5
        if np.any(idx2): result[idx2] = tau[idx2]
        if np.any(idx3):
            i = (tau[idx3] * 100.).astype(int) - 1
            result[idx3] = self.tab_qtau[i]
        if np.any(idx4):
            i = (tau[idx4] * 100.).astype(int) - 1
            result[idx4] = tau[idx4] * self.tab_qtau[i]/((i + 1) / 100.)
        return result

    def _set_radii(self):
        m = self.output['am'] # msun
        rho = self.output['rho'] # code units

        if self.output.simulation['cooling_type'] == 0: # Fluffy
            self.output['radius'] = 2. * self.output['hp'] # code units

            idx = np.logical_and(
                self.output['u'] != 0,
                self.output['dEemergdt'] >= self.output['dEdiffdt'],
            )

            self.output['radius'] = 2 * self.output['hp']
            
            if np.any(idx):
                self.output['radius'][idx] = (0.75 * m[idx] / (np.pi * rho[idx]))**(0.3333333333333333) # code units
                # tau and everything else gets updated later
            
        elif self.output.simulation['cooling_type'] == 1: # Dense
            self.output['radius'] = (0.75 * m / (np.pi * rho))**(0.3333333333333333) # code units
        else: raise NotImplementedError
        #"""

        self.output['surface area'] = 4 * np.pi * self.output['radius']**2 # code units

        # The optical depths don't need to be updated because those were
        # calculated using this same radius we obtained above.

        # The density doesn't need to be updated because we used the density to
        # calculate the radius above.

        # Because the density isn't changed, we also don't need to change the
        # temperature or opacity.
                    
    def _set_opacities(self):
        # Opacity depends on density and temperature. Thus, if we change the
        # opacity then we need to change the temperature, since we trust the
        # density values from StarSmasher.
        
        # Modifying the opacities necessarily means modifying the optical
        # depths, too, which means we also have to modify dEemergdt
        import starsmashertools.preferences
        import starsmashertools.lib.table
        import copy
        import starsmashertools.helpers.asynchronous
        import warnings
        
        T_dust_min, T_dust = starsmashertools.preferences.get_default(
            'FluxFinder', 'dust temperature range',
            throw_error = True,
        )
        kappa_dust = starsmashertools.preferences.get_default(
            'FluxFinder', 'dust opacity',
            throw_error = True,
        )

        T_dust_min /= self._temperature_unit # code units
        T_dust /= self._temperature_unit # code units
        kappa_dust /= self._opacity_unit # code units
        
        T = self.output['temperatures']
        
        idx = np.logical_and(
            self.output['u'] != 0,
            np.logical_and(
                np.logical_and(
                    T < T_dust,
                    T > T_dust_min,
                ),
                self.output['popacity'] < kappa_dust, # code units
            ),
        )
        
        # Do nothing if there's nothing to worry about
        if not np.any(idx): return

        if self.verbose:
            print("Some opacities may be adjusted according to the dust temperature range and dust opacities specified in starsmashertools.preferences. This will change values of dEemergdt, dEdiffdt, and dEmaxdiffdt, but will not affect the output files on the hard drive.")

        # Make adjustments to everything
        m = self.output['am'][idx]
        u = self.output['u'][idx]
        rho = self.output['rho'][idx]
        rout = self.output['radius'][idx]
        T = T[idx]
        
        self.output['popacity'][idx] = kappa_dust


    def _update_cooling_quantities(self):
        # Only update the particles whose opacities or radii have been changed
        idx = np.logical_and(
            self.output['uraddot'] != 0,
            np.logical_or(
                self.output['popacity'] != self._original_output['popacity'],
                np.logical_and(
                    self.output['u'] != 0,
                    self.output['dEemergdt'] >= self.output['dEdiffdt'],
                ),
            ),
        )

        if not np.any(idx): return
        
        m = self.output['am'][idx]
        u = self.output['u'][idx]
        rho = self.output['rho'][idx]
        rout = self.output['radius'][idx]
        T = self.output['temperatures'][idx]
        
        tdiff = self.output['popacity'][idx] * rho * rout**2 / self._c
        self.output['dEmaxdiffdt'][idx] = u * m / tdiff # code units
        self.output['tau'][idx] = rho * self.output['popacity'][idx] * rout
        A = 4 * np.pi * rout**2
        
        if self.output.simulation['cooling_type'] == 0: # Fluffy
            beta = self._a * T**4 / (rho * u)
            Urad = 0.75 * beta * u * m / (np.pi * rout**3)
        elif self.output.simulation['cooling_type'] == 1: # Dense
            Urad = self._a * T**4
        else:
            raise NotImplementedError
        
        Q = self._qtau(self.output['tau'][idx])
        self.output['dEemergdt'][idx] = 0.5 * A * self._c * Urad * Q
        
        # Print all adjustments that were made
        same = []
        different = []
        for key, orig in self._original_output.items():
            new = self.output[key]
            if isinstance(new, np.ndarray):
                if np.array_equal(orig, new):
                    same += [key]
                    continue
            elif new == orig:
                same += [key]
                continue
            different += [key]

        if not self.verbose: return
        
        if len(different) > 0:
            print()
            print("Remained the same: ", same)
            print()
            print("         difference = new - original")
            print("absolute difference = |difference|")
            print("abs.rel. difference = |difference / original|")
        else:
            print("All quantities remained the same")
        
        warnings.filterwarnings(action = 'ignore')
        for key in different:
            print()
            orig = self._original_output[key]
            new = self.output[key]
            print("'%s' changed" % key)
            diff = new - orig
            absolute = np.abs(diff)
            absrel = np.abs(diff / orig)
            diff = diff[np.isfinite(diff)]
            absolute = absolute[np.isfinite(absolute)]
            absrel = absrel[np.isfinite(absrel)]
            if isinstance(new, np.ndarray):
                idxmin = np.nanargmin(absolute)
                idxmax = np.nanargmax(absolute)
                print("  min difference = %15.7E" % diff[idxmin])
                print("  max difference = %15.7E" % diff[idxmax])
                print("  avg difference = %15.7E" % np.nanmean(diff))
                print("  --")
                print("  min absolute difference = %15.7E" % np.nanmin(absolute))
                print("  max absolute difference = %15.7E" % np.nanmax(absolute))
                print("  avg absolute difference = %15.7E" % np.nanmean(absolute))
                print("  --")
                print("  min abs.rel. difference = %15.7E" % np.nanmin(absrel))
                print("  max abs.rel. difference = %15.7E" % np.nanmax(absrel))
                print("  avg abs.rel. difference = %15.7E" % np.nanmean(absrel))
            else:
                print("  orig =",orig)
                print("  new  =",new)
                print("  difference = %15.7E" % diff)
                print("  absolute difference = %15.7E" % absolute)
                print("  abs.rel. difference = %15.7E" % absrel)
        print()
        warnings.resetwarnings()

    def prepare_output(self):
        import starsmashertools.preferences

        self._setup_angles()
        self._initialize_qtau()

        self._set_radii()
        self._set_opacities()
        self._update_cooling_quantities()
        
        tau_particle_cutoff = starsmashertools.preferences.get_default(
            'FluxFinder', 'tau particle cutoff',
            throw_error = True,
        )
        
        idx = self.output['tau'] >= tau_particle_cutoff
        
        self.output.mask(idx)
        self.output['ntot'] -= np.sum(~idx)

        self.output['opacity'] = self.output['popacity'] # code units

        # These are TOTAL values (over all rays)
        dEemergdt = self.output['dEemergdt'] # code units
        dEmaxdiffdt = self.output['dEmaxdiffdt'] # code units
        dEdiffdt = self.output['dEdiffdt'] # code units

        nothing = np.logical_and(
            dEemergdt == 0,
            dEmaxdiffdt == 0,
        )
        
        uses_emerg = np.logical_and(
            ~nothing,
            dEemergdt < dEdiffdt,
        )
        uses_diff = np.logical_and(
            ~nothing,
            np.logical_and(
                dEdiffdt < dEemergdt,
                dEdiffdt < dEmaxdiffdt,
            ),
        )
        uses_maxdiff = np.logical_and(
            ~nothing,
            np.logical_and(
                dEmaxdiffdt < dEemergdt,
                dEmaxdiffdt == dEdiffdt,
            ),
        )
        
        Nemerg = np.sum(uses_emerg)
        Ndiff = np.sum(uses_diff)
        Nmaxdiff = np.sum(uses_maxdiff)
        Nnothing = np.sum(nothing)
        
        if Nemerg + Ndiff + Nmaxdiff + Nnothing != self.output['ntot']:
            raise Exception("Improper detection of which particles use dEemergdt, dEdiffdt, and dEmaxdiffdt. %d %d %d %d %d %d" % (Nemerg, Ndiff, Nmaxdiff, Nnothing, Nemerg + Ndiff + Nmaxdiff + Nnothing, self.output['ntot']))

        if (np.any(np.logical_and(uses_emerg, uses_diff)) or
            np.any(np.logical_and(uses_emerg, uses_maxdiff)) or
            np.any(np.logical_and(uses_diff, uses_maxdiff))):
            idx1 = np.where(np.logical_and(uses_emerg, uses_diff))[0]
            idx2 = np.where(np.logical_and(uses_emerg, uses_maxdiff))[0]
            idx3 = np.where(np.logical_and(uses_diff, uses_maxdiff))[0]
            print(idx1)
            print(idx2)
            print(idx3)
            raise Exception("Improper detection of which particles use dEemergdt, dEdiffdt, and dEmaxdiffdt. Detected particles that use more than 1")
        
        self._uses_emerg = uses_emerg
        self._uses_diff = uses_diff
        self._uses_maxdiff = uses_maxdiff

        self._Nrays = len(self._unit_vertices)
        self._invNrays = 1. / self._Nrays
    
    def get_closest_angle(
            self,
            xyz : np.ndarray,
            ID : int | np.integer,
    ):
        vertex, idx_closest = self.get_closest_vertex(xyz, ID)
        return self._theta[idx_closest], self._phi[idx_closest]

    def get_vertices(self, ID : int | np.integer):
        return self.output['xyz'][ID] + self.output['radius'][ID] * self._unit_vertices

    #@profile
    def get_closest_vertex(
            self,
            xyz : np.ndarray,
            ID : int | np.integer,
    ):
        vertices = self.get_vertices(ID)
        diff = vertices - xyz
        dr2 = np.sum(diff * diff, axis=-1)
        idx = np.argmin(dr2)
        return vertices[idx], idx

    #@profile
    def get_dEdiffdt(
            self,
            ID,
            interacting_IDs,
            xyzpos, # rsun
    ):
        Ti = self.output['temperatures'][ID] # code units
        kappai = self.output['opacity'][ID] # code units
        rhoi = self.output['rho'][ID] # code units
        ui = self.output['u'][ID] # code units
        
        Ti4 = Ti**4
        warnings.filterwarnings(action = 'ignore')
        betai = self._a * Ti4 / (rhoi * ui) # code units
        warnings.resetwarnings()

        js = interacting_IDs[np.logical_and(
            self.output['u'][interacting_IDs] != 0,
            interacting_IDs != ID,
        )]

        # The other interacting particles will necessarily be the ones we need
        # to check to obtain the temperature gradient.
        vertex, vertex_index = self.get_closest_vertex(xyzpos, ID)
        
        if len(js) == 0: # Handle case if there's no other interacting particles
            idx = None
        else:
            dr2 = np.sum((vertex - self.output['xyz'][js])**2, axis=-1)
            idx = dr2 < self.output['radius'][js]**2
        
        if self.output.simulation['cooling_type'] == 0: # Fluffy
            if idx is not None and np.any(idx): # overlapping kernels exist
                # simple temperature comparison to the closest particle
                closest = np.argmin(dr2[idx])
                Tj = self.output['temperatures'][js][idx][closest] # code units
                if Tj >= Ti: return 0. # Heating event
            
            # The diffusion radiation per cooling ray (if C_s = 1 only 1 time)
            return betai * self.output['dEmaxdiffdt'][ID] * self._invNrays
        
        elif self.output.simulation['cooling_type'] == 1: # Dense
            Uradi = self._a * Ti4 # code units
            Uradj = 0.
            ri = r[ID] # code units
            deltar = ri
            
            if idx is not None and np.any(idx): # overlapping kernels exist
                closest = np.argmin(dr2[idx])
                j = interacting_IDs[idx][closest]
                Tj = self.output['temperatures'][j] # code units
                
                if Tj >= Ti: return 0. # Heating event
                
                Uradj = self._a * Tj**4 # code units
                deltar = np.sqrt(dr2[closest]) # code units
            
            gradUrad = (Uradj - Uradi) / deltar # code units
            
            # The diffusion radiation on a single cooling ray
            D = self._c / (kappai * rhoi) # code units
            dEdiffdt = -(4 * np.pi * ri**2) * D / 3 * self._invNrays * gradUrad # code units
            
            # Sanity check
            epsilon = np.finfo(float).eps
            if (dEdiffdt - (betai * self.output['dEmaxdiffdt'][ID] * self._invNrays) > epsilon and
                dEdiffdt * self._Nrays > 1):
                raise Exception("Too high dEdiffdt: %15.7E, %15.7E\nCheck quantities: %157.E, %15.7E, %15.7E, %d" % (dEdiffdt, self.output['dEmaxdiffdt'][ID] * self._invNrays, betai, betai*self.output['dEmaxdiffdt'][ID] * self._invNrays, epsilon, self._Nrays))

            return dEdiffdt
            
        else:
            raise NotImplementedError("cooling_type '%d' is not implemented" % self.output.simulation['cooling_type'])

    #@profile
    def get_particle_flux(
            self,
            ID, # The particle we want the flux from
            interacting_IDs,# The particles interacting on this (xpos, ypos) ray
            xyzpos, # rsun
            uses_emerg,
            uses_maxdiff,
            #dEemergdt, # code units
            #dEdiffdt, # code units
            #dEmaxdiffdt, # code units
    ):
        areas = self.output['surface area']
        Ai = areas[ID]
        
        #Ai = self.output['surface area'][ID] # code units
        if uses_emerg:
            return self.output['dEemergdt'][ID] / Ai # code units
        if uses_maxdiff:
            return self.output['dEmaxdiffdt'][ID] / Ai # code units
        
        # We need to recalculate dEdiffdt in this case
        dEdiffdt = self.get_dEdiffdt(
            ID,
            interacting_IDs,
            xyzpos,
        )

        if ID in [96852, 97848]:
            print(ID, dEdiffdt, Ai, self._invNrays)
            print()
            quit()

        Aray = Ai * self._invNrays # code units
        return dEdiffdt / Aray # code units
        
    #@profile
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
        # We can probably get rid of this function, but I want to make totally
        # sure that our calculations are correct here.

        xyzpos = np.array([xpos, ypos, np.nan])

        contributors = []
        
        total_flux = 0.
        weighted_averages = [0.] * len(flux_weighted_averages)
        total_attenuation = 0.
        total_unattenuated = 0.
        
        dzs = np.sqrt(r2 - drprime2)
        fronts = z + dzs
        backs = z - dzs
        
        interacting_uses_emerg = self._uses_emerg[IDs]
        interacting_uses_maxdiff = self._uses_maxdiff[IDs]
        
        idx_sorted = np.argsort(fronts)

        if len(fronts) > 1:
            _fronts = np.full(len(fronts), np.nan)
            
            for i in range(len(fronts) - 1):
                index = idx_sorted[i]
                _fronts[i + 1:] = fronts[index]

                _max = np.maximum(
                    backs[idx_sorted[i + 1:]],
                    _fronts[i + 1:],
                ) # rsun

                ds = fronts[idx_sorted[i + 1:]] - _max # rsun

                tau_ray = np.dot(kapparho[idx_sorted[i + 1:]], ds)

                if tau_ray <= self.tau_ray_max:
                    if tau_ray < 0: raise Exception("Bad tau_ray")
                    exptau = np.exp(-tau_ray)
                    # The temperature gradient (if it has been calculated) is
                    # incorrect. Send this particle off to correct its flux.
                    xyzpos[2] = fronts[index]
                    particle_flux = self.get_particle_flux(
                        IDs[idx_sorted[i]],
                        IDs,
                        xyzpos, # rsun
                        interacting_uses_emerg[index],
                        interacting_uses_maxdiff[index],
                    ) # code units
                    particle_flux *= self._flux_unit # cgs
                    f = particle_flux * exptau # cgs
                    
                    total_flux += f # cgs
                    total_attenuation += particle_flux - f # cgs
                    total_unattenuated += particle_flux # cgs

                    for j, val in enumerate(flux_weighted_averages):
                        weighted_averages[j] += val[index] * f # cgs

                    contributors += [IDs[index]]
        
        # Add the surface particle's flux. The temperature gradient is correct
        # here.
        index = idx_sorted[-1]
        xyzpos[2] = fronts[index]
        particle_flux = self.get_particle_flux(
            IDs[index],
            IDs,
            xyzpos,
            interacting_uses_emerg[index],
            interacting_uses_maxdiff[index],
        ) # code units
        particle_flux *= self._flux_unit # cgs
        f = particle_flux # cgs
        
        for j, val in enumerate(flux_weighted_averages):
            weighted_averages[j] += val[index] * f # cgs
        total_flux += f # cgs
        contributors += [IDs[index]]
        
        return total_flux, weighted_averages, total_attenuation, total_unattenuated, IDs[index], np.asarray(contributors, dtype=int)
    
        
        

