import starsmashertools.helpers.argumentenforcer
import starsmashertools.flux.fluxfinder
from starsmashertools.helpers.apidecorator import api
import numpy as np

class IvanovaFluxFinder(starsmashertools.flux.fluxfinder.FluxFinder, object):
    """
    This :class:`~.fluxfinder.FluxFinder` is meant to exactly mimic
    Natasha Ivanova's implementation (separate code) while still
    providing the helpful utilities of a 
    :class:`~.fluxfinder.FluxFinder` such as the 
    :class:`~.fluxfinder.FluxFinder.Result` object.
    """

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(
            self,
            output : starsmashertools.lib.output.Output,
            min_value : float | int = 2,
            max_value : float | int = 11,
            xmin : float | int = -100,
            xmax : float | int = 100,
            ymin : float | int = -100,
            ymax : float | int = 100,
            xyrad : float | int | type(None) = None,
            do_domain : int = 1,
            do_dust : int = 0,
            do_fluffy : int = 0,
            do_rays : int = 1,
            T_dust : float | int | type(None) = None,
            T_dust_min : float | int | type(None) = None,
            kappa_dust : float | int | type(None) = None,
            tau_s : float | int | type(None) = 20,
            tau_skip : float | int | type(None) = None,
            viewing_angle : list | tuple = (0., 0.),
            teff_cut : float | int = 3500.,
            **kwargs
    ):
        import starsmashertools.preferences
        
        self._min_value = min_value
        self._max_value = max_value
        
        if xyrad is not None:
            xmin, xmax = -xyrad, xyrad
            ymin, ymax = -xyrad, xyrad

        kwargs['extent'] = [xmin, xmax, ymin, ymax]
        
        self._do_domain = do_domain
        self._do_dust = do_dust
        self._do_fluffy = do_fluffy
        self._do_rays = do_rays

        if None in [T_dust, T_dust_min]:
            pref_T_dust_min, pref_T_dust = starsmashertools.preferences.get_default(
                'FluxFinder', 'dust temperature range',
                throw_error = True,
            )
            if T_dust is None: T_dust = pref_T_dust
            if T_dust_min is None: T_dust_min = pref_T_dust_min

        self._T_dust = T_dust
        self._T_dust_min = T_dust_min
            
        if kappa_dust is None:
            kappa_dust = starsmashertools.preferences.get_default(
                'FluxFinder', 'dust opacity',
                throw_error = True,
            )
        self._kappa_dust = kappa_dust

        kwargs['tau_ray_max'] = tau_s

        if tau_skip is not None:
            raise ValueError("Keyword argument 'tau_skip' must be 'None'. If you wish to set this parameter, edit the value of key 'tau particle cutoff' in 'FluxFinder' in your preferences, located in '%s'" % starsmashertools.preferences.__file__)

        self._tau_skip = starsmashertools.preferences.get_default(
            'FluxFinder', 'tau particle cutoff',
            throw_error = True,
        )

        self._theta, self._phi = viewing_angle
        self._teff_cut = teff_cut

        #if 'resolution' in kwargs.keys():
        #    kwargs['resolution'] = (np.asarray(kwargs['resolution']) + 1).tolist()
        
        super(IvanovaFluxFinder, self).__init__(output, **kwargs)


    def _initialize(self):
        import math
        import starsmashertools.math
        
        self.contributing_particle_IDs = []
        
        units = self.output.simulation.units
        current = self.output
        xmin, xmax, ymin, ymax = self.extent
        do_domain = self._do_domain
        do_fluffy = self._do_fluffy
        do_rays = self._do_rays
        min_value = self._min_value
        max_value = self._max_value
        T_dust = self._T_dust
        T_dust_min = self._T_dust_min
        kappa_dust = self._kappa_dust

        pi_const = math.pi
        c_speed=2.99792458e10 #2.99e10
        a_const=7.565767e-15 # cgs #7.5646e-15
        lsun=3.89e33
        sigma_const=5.6704e-5
        
        
        # in case, to  ake sure we use the same units
        msun=float(units['am'])
        rsun=float(units['x'])

        ntot = current['ntot']
        rho = current['rho'] * float(units['rho'])
        m = current['am']
        h = current['hp']
        u = current['u'] * float(units['u'])
        uraddot = current['uraddot'] * float(units['uraddot'])
        uraddotcool = current['uraddotcool'] * float(units['uraddotcool'])
        uraddot_emerg = current['dEemergdt'] * float(units['dEemergdt'])
        uraddot_diff = current['dEdiffdt'] * float(units['dEdiffdt'])
        kappa = current['popacity'] * float(units['popacity'])
        tau = current['tau']
        temp = current['temperatures']    
        rloc=current['x']
        x=current['x']
        y=current['y']
        z=current['z']
        id=current['ID']

        #======

        #x,y,z = flux_tools.rotate(x,y,z,theta,0,phi)
        x,y,z = starsmashertools.math.rotate(x,y,z,self._theta,0,self._phi)
        self.viewing_angle = (self._theta, 0., self._phi)

        #print("fname: %20s" % sph_name)
        if self.verbose:
            print("min_value: %12.5f" % min_value)
            print("max_value: %12.5f" % max_value)

        if do_domain != 1 and self.verbose:
            print("xmin: %12.5f" % xmin)
            print("xmax: %12.5f" % xmax)
            print("ymin: %12.5f" % ymin)
            print("ymax: %12.5f" % ymax)

        if do_domain == 1:
            rloc_arr = (0.75*m*msun/(pi_const*rho))**0.33333/rsun
            if do_fluffy == 1:
                rloc_arr = 2.*h
            xmin = np.amin(x - rloc_arr)
            xmax = np.amax(x + rloc_arr)
            ymin = np.amin(y - rloc_arr)
            ymax = np.amax(y + rloc_arr)

            max_coord=max(abs(xmin),abs(xmax), abs(ymin), abs(ymax))
            xmax=max_coord
            ymax=max_coord
            ymin=-max_coord
            xmin=-max_coord
            if self.verbose: print("domain  max: %12.5f" %  max_coord)
            self.extent = [xmin, xmax, ymin, ymax]
        
        dx = (xmax-xmin)/self.resolution[0]
        dy = (ymax-ymin)/self.resolution[1]
        self.dx = dx
        self.dy = dy
        
        if self.verbose:
            print("dx=%f dy=%f"%(dx,dy))

        flux = np.zeros(ntot)
        teff = np.zeros(ntot)

        for i in range(ntot):
            if u[i] == 0: continue

            iloc = int((x[i]-xmin)/dx)
            jloc = int((y[i]-ymin)/dy)

            if temp[i] < T_dust and temp[i] > T_dust_min and kappa[i] < kappa_dust and do_dust == 1:
                kappa[i]=kappa_dust

            if uraddot_emerg[i] < uraddot_diff[i] and do_fluffy == 1:
                # this particle is considered in "fluffy" approximation
                rloc[i]=2.*h[i]
                rho[i]=m[i]*msun/pow(rloc[i]*rsun,3.)
                tau[i]=rho[i]*kappa[i]*rloc[i]*rsun*4./3.
                flux_em = uraddot_emerg[i]/4./pi_const/rloc[i]/rloc[i]/rsun/rsun
                flux[i]=flux_em

            else :
                # this particle is considered in "dense" approximation 
                rloc[i]=pow(m[i]*msun/(4/3.*pi_const)/rho[i],0.33333)/rsun
                tau[i]=rho[i]*kappa[i]*rloc[i]*rsun*4./3.
                ad      = 4.*pi_const*rloc[i]*rloc[i]*rsun*rsun
                ad_cool = 2.*pi_const*rloc[i]*rloc[i]*rsun*rsun
                flux_em = uraddot_emerg[i]/ad
                dd=c_speed/kappa[i]/rho[i]
                tdiffloc=kappa[i]*rho[i]*rloc[i]*rloc[i]*rsun*rsun/c_speed
                gradt=a_const*pow(temp[i],4)/rloc[i]/rsun
                dedt=ad/3.*dd*gradt/m[i]/msun
                flux[i]=dedt*m[i]*msun/ad
                flux_cool=-uraddotcool[i]*m[i]*msun/ad
                ratio = -uraddotcool[i]/dedt*12. # to figure out how many effective cooling rays
                if do_rays == 1:
                    if ratio <= 6: # all what is cooling it, goes through the outer hemisphere
                        flux[i]=-uraddotcool[i]*m[i]*msun/ad_cool # all what is cooled is to goes through the outer hemisphere
                    if ratio > 6: # the particle is getting cooled also to the back, but cools at max through the outer hemisphere
                        flux[i]=dedt*m[i]*msun/ad
                flux[i]=min(flux[i],flux_em)

            teff[i]=pow(flux[i]/sigma_const,0.25)

        self.output['x'] = x
        self.output['y'] = y
        self.output['z'] = z
        self.output['tau'] = tau
        self.output['radius'] = rloc
        self.output['flux'] = flux
        self.output['teff'] = teff


    def _get_surf_d_and_surf_id(self):
        rows = self.resolution[0] + 1
        cols = self.resolution[1] + 1
        surf_d = np.full((rows, cols), -1.e30)
        surf_id = np.full((rows, cols), -1, dtype=int)
        
        x = self.output['x']
        y = self.output['y']
        z = self.output['z']
        u = self.output['u']
        tau = self.output['tau']
        rloc = self.output['radius']
        ntot = self.output['ntot']
        rloc2 = rloc * rloc

        xmin, xmax, ymin, ymax = self.extent
        dx, dy = self.dx, self.dy

        
        ijloc = np.column_stack((
            ((x - xmin) / dx).astype(int),
            ((y - ymin) / dy).astype(int),
        ))
        idx = np.logical_and(
            np.logical_and(
                ijloc[:,0] < 0,
                ijloc[:,0] > self.resolution[0],
            ),
            np.logical_and(
                ijloc[:,1] < 0,
                ijloc[:,1] > self.resolution[1],
            ),
        )
        # Filter core particles
        idx = np.logical_or(u == 0, idx)
        # Filter small tau particles
        idx = np.logical_or(tau < self.tau_ray_max, idx)
        
        for i, (iloc, jloc) in enumerate(ijloc):
            if idx[i]: continue
            zi = z[i]
            if zi <= surf_d[iloc, jloc]: continue
            
            xi = x[i]
            yi = y[i]
            rloci = rloc[i]
            rloc2i = rloc2[i]
            
            imin = max(int((xi - rloci - xmin) / dx)    , 0)
            imax = min(int((xi + rloci - xmin) / dx) + 1, self.resolution[0])
            jmin = max(int((yi - rloci - ymin) / dy)    , 0)
            jmax = min(int((yi + rloci - ymin) / dy) + 1, self.resolution[1])
            
            for ii in range(imin, imax):
                xloc = xmin + dx * ii
                deltax = xloc - xi
                deltax2 = deltax * deltax
                for jj in range(jmin, jmax):
                    if zi <= surf_d[ii, jj]: continue
                    yloc = ymin + dy * jj
                    deltay = yloc - yi
                    dist2 = deltax2 + deltay * deltay
                    if dist2 > rloc2i: continue
                    surf_d[ii, jj] = zi
                    surf_id[ii, jj] = i
        return surf_d, surf_id

    def _get_ray_id_and_ray_n(self, surf_d):
        rows = self.resolution[0] + 1
        cols = self.resolution[1] + 1
        
        ray_id =  np.zeros((rows, cols, 600), dtype=int)
        ray_n =  np.zeros((rows, cols), dtype=int)
        
        x = self.output['x']
        y = self.output['y']
        z = self.output['z']
        u = self.output['u']
        tau = self.output['tau']
        flux = self.output['flux']
        teff = self.output['teff']
        rloc = self.output['radius']
        ntot = self.output['ntot']
        rloc2 = rloc * rloc

        dx, dy = self.dx, self.dy
        xmin, xmax, ymin, ymax = self.extent
        
        tau_min = 1

        ijloc = np.column_stack((
            ((x - xmin) / dx).astype(int),
            ((y - ymin) / dy).astype(int),
        ))

        idx = np.logical_and(
            np.logical_and(
                ijloc[:,0] < 0,
                ijloc[:,0] > self.resolution[0],
            ),
            np.logical_and(
                ijloc[:,1] < 0,
                ijloc[:,1] > self.resolution[1],
            ),
        )
        
        idx = np.logical_or(tau < self._tau_skip, idx)

        for i, (iloc, jloc) in enumerate(ijloc[~idx]):
            xi = x[i]
            yi = y[i]
            zi = z[i]
            rloci = rloc[i]
            rloc2i = rloc2[i]
            taui = tau[i]
            
            imin = max(int((xi - rloci - xmin) / dx)    , 0)
            imax = min(int((xi + rloci - xmin) / dx) + 1, self.resolution[0])
            jmin = max(int((yi - rloci - ymin) / dy)    , 0)
            jmax = min(int((yi + rloci - ymin) / dy) + 1, self.resolution[1])
            for ii in range(imin, imax):
                xloc = xmin + dx * ii
                deltax2 = (xloc - xi)**2
                for jj in range(jmin, jmax):
                    if zi <= surf_d[ii, jj]: continue
                    yloc = ymin + dy * jj
                    dist2 = deltax2 + (yloc - yi)**2
                    if dist2 > rloc2i: continue
                    ray_id[ii, jj][ray_n[ii, jj]] = i
                    ray_n[ii, jj] += 1
                    tau_min = min(tau_min, taui)

        i_maxray, j_maxray = np.unravel_index(np.argmax(ray_n), ray_n.shape)
        max_ray = ray_n[i_maxray, j_maxray]

        if self.verbose:
            print("maximum number of particles above the cut off optically thick surface is %d %d %d"%(max_ray,i_maxray,j_maxray))            
            print("minimum tau account for is  is %f"%(tau_min))

        # this sorting is as simple as hell 
        for ii in range(self.resolution[0]):
            for jj in range(self.resolution[1]):
                if ray_n[ii, jj] <= 1: continue
                for kk in range(ray_n[ii, jj]):
                    swap = 0
                    for ri in range(ray_n[ii, jj] - 1):
                        ir1 = ray_id[ii, jj, ri]
                        ir2 = ray_id[ii, jj, ri + 1]
                        if ir1 == 0 or ir2 == 0:
                            raise Exception("ray to the core? %d %d", ii, jj)
                        if z[ir1] <= z[ir2]: continue
                        ray_id[ii, jj, ri]=ir2
                        ray_id[ii, jj, ri + 1]=ir1
                        swap += 1
                    if swap == 0:
                        break
                if swap > 0:
                    raise Exception("Did not complete sorting")
        return ray_id, ray_n

    def get_flux(
            self,
            tau,
            flux,
            teff,
            teff_cut,
            IDs,
            surf_ID,
            flux_weighted_averages = [],
    ):
        attenuation = 0.
        unattenuated = 0.
        total_flux = 0.
        surf_br_v = 0.
        contributors = []
        weighted_averages = np.zeros(len(flux_weighted_averages))
        
        tau_ray = 0.
        if len(IDs) > 0:
            for ri in range(len(IDs) - 1, -1, -1): #ray_n[ii, jj] - 1, -1, -1):
                ir = IDs[ri] #ray_id[ii, jj, ri]
                particle_flux = flux[ir]
                f = particle_flux * np.exp(-tau_ray)

                total_flux += f
                attenuation += particle_flux - f
                unattenuated += particle_flux
                contributors += [ir]

                for _i, val in enumerate(flux_weighted_averages):
                    weighted_averages[_i] += val[ir] * f
                
                if teff[ir] > teff_cut:
                    surf_br_v += f
                
                tau_ray += tau[ir]

                if surf_br_v > total_flux:
                    raise Exception("part of the flux is larger then whole flux?   %d   tau=%8.4e  %8.4e  %8.4e  %8.4e"%(ir,tau[ir],total_flux,surf_br_v,flux[ir]))

        i = surf_ID
        if i > 0:
            particle_flux = flux[-1]
            f = particle_flux * np.exp(-tau_ray)
            total_flux += f

            attenuation += particle_flux - f
            unattenuated += particle_flux
            contributors += [i]

            for _i, val in enumerate(self.flux_weighted_averages):
                weighted_averages[_i] += val[i] * f

            if teff[i] > teff_cut: surf_br_v += f

        return total_flux, surf_br_v, attenuation, unattenuated, list(set(contributors)), weighted_averages

    def get(self, parallel : bool = False):
        import starsmashertools.flux.fluxfinder
        
        teff_cut = self._teff_cut
        
        tau = self.output['tau']
        flux = self.output['flux']
        teff = self.output['teff']

        surf_d, surf_id = self._get_surf_d_and_surf_id()
        ray_id, ray_n = self._get_ray_id_and_ray_n(surf_d)

        rows = self.resolution[0] + 1
        cols = self.resolution[1] + 1
        surf_br = np.zeros((rows,cols))
        surf_br_v = np.zeros((rows,cols))
        self._radiation_map = np.full(surf_br.shape, np.nan)

        weighted_averages = []
        for _ in range(len(self.flux_weighted_averages)):
            weighted_averages += [np.zeros(surf_br.shape)]
        
        attenuation = np.zeros(surf_br.shape)
        unattenuated = np.zeros(surf_br.shape)
        contributors = np.full(self.output['ntot'], False)

        all_args = []
        all_kwargs = []
        indices = []
        
        for ii in range(self.resolution[0]):
            for jj in range(self.resolution[1]):
                IDs = []
                if ray_n[ii, jj] > 0:
                    IDs = ray_id[ii, jj]
                #IDs += [surf_id[ii, jj]]
                all_args += [(tau,flux,teff,teff_cut,IDs,surf_id[ii, jj],)]
                all_kwargs += [{
                    'flux_weighted_averages' : self.flux_weighted_averages,
                }]
                indices += [[ii, jj]]

        def handle_output(ii, jj, output):
            result = self.get_flux(*arg, **kw)
            surf_br[ii, jj] = result[0]
            surf_br_v[ii, jj] = result[1]
            attenuation[ii, jj] = result[2]
            unattenuated[ii, jj] = result[3]
            _contributors = result[4]
            _weighted_averages = result[5]
            
            contributors[_contributors] = True
            for i, val in enumerate(_weighted_averages):
                weighted_averages[i][ii, jj] = val

        increment = 1. / float(len(all_args))
        progress = 0.
        self.print_progress(progress)
        if not parallel:
            for (ii, jj), arg, kw in zip(indices, all_args, all_kwargs):
                handle_output(ii, jj, self.get_flux(*arg, **kw))
                progress += increment
                if self.verbose: self.print_progress(progress)
        else:
            import starsmashertools.helpers.asynchronous
            parallel = starsmashertools.helpers.asynchronous.ParallelFunction(
                self.get_flux,
                args = all_args,
                kwargs = all_kwargs,
                daemon = True,
            )
            
            if self.verbose:
                parallel.do(
                    0.25,
                    lambda parallel=parallel: self.print_progress(parallel.get_progress()),
                )

            outputs = parallel.get_output()
            for index, result in zip(indices, outputs):
                ii, jj = index
                handle_result(ii, jj, result)

        self.contributing_particle_IDs = np.where(contributors)[0]

        # Complete the weighted averages
        idx = np.logical_and(surf_br > 0, np.isfinite(surf_br))
        if np.any(idx):
            for i in range(len(weighted_averages)):
                weighted_averages[i][idx] /= surf_br[idx]
                weighted_averages[i][~idx] = 0
        
        return starsmashertools.flux.fluxfinder.FluxFinder.Result(
            surf_br,
            weighted_averages,
            attenuation,
            unattenuated,
            surf_id,
            fluxfinder = self,
        )
