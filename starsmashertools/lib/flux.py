# I hate how this is written, but at least it works specifically for us...
import starsmashertools.preferences
from starsmashertools.preferences import Pref
import math
import numpy as np
import starsmashertools.lib.output
from starsmashertools.helpers.apidecorator import api
import starsmashertools.helpers.argumentenforcer
from starsmashertools.lib.units import constants
import starsmashertools.helpers.nesteddict
import copy
import warnings
import typing
import collections
import gc

try:
    import matplotlib.axes
    has_matplotlib = True
except ImportError:
    has_matplotlib = False


def get(*args, **kwargs):
    weighted_averages = kwargs.pop('weighted_averages', [])
    finder = FluxFinder(*args, **kwargs)
    finder.get()
    return finder.result

@starsmashertools.preferences.use
class FluxFinder(object):
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(
            self,
            output : starsmashertools.lib.output.Output,
            weighted_averages : list | tuple = [],

            units : dict | type(None) = None,

            # Image
            resolution : list | tuple = Pref('resolution', (400, 400)),
            extent : list | tuple | type(None) = None,
            extent_limits : list | tuple | type(None) = None,
            theta : int | float = Pref('theta', 0.),
            phi : int | float = Pref('phi', 0.),
            fluffy : bool = Pref('fluffy', False),
            rays : bool = Pref('rays', True),
            tau_s : float | int = Pref('tau_s', 20.),
            tau_skip : float | int = Pref('tau_skip', 1.e-5),
            teff_cut : float | int = Pref('teff_cut', 3500.),
            dust_opacity : float | int | type(None) = Pref('dust_opacity', 1),
            dust_Trange : list | tuple | type(None) = Pref('dust_Trange', (100, 1000)),
            flux_limit_min : int | float | type(None) = None,

            # Spectrum
            spectrum_size : int | float = Pref('spectrum_size', 1000),
            lmax : int | float = Pref('lmax', 3000),
            dl_il : int | float = Pref('dl_il', 3.),
            dl : int | float = Pref('dl', 10),
            nbox : int = Pref('nbox', 10),
            nboxh : int = Pref('nboxh', 5),
            l_range : list | tuple = Pref('l_range', (10, 3000)),
            factor1 : int | float = Pref('factor1', 1.191045e-05),
            factor2 : int | float = Pref('factor2', 1.438728e+00),
            filters : dict = Pref('filters', {}),

            # Debugging
            verbose : bool = False,
            test_ray : bool = False,
            itr : int = -1,
            jtr : int = -1,
            test_particle : bool = False,
            ic : int = -1,
    ):
        _kwargs = locals()
        _kwargs.pop('self')
        _kwargs.pop('output')
        
        self.output = copy.deepcopy(output)
        del output

        if self.output.simulation['ncooling'] not in [2, 3]:
            raise NotImplementedError("FluxFinder can only be used on simulations which have input parameter 'ncooling' 2 or 3, not %d" % self.output.simulation['ncooling'])

        if units is None:
            units = {
                'length' : self.output.simulation.units.length,
            }
        self.units = units
        
        self.resolution = resolution
        self.extent = extent
        self.extent_limits = extent_limits
        self.theta = theta
        self.phi = phi
        self.fluffy = fluffy
        self.rays = rays
        self.tau_s = tau_s
        self.tau_skip = tau_skip
        self.teff_cut = teff_cut
        self.dust_opacity = dust_opacity
        self.dust_Trange = dust_Trange
        self.flux_limit_min = flux_limit_min
        
        self.nbox = nbox
        self.nboxh = nboxh
        
        self.verbose = verbose
        self.test_ray = test_ray
        self.itr = itr
        self.jtr = jtr
        self.test_particle = test_particle
        self.ic = ic

        # Set the viewing angle
        self.output.rotate(xangle = theta, zangle = phi)
        
        self._tau = self.output['tau']
        self._rloc = np.full(self.output['ntot'], np.nan)
        
        self.spectrum = FluxFinder.Spectrum(
            spectrum_size,
            lmax,
            dl_il,
            dl,
            nbox,
            nboxh,
            l_range,
            factor1,
            factor2,
            filters,
            self.output['ntot'],
        )

        self.dx = None
        self.dy = None

        self.images = collections.OrderedDict()
        self.images['flux'] = FluxFinder.Image(self.resolution)
        self.images['flux v'] = FluxFinder.Image(self.resolution)

        # We multiply by the flux later in _init_particles
        for i, array in enumerate(weighted_averages):
            self.images['weightedaverage%d' % i] = FluxFinder.Image(
                self.resolution, particle_array = copy.deepcopy(array),
            )

        self.result = FluxResult({
            'version' : starsmashertools.__version__,
            'output' : self.output.path,
            'kwargs' : _kwargs,
            'simulation' : self.output.simulation.directory,
            'units' : self.units,
            'time' : self.output['t'] * self.output.simulation.units['t'],
            'particles' : {},
            'image' : {},
            'spectrum' : {},
        })

    def initialize(self):
        self._init_grid()
        self._init_particles()

    def get(self):
        import starsmashertools.lib.units
        
        self.initialize()
        self._process_images()

        # Vectorized:
        area_br = 0.25 * (\
            self.images['flux'].array[ :self.resolution[0]-1, :self.resolution[1]-1] + \
            self.images['flux'].array[ :self.resolution[0]-1,1:self.resolution[1]] + \
            self.images['flux'].array[1:self.resolution[0]  , :self.resolution[1]-1] + \
            self.images['flux'].array[1:self.resolution[0]  ,1:self.resolution[1]])

        surf_t = np.zeros(self.images['flux'].array.shape, dtype = float)
        surf_t[:self.resolution[0]-1,:self.resolution[1]-1] = (area_br/float(constants['sigmaSB']))**0.25
        flux_tot = self.images['flux'].get_total()
        flux_tot_v = np.sum(self.images['flux v'].array[:self.resolution[0]-1,:self.resolution[1]-1])

        # Vectorized:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            teff_aver = np.sum(surf_t[:self.resolution[0]-1, :self.resolution[1]-1] * area_br / flux_tot)

        # Units of energy/time/area
        flux_tot = starsmashertools.lib.units.Unit(flux_tot, 'g/s*s*s')
        flux_tot_v = starsmashertools.lib.units.Unit(flux_tot_v, 'g/s*s*s')
        
        # Finish the ltot and l_v calculations
        length = self.units.get('length', 1.)
        cell_area = self.dx * self.dy * length**2
        ltot = 4 * cell_area * flux_tot
        l_v  = 4 * cell_area * flux_tot_v

        self.spectrum.finalize()
        
        ltot_spectrum = 4 * cell_area * self.spectrum.flux['total']
        l_spectrum = {key:4 * cell_area * val for key, val in self.spectrum.flux.items() if key != 'total'}

        self.result['image','Teff_cell'] = surf_t
        self.result['image','teff_aver'] = teff_aver
        self.result['image','ltot'] = ltot
        self.result['image','flux_tot'] = flux_tot
        self.result['image','l_v'] = l_v
        self.result['spectrum','ltot_spectrum'] = ltot_spectrum
        self.result['spectrum','l_spectrum'] = l_spectrum
        self.result['spectrum','output'] = self.spectrum.output
        self.result['spectrum','teff'] = self.spectrum.teff

        del area_br, surf_t, flux_tot, flux_tot_v, teff_aver, length, cell_area
        del ltot, l_v, ltot_spectrum, l_spectrum
        gc.collect()
        

    def _init_particles(self):
        # Below is a much faster alternative to looping through particles to set
        # values.
        c_speed = float(constants['c'])
        sigma_const = float(constants['sigmaSB'])
        a_const = float(constants['a'])

        units = self.output.simulation.units

        kappa = self.output['popacity'] * float(units['popacity'])
        if self.dust_opacity is not None:
            kappa = self._apply_dust(kappa)

        # No heating
        uraddotcool = self.output['uraddot'] * float(units['uraddot'])

        # Heating
        if self.output.simulation['ncooling'] == 3:
            uraddotcool = self.output['uraddotcool'] * float(units['uraddotcool'])
        
        uraddot_emerg = self.output['dEemergdt'] * float(units['dEemergdt'])
        uraddot_diff = self.output['dEdiffdt'] * float(units['dEdiffdt'])
        
        self._flux = np.zeros(self.output['ntot'], dtype = float)
        self._teff = np.zeros(self.output['ntot'], dtype = float)
        
        flux_rad = np.zeros(self._flux.shape)
        flux_em = np.zeros(self._flux.shape)

        idx = np.full(self.output['ntot'], False)
        if self.fluffy:
            idx = uraddot_emerg < uraddot_diff
            
        cores = self.output['u'] == 0

        self._rloc[cores] = 0

        m = self.output['am'] * float(units['am'])
        rho = copy.deepcopy(self.output['rho']) * float(units['rho'])
        
        i = np.logical_and(idx, ~cores)
        if i.any():
            self._rloc[i] = 2.*self.output['hp'][i] * float(units['hp'])
            rho[i]=m[i]/pow(self._rloc[i],3.)
            self._tau[i]=rho[i]*kappa[i]*self._rloc[i]*4./3.
            flux_em[i] = uraddot_emerg[i]*0.25/(np.pi*self._rloc[i]**2)
            self._flux[i]=flux_em[i]
            flux_rad[i] = flux_em[i]

        temp = self.output['temperatures'] * float(self.output.simulation.units['temperatures'])
        i = np.logical_and(~idx, ~cores)
        if i.any():
            # this particle is considered in "dense" approximation
            self._rloc[i]=pow(m[i]/(4/3.*np.pi)/rho[i],0.33333)
            self._tau[i]=rho[i]*kappa[i]*self._rloc[i]*4./3.
            ad      = 4.*np.pi*self._rloc[i]**2
            ad_cool = 2.*np.pi*self._rloc[i]**2
            flux_em[i] = uraddot_emerg[i]/ad
            dd=c_speed/kappa[i]/rho[i]
            gradt=a_const*pow(temp[i],4)/self._rloc[i]
            dedt=ad/3.*dd*gradt/m[i]
            self._flux[i]=dedt*m[i]/ad
            flux_rad[i] = dedt * m[i] / ad

            if self.rays:
                ratio = np.full(self._flux.shape, np.nan)
                ratio[i] = -uraddotcool[i]/dedt*12. # to figure out how many effective cooling rays
                _idx1 = np.logical_and(
                    ~cores,
                    np.logical_and(i, ratio <= 6),
                )
                if _idx1.any():
                    ad_cool = 2.*np.pi*self._rloc[_idx1]*self._rloc[_idx1]
                    self._flux[_idx1]=-uraddotcool[_idx1]*m[_idx1]/ad_cool


                _idx2 = np.logical_and(
                    ~cores,
                    np.logical_and(i, ratio > 6),
                )
                if _idx2.any():
                    ad      = 4.*np.pi*self._rloc[_idx2]*self._rloc[_idx2]
                    gradt = a_const*pow(temp[_idx2],4)/self._rloc[_idx2]
                    dd = c_speed/kappa[_idx2]/rho[_idx2]
                    dedt = ad/3.*dd*gradt/m[_idx2]
                    self._flux[_idx2] = dedt*m[_idx2]/ad

                del ratio, _idx1, _idx2

            _idx = np.logical_and(
                ~cores,
                np.logical_and(i, self._tau < 100),
            )
            if _idx.any():
                self._flux[_idx] = np.minimum(self._flux[_idx], flux_em[_idx])

            del ad, ad_cool, dd, gradt, dedt, _idx


        self._teff[~cores] = (flux_rad[~cores] / sigma_const)**0.25

        #self.spectrum.spect_type[~cores] = 1 # black body
        i = np.logical_and(
            ~cores,
            np.logical_and(flux_rad > flux_em, self._tau < 100),
        )
        if i.any():
            self._teff[i] = temp[i]
        #    self.spectrum.spect_type[i] = 2 # emerging

        self._rloc /= float(self.units.get('length', 1.))

        
        flux_v = copy.deepcopy(self._flux)
        flux_v[self._teff <= self.teff_cut] = 0

        self.images['flux'].particle_array = self._flux
        self.images['flux v'].particle_array = flux_v

        # Finish adding the weights for the averages
        for key in self.images.keys():
            if 'weightedaverage' not in key: continue
            self.images[key].particle_array *= self._flux

        self.result['particles','contributing_IDs'] = np.arange(self.output['ntot'], dtype=int)
        self.result['particles','rloc'] = self._rloc
        self.result['particles','tau'] = self._tau
        self.result['particles','kappa'] = kappa
        self.result['particles','flux'] = self._flux

        del c_speed, sigma_const, a_const, units, kappa, uraddotcool
        del uraddot_emerg, uraddot_diff, flux_rad, flux_em, idx
        del cores, m, rho, i, temp, flux_v
        gc.collect()

    def _init_grid(self):
        if self.extent is None:
            units = self.output.simulation.units
            
            # Get particle quantities in CGS
            x = self.output['x'] * float(units['x'])
            y = self.output['y'] * float(units['y'])
            m = self.output['am'] * float(units['am'])
            rho = self.output['rho'] * float(units['rho'])
            h = self.output['hp'] * float(units['hp'])

            delta = 2 * h
            if not self.fluffy:
                cores = self.output['u'] == 0
                if (~cores).any():
                    delta[~cores] = (0.75*m[~cores]/(np.pi*rho[~cores]))**0.33333
                del cores

            length = float(self.units.get('length', 1.))
            xmin = np.amin(x - delta) / length
            xmax = np.amax(x + delta) / length
            ymin = np.amin(y - delta) / length
            ymax = np.amax(y + delta) / length

            max_coord=max(abs(xmin),abs(xmax), abs(ymin), abs(ymax))
            xmax=max_coord
            ymax=max_coord
            ymin=-max_coord
            xmin=-max_coord
            self.extent = [xmin, xmax, ymin, ymax]
            if self.verbose: print("domain  max: %12.5f" %  max_coord)
            del units, x, y, m, rho, h, max_coord, length, delta
        else:
            xmin, xmax, ymin, ymax = self.extent

        if self.extent_limits is not None:
            xminlim, xmaxlim, yminlim, ymaxlim = self.extent_limits
            if xminlim is not None: xmin = max(xmin, xminlim)
            if xmaxlim is not None: xmax = min(xmax, xmaxlim)
            if yminlim is not None: ymin = max(ymin, yminlim)
            if ymaxlim is not None: ymax = min(ymax, ymaxlim)
            self.extent = [xmin, xmax, ymin, ymax]
            
            del xminlim, xmaxlim, yminlim, ymaxlim

        self.dx = (xmax - xmin)/self.resolution[0]
        self.dy = (ymax - ymin)/self.resolution[1]

        if self.verbose:
            print("_init_grid:")
            print("   xmin, xmax, dx = %12.5f %12.5f %12.5f" % (xmin, xmax, self.dx))
            print("   ymin, ymax, dy = %12.5f %12.5f %12.5f" % (ymin, ymax, self.dy))

        self.result['image','extent'] = self.extent
        self.result['image','dx'] = self.dx
        self.result['image','dy'] = self.dy
        del xmin, xmax, ymin, ymax
        gc.collect()

    def _apply_dust(self, kappa):
        T = self.output['temperatures'] * float(self.output.simulation.units['temperatures'])
        # Vectorized dust handling
        def find_dust():
            T_dust_min, T_dust_max = self.dust_Trange
            if self.dust_opacity is not None:
                if T_dust_max is None and T_dust_min is not None:
                    return np.logical_and(
                        T > T_dust_min,
                        kappa < self.dust_opacity,
                    )
                elif T_dust_max is not None and T_dust_min is None:
                    return np.logical_and(
                        T < T_dust_max,
                        kappa < self.dust_opacity,
                    )

                if T_dust_max is None and T_dust_min is None:
                    return kappa < self.dust_opacity

                return np.logical_and(
                    np.logical_and(
                        T < T_dust_max,
                        T > T_dust_min,
                    ),
                    kappa < self.dust_opacity,
                )

            if T_dust_max is None and T_dust_min is not None:
                return T > T_dust_min
            elif T_dust_max is not None and T_dust_min is None:
                return T < T_dust_max
            
            return np.full(False, T.shape, dtype = bool)

        # Find the dust
        idx = find_dust()
        # Apply the dust, if there is any
        if idx.any(): kappa[idx] = self.dust_opacity

        del idx, T, find_dust
        gc.collect()
        
        return kappa

    def _process_images(self):
        xmin, xmax, ymin, ymax = self.extent
        
        units = self.output.simulation.units
        length = float(self.units.get('length', 1.))
        
        x = self.output['x'] * float(units['x']) / length
        y = self.output['y'] * float(units['y']) / length
        z = self.output['z'] * float(units['z']) / length

        flux_from_contributors = np.zeros(self.output['ntot'], dtype = float)

        # This method is equivalent to creating an empty numpy array of type
        # list with shape (rows, cols) and then filling the array with list()
        # instances,
        # e.g.:
        #    ray_id = np.empty((rows,cols), dtype=list)
        #    for i in range(rows):
        #        for j in range(cols):
        #           ray_id[i][j] = list()
        # However, this method is much, much faster.
        # https://stackoverflow.com/a/33987165
        shape = np.asarray(self.resolution, dtype = int) + 1
        ray_id=np.frompyfunc(list, 0, 1)(np.empty(shape, dtype = object))
        ray_n = np.zeros(shape, dtype = int)
        surf_d = np.full(shape, -1.e30, dtype = float)
        surf_id = np.full(shape, -1, dtype=int)
        
        grid_indices = np.indices(shape).transpose(1,2,0)
        
        # These are used throughout for much faster processing
        ijloc_arr = np.column_stack((
            np.floor((x - xmin) / self.dx).astype(int),
            np.floor((y - ymin) / self.dy).astype(int),
        ))

        rloc = self._rloc
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ijminmax_arr = np.column_stack((
                # imin
                np.maximum(
                    np.floor((x - rloc - xmin) / self.dx).astype(int),
                    np.zeros(self.output['ntot'], dtype = int),
                ),
                # imax
                np.minimum(
                    np.ceil((x + rloc - xmin) / self.dx).astype(int),
                    np.full(self.output['ntot'], self.resolution[0], dtype = int),
                ),
                # jmin
                np.maximum(
                    np.floor((y - rloc - ymin) / self.dy).astype(int),
                    np.zeros(self.output['ntot'], dtype = int),
                ),
                # jmax
                np.minimum(
                    np.ceil((y + rloc - ymin) / self.dy).astype(int),
                    np.full(self.output['ntot'], self.resolution[1], dtype = int),
                ),
            ))

        # Use only imin,imax,jmin,jmax arguments which result in a non-empty slice
        mask = (ijminmax_arr[:,0] != ijminmax_arr[:,1]) & (ijminmax_arr[:,2] != ijminmax_arr[:,3])
        # Use only particles inside the image
        mask &= (ijloc_arr[:,0] >= 0) & (ijloc_arr[:,0] <= self.resolution[0])
        mask &= (ijloc_arr[:,1] >= 0) & (ijloc_arr[:,1] <= self.resolution[1])
        
        if not mask.any():
            raise Exception("There are no particles located within the given extents")

        # Use only particles with large enough optical depths
        mask1 = mask & (self._tau >= self.tau_s)

        # To vectorize the operation, we build a single big array of particle
        # values. We know that this will fit in memory because otherwise the
        # Output objects wouldn't fit in memory.

        dtype = np.dtype({
            'names' : ['i','x','y','z','rloc','tau','iloc','jloc','slice'],
            'formats' : [int,float,float,float,float,float,int,int,object],
        })
        array = np.empty(len(x), dtype = dtype)

        array['i'] = np.arange(len(array), dtype = int)
        array['x'] = x
        array['y'] = y
        array['z'] = z
        array['rloc'] = rloc
        array['tau'] = self._tau
        array['iloc'] = ijloc_arr[:,0]
        array['jloc'] = ijloc_arr[:,1]
        array['slice'] = [np.s_[i0:i1,j0:j1] for i0,i1,j0,j1 in ijminmax_arr.tolist()]
            
        # Find the z positions closest to the observer on each grid cell. This is
        # where the first particle is found along each line of sight.

        # I'm not sure this operation can be vectorized, since each loop
        # iteration depends on the results from the previous
        # tolist() is faster, which is a bit silly.
        for _i,(i,_x,_y,_z,_r,_tau,iloc,jloc,_slice) in enumerate(array[mask1].tolist()):
            if _z <= surf_d[iloc, jloc]: continue

            idx = _z > surf_d[_slice]
            if not idx.any(): continue

            # Minimal allocations
            dist2_arr = ((xmin + self.dx * np.arange(_slice[0].start, _slice[0].stop) - _x)**2)[:,None] + \
                (ymin + self.dy * np.arange(_slice[1].start, _slice[1].stop) - _y)**2
            
            idx &= dist2_arr <= _r**2
            if not idx.any(): continue

            surf_d[_slice][idx] = _z
            surf_id[_slice][idx] = i

        tau_min=1

        # Use only particles with large enough tau
        mask2 = mask & (self._tau >= self.tau_skip)
        if self.flux_limit_min is not None:
            # Use only particles which meet the flux threshold
            mask2 &= self._flux > self.flux_limit_min

        if mask2.any():
            mask2[mask2] &= list((_z > surf_d[_slice]).any() for _z,_slice in zip(
                array[mask2]['z'].tolist(), array[mask2]['slice'].tolist()
            ))
            
        for i,_x,_y,_z,_r,_tau,iloc,jloc,_slice in array[mask2].tolist():
            # Minimal allocations
            dist2_arr = ((xmin + self.dx * np.arange(_slice[0].start, _slice[0].stop) - _x)**2)[:,None] + \
                (ymin + self.dy * np.arange(_slice[1].start, _slice[1].stop) - _y)**2

            idx = (_z > surf_d[_slice]) & (dist2_arr <= _r**2)
            if not idx.any(): continue
            
            tau_min = min(tau_min, _tau)

            for ii, jj in grid_indices[_slice][idx]:
                ray_id[ii, jj].append(i)
                ray_n[ii, jj] += 1

        
        # Much more efficient calculations below
        if self.verbose:
            max_ray = np.amax(ray_n)
            idx_maxray = np.argmax(ray_n.flatten())
            i_maxray, j_maxray = np.unravel_index(idx_maxray, ray_n.shape)

            print("maximum number of particles above the cut off optically thick surface is %d %d %d"%(max_ray,i_maxray,j_maxray))            
            print("minimum tau account for is  is %f"%(tau_min))
            del max_ray,idx_maxray, i_maxray, j_maxray

        
        # this sorting is as simple as hell 
        # Package array in the loop for quicker iterations (moves the elements onto
        # lower level caches)
        for ii, ray_n_ii in enumerate(ray_n[:self.resolution[0]]):
            for jj, ray_n_iijj in enumerate(ray_n_ii[:self.resolution[1]]):
                if ray_n_iijj > 1:
                    # Pre-build ranges for faster iterations
                    ri_range = range(ray_n_iijj - 1)
                    for kk in range(ray_n_iijj):
                        swap=0
                        for ri in ri_range:
                            ir1=ray_id[ii][jj][ri]
                            ir2=ray_id[ii][jj][ri+1]
                            if ir1==0 or ir2 ==0:
                                raise Exception("ray to the core? %d %d", ii, jj)
                            if z[ir1] > z[ir2]:
                                ray_id[ii][jj][ri]=ir2
                                ray_id[ii][jj][ri+1]=ir1
                                swap=swap+1
                        if swap == 0:
                            break
                    if swap > 0:
                        raise Exception("Did not complete sorting")

        # Package the loop for quicker enumeration
        zipped = zip(
            surf_id[:self.resolution[0],:self.resolution[1]],
            ray_id[:self.resolution[0],:self.resolution[1]],
        )
        for ii, (surf_id_row, ray_id_ii) in enumerate(zipped):
            for jj, (i, ray_id_iijj) in enumerate(zip(surf_id_row, ray_id_ii)):
                tau_ray=0.    
                # Reversing an array is easy. This produces an iterator ('view'
                # of the list) which runs faster than ray_id[ii][jj][::-1] and
                # also faster than range(ray_n[ii][jj]-1, -1, -1).
                for ir in reversed(ray_id_iijj):
                    attenuation = math.exp(-tau_ray)
                    f = self._flux[ir] * attenuation
                    
                    for key, image in self.images.items():
                        image.add(ir, ii, jj, attenuation)
                    
                    if self.images['flux v'].array[ii,jj] > self.images['flux'].array[ii,jj]:
                        raise Exception("part of the flux is larger than the whole flux? " + str(ir) + " "+ str(ii) + " " + str(jj))
                    
                    flux_from_contributors[ir] += f
                    self.spectrum.add(f, self._teff[ir])

                    tau_ray += self._tau[ir]
                    
                if i > 0:
                    attenuation = math.exp(-tau_ray)
                    f = self._flux[i] * math.exp(-tau_ray)
                    for key, image in self.images.items():
                        image.add(i, ii, jj, attenuation)

                    if self.images['flux v'].array[ii,jj] > self.images['flux'].array[ii,jj]:
                        raise Exception("part of the flux is larger than the whole flux? " + str(ir) + " "+ str(ii) + " " + str(jj))
                    
                    flux_from_contributors[i] += f
                    
                    self.spectrum.add(f, self._teff[i])
                    
        # Finish obtaining the weighted averages
        idx = np.logical_and(
            self.images['flux'].array > 0,
            np.isfinite(self.images['flux'].array),
        )
        if np.any(idx):
            for key, val in self.images.items():
                if 'weightedaverage' not in key: continue
                self.images[key].array[idx] /= self.images['flux'].array[idx]
                
        if np.any(~idx):
            for key, val in self.images.items():
                if 'weightedaverage' not in key: continue
                self.images[key].array[~idx] = 0

        # Narrow down the output arrays to minimize storage space
        contributors = flux_from_contributors > 0
        for key in ['contributing_IDs', 'rloc', 'tau', 'kappa', 'flux']:
            self.result['particles',key] = self.result['particles',key][contributors]
        self.result['particles','flux_from_contributors'] = flux_from_contributors[contributors]
        
        self.result['image','flux'] = self.images['flux'].array
        self.result['image','flux_v'] = self.images['flux v'].array
        self.result['image','surf_d'] = surf_d
        self.result['image','surf_id'] = surf_id
        self.result['image','ray_n'] = ray_n
        self.result['image','weighted_averages'] = [image.array for key, image in self.images.items() if 'weightedaverage' in key]

        del xmin,xmax,ymin,ymax,units,length,x,y,z,flux_from_contributors
        del shape,ray_id,ray_n,surf_d,surf_id,grid_indices,ijloc_arr
        del rloc,ijminmax_arr,mask,mask1,dtype,array,tau_min,mask2
        del zipped,idx,contributors
        gc.collect()

        



    class Spectrum(object):
        def __init__(
                self,
                spectrum_size,
                lmax,
                dl_il,
                dl,
                nbox,
                nboxh,
                l_range,
                factor1,
                factor2,
                filters,
                ntot,
        ):
            self.dl = dl
            self.filters = filters
            self.factor1 = factor1
            self.factor2 = factor2
            self._sigma = float(constants['sigmaSB'])
            
            # Spectrum setup
            self.spectrum = np.zeros(spectrum_size, dtype=float)
            self.b_l = np.zeros(self.spectrum.shape, dtype=float)
            self._n_l = math.floor((l_range[1] - l_range[0]) / dl) + 1
            self.lam = (l_range[0] + dl*np.arange(self._n_l))*1e-7 # to CGS from nanometers
            self._invlam = 1. / self.lam
            self._invlam5 = self._invlam**5
            
            #self.spect_type = np.zeros(ntot, dtype=int)

            self.il_max = math.floor(lmax / dl_il)

            self.output = None
            self.flux = None
            self.teff = None

        def add(self, flux, teff):
            b_full = self._sigma / np.pi * teff**4
            expon = self.factor2 / (teff * self.lam[:self._n_l])
            idx = expon < 150
            b_l = self.factor1 / (np.exp(expon[idx]) - 1) * self._invlam5[:self._n_l][idx]
            # This is the non-allocating way of doing the calculation
            self.spectrum[:self._n_l][idx] += flux * (b_l / b_full)
            del expon, idx, b_full, b_l
        
        def finalize(self):
            import starsmashertools.lib.units
            
            # Finish spectrum calculations
            self.teff = 0
            max_flux = 0
            #ltot_spectrum = 0
            # [0.00000000e+00 7.39256413e-21 2.00355212e-07 6.82084616e-01
            #  4.39889814e+03 1.28926397e+06 6.63126098e+07 1.17342608e+09
            #  1.04031610e+10 5.80568647e+10]

            self.flux = {
                'total' : np.sum(self.spectrum[:self.il_max - 1] * self.dl * 1e-7),
            }
            for key in self.filters.keys(): self.flux[key] = 0.
            
            for il in range(self.il_max - 1):
                sp = self.spectrum[il]
                #ltot_spectrum += sp * self.dl * 1e-7 # nanom = 1e-7
                if sp <= 1: continue

                sp = np.log10(sp)
                lamb = self.lam[il] / 1e-7 # nanom = 1e-7
                teff_loc = 2.9*1.e6/lamb
                if sp > max_flux:
                    max_flux = sp
                    self.teff = teff_loc

                for key, (lamb_low, lamb_hi) in self.filters.items():
                    if lamb_low is not None and lamb < lamb_low: continue
                    if lamb_hi is not None and lamb > lamb_hi: continue

                    self.flux[key] += self.spectrum[il] * self.dl * 1e-7 # nanom = 1e-7
            
            self.output = []
            b_full = self._sigma / np.pi * self.teff**4
            for il in range(self.il_max - 1):
                sp = self.spectrum[il]
                if sp <= 1: continue
                sp = np.log10(sp)
                lamb = self.lam[il] / 1e-7
                teff_loc = 2.9*1.e6/lamb

                b_l = 0
                expon = self.factor2 / (self.teff * self.lam[il])
                if expon < 150:
                    b_l = self.factor1 / (np.exp(expon) - 1) * self._invlam5[il]
                self.output += [[il, lamb, teff_loc, sp, b_l]]

            self.output = np.asarray(self.output, dtype = float)

            for key, val in self.flux.items():
                # In units of energy/time/area
                self.flux[key] = starsmashertools.lib.units.Unit(val, 'g/s*s*s')

            
            
            
            
            

    class Image(object):
        def __init__(
                self,
                resolution : list | tuple | np.ndarray,
                particle_array : list | tuple | np.ndarray | type(None) = None,
        ):
            self.resolution = resolution
            self.particle_array = np.asarray(particle_array)
            self.array = np.zeros(np.asarray(resolution) + 1)

        def add(self, ID, i, j, attenuation):
            self.array[i, j] += self.particle_array[ID] * attenuation

        def get_total(self):
            # Vectorized. Same as Natasha's way of doing it.
            return np.sum(0.25 * (\
                self.array[  :self.resolution[0]-1,  :self.resolution[1]-1 ] + \
                self.array[  :self.resolution[0]-1, 1:self.resolution[1]   ] + \
                self.array[ 1:self.resolution[0]  ,  :self.resolution[1]-1 ] + \
                self.array[ 1:self.resolution[0]  , 1:self.resolution[1]   ]))












@starsmashertools.preferences.use
class FluxResult(starsmashertools.helpers.nesteddict.NestedDict, object):
    def __init__(self, *args, **kwargs):
        self._simulation = None
        self._output = None
        super(FluxResult, self).__init__(*args, **kwargs)
    
    @property
    def simulation(self):
        if self._simulation is None:
            import starsmashertools
            self._simulation = starsmashertools.get_simulation(self['simulation'])
        return self._simulation

    @property
    def output(self):
        if self._output is None:
            self._output = starsmashertools.lib.output.Output(
                self['output'], self.simulation,
            )
        return self._output
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def save(
            self,
            filename : str = Pref('save.filename'),
            allowed : dict | starsmashertools.helpers.nesteddict.NestedDict = Pref('save.allowed'),
            **kwargs
    ):
        """
        Save the results to disk as an :class:`~.lib.archive.Archive`.
        
        Other Parameters
        ----------
        filename : str, default = Pref('save.filename')
            The name of the Archive.

        allowed : dict, :class:`~.helpers.nesteddict.NestedDict`, default = Pref('save.allowed')
            The items to include in the file. Only the keys of the dictionary
            are checked. If ``allowed`` contains a key which is also in this
            FluxResult, then the value is saved to the file. You can specify
            a nested dictionaries in this way, and only the keys given at the
            deepest nesting levels (the "flowers") are considered. See the 
            preferences file for an example.

        **kwargs
            Keyword arguments are passed directly to 
            :meth:`~.lib.archive.Archive.__init__`.
        
        Returns
        -------
        archive : :class:`~.lib.archive.Archive`
            The newly created Archive.

        See Also
        --------
        :meth:`~.load`
        """
        import starsmashertools.lib.archive
        
        if not isinstance(allowed, starsmashertools.helpers.nesteddict.NestedDict):
            allowed = starsmashertools.helpers.nesteddict.NestedDict(allowed)

        allowed_branches = allowed.branches()
        kwargs['auto_save'] = False
        archive = starsmashertools.lib.archive.Archive(filename, **kwargs)
        for branch, leaf in allowed.flowers():
            if branch not in allowed_branches: continue
            if not allowed[branch]: continue
            archive.add(
                str(branch),
                self[branch],
                origin = self['output'],
            )
        archive.save()

        return archive
    
    @staticmethod
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def load(
            filename : str,
            allowed : dict | starsmashertools.helpers.nesteddict.NestedDict = Pref('save.allowed'),
            deserialize : bool = True,
    ):
        """
        Load results from disk which were saved by :meth:`~.save`.

        Parameters
        ----------
        filename : str
            The location of the file on the disk to load.

        Other Parameters
        ----------------
        allowed : dict, :class:`~.helpers.nesteddict.NestedDict`, default = Pref('save.allowed')
            The same as ``allowed`` in :meth:`~.save`.

        deserialize : bool, default = True
            Given to :meth:`~.lib.archive.Archive.get`.

        Returns
        -------
        :class:`~.FluxResult`

        See Also
        --------
        :meth:`~.save`
        """
        import starsmashertools.lib.archive
        archive = starsmashertools.lib.archive.Archive(filename, readonly=True)

        if not isinstance(allowed, starsmashertools.helpers.nesteddict.NestedDict):
            allowed = starsmashertools.helpers.nesteddict.NestedDict(allowed)

        toload = starsmashertools.helpers.nesteddict.NestedDict()
        keys = [str(a) for a in allowed.branches()]
        for key, val in zip(keys, archive.get(keys, deserialize=deserialize)):
            try: key = eval(key)
            except: pass
            if deserialize: toload[key] = val.value
            else: toload[key] = val
        return FluxResult(toload)
    
    if has_matplotlib:
        @starsmashertools.helpers.argumentenforcer.enforcetypes
        @api
        def plot(
                self,
                ax : matplotlib.axes.Axes | type(None) = None,
                key : str | typing.Callable = 'flux',
                weighted_average : int | type(None) = None,
                log10 : bool = False,
                **kwargs
        ):
            """
            Creates a new :class:`~.mpl.artists.FluxPlot` and calls its 
            :meth:`~.mpl.artists.FluxPlot.imshow` method on the given Matplotlib
            :class:`matplotlib.axes.Axes`.
            
            Other Parameters
            ----------------
            ax : :class:`matplotlib.axes.Axes`, None, default = None
                The Matplotlib :class:`matplotlib.axes.Axes` to plot on. If 
                `None`, the plot is created on the axes returned by
                ``plt.gca()``.

            key : str, default = 'flux'
                The dictionary key in the ``['image']`` :py:class:`dict` to 
                obtain the data for plotting, or a function which accepts a 
                FluxResult as input and returns a 2D NumPy array and a dict, 
                where the 2D array is the image contents and the dict is the 
                keywords to pass to :func:`matplotlib.axes.Axes.imshow`. If 
                ``weighted_average`` is given, this argument is ignored.

            weighted_average : int, None, default = None
                The integer index of the array to plot from the 
                ``weighted_averages`` key in the results. If `None`, keyword 
                argument ``key`` is ignored.

            log10 : bool, default = False
                If `True`, the log10 operation will be done on the data.
            
            kwargs
                Other keyword arguments are passed directly to 
                :meth:`~.mpl.artists.FluxPlot.imshow`. Note, keyword ``origin`` 
                is always set to ``'lower'`` regardless of any value it has in
                ``kwargs``.

            Returns
            -------
            :class:`matplotlib.axes.AxesImage`, :class:`~.mpl.artists.FluxPlot`
                The :class:`matplotlib.axes.AxesImage` object created by 
                :meth:`~.mpl.artists.FluxPlot.imshow` and the 
                :class:`~.mpl.artists.FluxPlot` object which created it.
            
            See Also
            --------
            :class:`~.mpl.artists.FluxPlot`
            """
            import starsmashertools.mpl.artists
            import matplotlib.pyplot as plt
            
            if ax is None: ax = plt.gca()

            if weighted_average is not None:
                data = self['image']['weighted_averages'][weighted_average]
            else:
                if isinstance(key, str):
                    data = self['image'][key]
                elif callable(key):
                    data, kwargs = key(self)
                else:
                    raise Exception("This should never be possible")

            if log10:
                idx = data < 0
                if idx.any():
                    starsmashertools.helpers.warnings.warn(
                        'Some data is <= 0 and log10 = True. The data which is <= 0 will be set to NaN.',
                    )
                # It's common for the empty cells to have flux = 0, so we step
                # around the warnings by checking for <= 0 here.
                idx = data <= 0
                data[idx] = np.nan
                idx = np.isfinite(data)
                data[idx] = np.log10(data[idx])
                data[~idx] = np.nan
            
            ret = starsmashertools.mpl.artists.FluxPlot(ax, self)
            im = ret.imshow(data, **kwargs)
            return ret, im





@starsmashertools.preferences.use
class FluxResults(starsmashertools.helpers.nesteddict.NestedDict, object):
    """
    A container for multiple :class:`~.FluxResult` objects. Permits for multiple
    FluxResult objects to be saved in a single file, which can be convenient
    when working with many FluxResult objects.
    """
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(
            self,
            *args,
            allowed : dict | starsmashertools.helpers.nesteddict.NestedDict = Pref('allowed'),
            **kwargs
    ):
        """
        Constructor.
        
        Parameters
        ----------
        *args
            Positional arguments are passed directly to
            :meth:`~.helpers.nesteddict.NestedDict.__init__`.

        Other Parameters
        ----------------
        allowed : dict, :class:`~.helpers.nesteddict.NestedDict`, default = Pref('exclude')
            Items to include from each added :class:`~.FluxResult`. This cannot
            be changed after initialization.

        **kwargs
            Other keyword arguments are passed directly to 
            :meth:`~.helpers.nesteddict.NestedDict.__init__`.
        """
        if isinstance(allowed, dict):
            allowed = starsmashertools.helpers.nesteddict.NestedDict(allowed)
        self._allowed = allowed
        super(FluxResults, self).__init__(*args, **kwargs)

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def add(
            self,
            result : str | FluxResult,
            order : str | list | tuple = Pref('add.order'),
            deserialize : bool = True,
    ):
        """
        Add a :class:`~.FluxResult`. The values in the FluxResult will be
        inserted in a way which preserves the specified order.
        
        Parameters
        ----------
        result : str, FluxResult
            If a `str`, it must be a path to a saved :class:`~.FluxResult`. The
            FluxResult is loaded and its contents are added to this dictionary.


        Other Parameters
        ----------------
        order : str, list, tuple, default = Pref('add.order')
            The order with which to insert the FluxResult. If a `list` or 
            `tuple` are given, it refers to the nested key. For example, 
            ``['image', 'teff_aver']`` refers to the value found at
            ``FluxResult['image']['teff_aver']``. See 
            :class:`~.helpers.nesteddict.NestedDict` for details.

        deserialize : bool, default = True
            Given to :meth:`~.lib.archive.Archive.get` when loading the
            :class:`~.FluxResult` objects. If `False`, the values will be in
            their original binary format as stored on the disk. This can be
            helpful for operations such as condensing many :class:`~.FluxResult`
            objects into a single compressed file.

        See Also
        --------
        :class:`~.helpers.nesteddict.NestedDict`
        """
        import bisect
        
        if isinstance(result, str):
            result = FluxResult.load(
                result,
                allowed = self._allowed,
                deserialize = deserialize,
            )

        index = None
        if order in self.branches():
            index = bisect.bisect(self.get(order), result.get(order))

        for branch, leaf in result.flowers():
            if not self._allowed.get(branch, False): continue
            
            if branch not in self.branches(): self[branch] = [leaf]
            else:
                if index is None:
                    self[branch].insert(len(self[branch]), leaf)
                else:
                    self[branch].insert(index, leaf)
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def save(
            self,
            filename : str = Pref('save.filename'),
            **kwargs
    ):
        """
        Save the :class:`~.FluxResult` objects to disk. Each result is stored in
        the same format as :meth:`~.FluxResult.save`, except stacked in a list.
        For example, the ``'image'`` key in the archive will be a list, where 
        each element is the ``'image'`` key from the corresponding FluxResult.
        
        Other Parameters
        ----------
        filename : str, default = Pref('save.filename')
            The path to the file to create.
        
        **kwargs
            Other keyword arguments are passed directly to 
            :meth:`~.lib.archive.Archive.__init__`.

        Returns
        -------
        archive : :class:`~.lib.archive.Archive`
            The newly created Archive.

        See Also
        --------
        :meth:`~.load`
        """
        import starsmashertools.lib.archive
        import time
        
        kwargs['auto_save'] = False
        archive = starsmashertools.lib.archive.Archive(filename, **kwargs)
        mtime = time.time()
        for branch, leaf in self.flowers():
            archive.add(str(branch), leaf, mtime = mtime)
        archive.save()

    @staticmethod
    def load(
            filename : str,
            deserialize : bool = True,
    ):
        """
        Load the :class:`~.FluxResult` objects from disk. Each result is stored
        in the same format as :meth:`~.FluxResult.save`, except stacked in a 
        list. For example, the ``'image'`` key in the archive will be a list, 
        where each element is the ``'image'`` key from the corresponding 
        FluxResult.
        
        Other Parameters
        ----------
        filename : str
            The path to the file to load.

        deserialize : bool, default = True
            Convert :class:`~.lib.archive.ArchiveValue` objects in each list to
            whatever their real values should be. Set this to `True` if you
            used ``deserialize=False`` in :meth:`~.add`.
        
        Returns
        -------
        fluxresults : :class:`~.FluxResults`
            The `FluxResult` objects in the same order as stored in the archive.
        
        See Also
        --------
        :meth:`~.save`
        """
        import starsmashertools.lib.archive
        archive = starsmashertools.lib.archive.Archive(filename, readonly=True)
        toload = starsmashertools.helpers.nesteddict.NestedDict()
        for key, val in archive.items():
            try: key = eval(key)
            except: pass
            
            if deserialize:
                toload[key] = [starsmashertools.lib.archive.ArchiveValue.deserialize(
                    key,
                    v,
                ) for v in val.value]
            else:
                toload[key] = val.value
        return FluxResults(toload)


