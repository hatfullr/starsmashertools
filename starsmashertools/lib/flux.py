# I hate how this is written, but at least it works specifically for us...
import starsmashertools.preferences
from starsmashertools.preferences import Pref
import math
import numpy as np
import starsmashertools.lib.output
import starsmashertools.lib.units
from starsmashertools.lib.units import constants
from starsmashertools.helpers.apidecorator import api
import starsmashertools.helpers.argumentenforcer
import starsmashertools.helpers.nesteddict
import copy
import warnings
import typing
import collections
import gc
import pickle
import gzip
import tempfile
import os

try:
    import matplotlib.axes
    has_matplotlib = True
except ImportError:
    has_matplotlib = False


@starsmashertools.preferences.use
class FluxFinder(object):
    r"""
    This class is used to calculate flux images for StarSmasher simulations in
    Hatfull et al. (2024) that use radiative cooling.
        
    Attributes
    ----------
    result : :class:`~.FluxResult`
        Contains information about the images, spectrum, and more. It is filled
        with values only after calling :meth:`~.get`\.
    """
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(
            self,
            output : starsmashertools.lib.output.Output,
            
            # Image
            length_scale : float | int = Pref('length_scale', 1.),
            resolution : list | tuple | np.ndarray = Pref('resolution', (400, 400)),
            extent : list | tuple | np.ndarray | type(None) = Pref('extent', None),
            extent_limits : list | tuple | np.ndarray | type(None) = Pref('extent_limits', None),
            theta : float | int = Pref('theta', 0.),
            phi : float | int = Pref('phi', 0.),
            fluffy : bool = Pref('fluffy', False),
            rays : bool = Pref('rays', True),
            weighted_averages : list | tuple = Pref('weighted_averages', []),

            # Limits
            tau_s : float | int = Pref('tau_s', 20.),
            tau_skip : float | int = Pref('tau_skip', 1.e-5),
            flux_limit_min : float | int | type(None) = Pref('flux_limit_min', None),

            # Dust
            dust_opacity : float | int | type(None) = Pref('dust_opacity', 1),
            dust_Trange : list | tuple | type(None) = Pref('dust_Trange', (100, 1000)),

            # Spectrum
            spectrum_size : int = Pref('spectrum_size', 1000),
            lmax : float | int = Pref('lmax', 3000),
            dl_il : float | int = Pref('dl_il', 3.),
            dl : float | int = Pref('dl', 10),
            l_range : list | tuple = Pref('l_range', (10, 3000)),
            factor1 : float | int = Pref('factor1', 1.191045e-05),
            factor2 : float | int = Pref('factor2', 1.438728e+00),
            filters : dict = Pref('filters', {'V':[500,600], 'R':[590,730], 'I':[720,880]}),

            # Debugging
            verbose : bool = False,
            test_ray : bool = False,
            itr : int = -1,
            jtr : int = -1,
            test_particle : bool = False,
            ic : int = -1,
    ):
        r"""
        Parameters
        ----------
        output : :class:`~.lib.output.Output`
            The StarSmasher output file to process. It must have all of the 
            following keys:

            * ``ntot``\: Total number of particles.
            * ``t``\: Simulation time.
            * ``x``\: Particle x positions.
            * ``y``\: Particle y positions.
            * ``z``\: Particle z positions.
            * ``am``\: Particle masses.
            * ``hp``\: Particle smoothing lengths. It is assumed that the kernel function used in the simulation had compact support 2h. 
            * ``rho``\: Particle densities.
            * ``u``\: Particle specific internal energies.
            * ``temperatures``\: Particle temperature.
            * ``popacity``\: Particle opacity.
            * ``dEemergdt``\: :math:`dE_\mathrm{emerg}/dt` as in Equations (27) or (36)
            * ``dEdiffdt``\: :math:`dE_\mathrm{diff}/dt` as in Equations (27) or (36)

        Other Parameters
        ----------------
        length_scale : float, int, default = ``Pref('length_scale', None)``
            The domain of the images will be in units of centimeters, divided by
            ``length_scale``\.

        resolution : list, tuple, :class:`numpy.ndarray`\, default = ``Pref('resolution', (400, 400))``
            The resolution of the images.

        extent : list, tuple, :class:`numpy.ndarray`\, default = ``Pref('extent', None)``
            The physical domain of the image as ``[xmin,xmax,ymin,ymax]``. If 
            `None`\, the domain will be set automatically to fit all particles.
        
        extent_limits : list, tuple, :class:`numpy.ndarray`\, default = ``Pref('extent_limits', None)``
            If ``extent=None``\, the automatically determined extent will be
            limited to extend no further than the values given in this array, of
            content ``[xmin,xmax,ymin,ymax]``\. Any of the values can be 
            `None`\, in which case the corresponding extent will fit to the edge
            of the particles as normal.

        theta : float, int, default = ``Pref('theta', 0.)``
            The azimuthal viewing angle, equivalent to :math:`90^\circ-i`\, 
            where :math:`i` is the inclination angle from the orbital plane. 
            Sets ``xangle`` in :meth:`~.lib.output.Output.rotate`\.

        phi : float, int, default = ``Pref('phi', 0.)``
            The polar viewing angle. Sets ``zangle`` in 
            :meth:`~.lib.output.Output.rotate`\.

        fluffy : bool, default = ``Pref('fluffy', False)``
            Images are usually generated by considering the particles as members
            of the "dense" cooling regime in Hatfull et al. (2024). However, you
            can have them considered inthe "fluffy" cooling regime instead by
            setting ``fluffy=True``\. Doing so is not recommended.

        rays : bool, default = ``Pref('rays', True)``
            If ``True``\, the number of cooling rays by which a particle has 
            undergone cooling is dynamically determined, and the flux values of
            the particles are adjusted accordingly.

        weighted_averages : list, tuple, default = []
            Compute images of flux-weighted averages. The elements in this list
            can either be str or an iterable array with the same length as the
            number of particles. Strings are used as keys to obtain arrays from
            the output file. The results are stored in 
            ``('image', 'weighted_averages')`` in the :class:`~.FluxResult` and
            corresponds with quantity :math:`H_\mathrm{cell}`\. Thus, to get the
            value of :math:`\langle H_\mathrm{rad}\rangle`\, as in Equation 
            (45), you must multiply by ``('image', 'flux')``\, get the sum, and
            then divide that quantity by ``('image', 'flux_tot')``\.

        tau_s : float, int, default = ``Pref('tau_s', 20.)``
            For each individual image ray, when :math:`\tau_\mathrm{ray}` 
            exceeds this value, no further flux contributions from particles 
            deeper into the fluid are considered.
        
        tau_skip : float, int, default = ``Pref('tau_skip', 1.e-5)``
            Particles with optical depth less than this value are ignored.

        flux_limit_min : float, int, None, default = ``Pref('flux_limit_min', None)``
            Particles with flux values less than this will be ignored. If 
            `None`\, then no particles will be ignored based on their flux 
            values.

        dust_opacity : float, int, None, default = ``Pref('dust_opacity', 1.)``
            The artificial dust opacity :math:`\kappa_\mathrm{dust}` in units of
            :math:`\mathrm{cm}^2\,\mathrm{g}^{-1}`\. If `None` then artificial
            dust will not be used.

        dust_Trange : list, tuple, None, default = ``Pref('dust_Trange', (100, 1000))``
            The temperature range within which particles will be assigned 
            opacities equal to ``dust_opacity``\.

        spectrum_size : int, default = ``Pref('spectrum_size', 1000)``
            The length of the array of bins for the spectrum.
        
        lmax : float, int, default = ``Pref('lmax', 3000)``
            In part sets the value of ``il_max`` in the spectrum, where 
            ``il_max = math.flooat(lmax / dl_il)``\.
        
        dl_il : float, int, defualt = ``Pref('dl_il', 3.)``
            In part sets the value of ``il_max`` in the spectrum, where 
            ``il_max = math.flooat(lmax / dl_il)``\.
        
        dl : float, int, default = ``Pref('dl', 10)``
            The size of the spectrum bins.

        l_range : list, tuple, default = ``Pref('l_range', (10, 3000))``
            The range of the spectrum.

        factor1 : float, int, default = ``Pref('factor1', 1.191045e-05)``
            A numerical factor used in the spectrum.

        factor2 : float, int, default = ``Pref('factor2', 1.438728e+00)``
            A numerical factor used in the spectrum.
        
        filters : dict, default = ``Pref('filters', {'V':[500,600], 'R':[590,730], 'I':[720,880]})``
            Optical filter ranges in nanometers for the spectrum.
        """
        
        _kwargs = locals()
        _kwargs.pop('self')
        _kwargs.pop('output')
        
        self.output = copy.deepcopy(output)
        del output

        # Check that the output file has all the required information
        keys = list(self.output.keys())
        missing_keys = []
        for key in ['ntot', 't', 'x', 'y', 'z', 'am', 'hp', 'rho', 'u', 'temperatures', 'popacity', 'dEemergdt', 'dEdiffdt']:
            if key in keys: continue
            missing_keys += [key]
        if 'uraddot' not in keys:
            if 'uraddotcool' not in keys and 'uraddotheat' not in keys:
                missing_keys += ['uraddot OR (uraddotcool and uraddotheat)']
        
        if missing_keys:
            raise KeyError("Output file '%s' is missing required keys: %s" % (str(self.output), str(missing_keys)))

        self.units = self.output.simulation.units
        
        self.length_scale = length_scale
        self.resolution = resolution
        self.extent = extent
        self.extent_limits = extent_limits
        self.theta = theta
        self.phi = phi
        self.fluffy = fluffy
        self.rays = rays
        self.tau_s = tau_s
        self.tau_skip = tau_skip
        self.dust_opacity = dust_opacity
        self.dust_Trange = dust_Trange
        self.flux_limit_min = flux_limit_min
        
        self.verbose = verbose
        self.test_ray = test_ray
        self.itr = itr
        self.jtr = jtr
        self.test_particle = test_particle
        self.ic = ic

        # Set the viewing angle
        self.output.rotate(xangle = theta, zangle = phi)
        
        self._rloc = np.full(self.output['ntot'], np.nan)
        
        self.spectrum = FluxFinder.Spectrum(
            spectrum_size,
            lmax,
            dl_il,
            dl,
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

        # We multiply by the flux later in _init_particles
        for i, array in enumerate(weighted_averages):
            if isinstance(array, str):
                self.images['weightedaverage%d' % i] = FluxFinder.Image(
                    self.resolution,
                    # This should hopefully send a reference to the image for
                    # use. This way, if the user wants to use, say, a
                    # component of the particle velocity, then the value used
                    # will be the one after rotations have been done.
                    particle_array = self.output[array],
                )
            else:
                self.images['weightedaverage%d' % i] = FluxFinder.Image(
                    self.resolution,
                    particle_array = array, # a reference (rotate)
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

    @api
    def get(self):
        r"""
        Initialize values and process the images. Fills the ``result`` attribute
        with values.
        """
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

        # Vectorized:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            teff_aver = np.sum(surf_t[:self.resolution[0]-1, :self.resolution[1]-1] * area_br / flux_tot)

        # Units of energy/time/area
        flux_tot = starsmashertools.lib.units.Unit(flux_tot, 'g/s*s*s')
        
        # Finish the ltot calculations
        cell_area = self.dx * self.dy * self.length_scale**2
        ltot = 4 * cell_area * flux_tot
        
        self.spectrum.finalize()
        
        ltot_spectrum = 4 * cell_area * self.spectrum.flux['total']
        l_spectrum = {key:4 * cell_area * val for key, val in self.spectrum.flux.items() if key != 'total'}

        self.result['image','Teff_cell'] = surf_t
        self.result['image','teff_aver'] = teff_aver
        self.result['image','ltot'] = ltot
        self.result['image','flux_tot'] = flux_tot
        self.result['spectrum','ltot'] = ltot_spectrum
        self.result['spectrum','luminosities'] = l_spectrum
        self.result['spectrum','output'] = self.spectrum.output
        self.result['spectrum','teff'] = self.spectrum.teff
        self.result['spectrum','teff_min'] = self.spectrum.teff_min
        self.result['spectrum','teff_max'] = self.spectrum.teff_max

        del area_br, surf_t, flux_tot, teff_aver, cell_area
        del ltot, ltot_spectrum, l_spectrum
        gc.collect()
        

    def _init_particles(self):
        # Below is a much faster alternative to looping through particles to set
        # values.
        c_speed = float(constants['c'])
        sigma_const = float(constants['sigmaSB'])
        a_const = float(constants['a'])

        kappa = self.output['popacity'] * float(self.units['popacity'])
        if self.dust_opacity is not None:
            kappa = self._apply_dust(kappa)

        # If the code version has 'uraddotheat' then it was written with heating
        # considered. 
        if 'uraddotheat' in self.output.keys():
            if (self.output['uraddotheat'] == 0).all(): # No heating
                uraddotcool = self.output['uraddot'] * float(self.units['uraddot'])
            else: # Heating
                uraddotcool = self.output['uraddotcool'] * float(self.units['uraddotcool'])
        else:
            # Otherwise, it's an earlier version of the code where only uraddot
            # was considered and there was no heating
            uraddotcool = self.output['uraddot'] * float(self.units['uraddot'])
        
        uraddot_emerg = self.output['dEemergdt'] * float(self.units['dEemergdt'])
        uraddot_diff = self.output['dEdiffdt'] * float(self.units['dEdiffdt'])
        
        self._flux = np.zeros(self.output['ntot'], dtype = float)
        self._teff = np.zeros(self.output['ntot'], dtype = float)
        
        flux_rad = np.zeros(self._flux.shape)
        flux_em = np.zeros(self._flux.shape)

        idx = np.full(self.output['ntot'], False)
        if self.fluffy:
            idx = uraddot_emerg < uraddot_diff
            
        cores = self.output['u'] == 0

        self._rloc[cores] = 0

        m = self.output['am'] * float(self.units['am'])
        rho = copy.deepcopy(self.output['rho']) * float(self.units['rho'])

        self._tau = np.zeros(self.output['ntot'])
        
        i = idx & ~cores
        # if uraddot_emerg[i] < uraddot_diff[i] and do_fluffy == 1:
        if self.fluffy and i.any():
            self._rloc[i] = 2.*self.output['hp'][i] * float(self.units['hp'])
            rho[i]=m[i]/pow(self._rloc[i],3.)
            self._tau[i]=rho[i]*kappa[i]*self._rloc[i]*4./3.
            flux_em[i] = uraddot_emerg[i]*0.25/(np.pi*self._rloc[i]**2)
            self._flux[i]=flux_em[i]
            flux_rad[i] = flux_em[i]

        temp = self.output['temperatures'] * float(self.units['temperatures'])
        i = ~idx & ~cores
        # else:
        if not self.fluffy or i.any():
            # this particle is considered in "dense" approximation
            self._rloc[i]=pow(m[i]/(4/3.*np.pi)/rho[i],0.33333)
            self._tau[i]=rho[i]*kappa[i]*self._rloc[i]*4./3.
            ad      = 4.*np.pi*self._rloc[i]**2
            ad_cool = 2.*np.pi*self._rloc[i]**2
            flux_em[i] = uraddot_emerg[i]/ad
            dd=c_speed/kappa[i]/rho[i]
            gradt=a_const*pow(temp[i],4)/self._rloc[i]
            dedt=ad/3.*dd*gradt/m[i]

            flux_rad[i] = dedt * m[i] / ad
            self._flux[i] = dedt * m[i] / ad
            
            if self.rays:
                ratio = np.full(self._flux.shape, np.nan)
                ratio[i] = np.abs(uraddotcool[i]/dedt*12.) # to figure out how many effective cooling rays
                _idx1 = ~cores & i & (ratio <= 6)
                # if ratio <= 6:
                if _idx1.any():
                    ad_cool = 2.*np.pi*self._rloc[_idx1]*self._rloc[_idx1]
                    self._flux[_idx1]=np.abs(uraddotcool[_idx1]*m[_idx1]/ad_cool) # all what is cooled is to goes through the outer hemisphere
                
                _idx2 = ~cores & i & (ratio > 6)
                # if ratio > 6:
                if _idx2.any():
                    ad      = 4.*np.pi*self._rloc[_idx2]*self._rloc[_idx2]
                    gradt = a_const*pow(temp[_idx2],4)/self._rloc[_idx2]
                    dd = c_speed/kappa[_idx2]/rho[_idx2]
                    dedt = ad/3.*dd*gradt/m[_idx2]
                    self._flux[_idx2] = dedt*m[_idx2]/ad

                del ratio, _idx1, _idx2

            _idx = ~cores & i & (self._tau < 100) # we anticipate that emergent flux computation breaks down at very large tau, where flux_em becomes zero and nullifies total flux is no limit on tau is placed
            if _idx.any():
                self._flux[_idx] = np.minimum(self._flux[_idx], flux_em[_idx])

            del ad, ad_cool, dd, gradt, dedt, _idx


        self._teff[~cores] = (flux_rad[~cores] / sigma_const)**0.25

        i = ~cores & (flux_rad > flux_em) & (self._tau < 100)
        if i.any():
            self._teff[i] = temp[i]

        self._rloc /= self.length_scale
        
        self.images['flux'].particle_array = self._flux

        # Finish adding the weights for the averages
        for key in self.images.keys():
            if 'weightedaverage' not in key: continue
            self.images[key].particle_array *= self._flux

        self.result['particles','contributing_IDs'] = np.arange(self.output['ntot'], dtype=int)
        self.result['particles','rloc'] = self._rloc
        self.result['particles','tau'] = self._tau
        self.result['particles','kappa'] = kappa
        self.result['particles','flux'] = self._flux

        del c_speed, sigma_const, a_const, kappa, uraddotcool
        del uraddot_emerg, uraddot_diff, flux_rad, flux_em, idx
        del cores, m, rho, i, temp
        gc.collect()

    def _init_grid(self):
        if self.extent is None:
            # Get particle quantities in CGS
            x = self.output['x'] * float(self.units['x'])
            y = self.output['y'] * float(self.units['y'])
            m = self.output['am'] * float(self.units['am'])
            rho = self.output['rho'] * float(self.units['rho'])
            h = self.output['hp'] * float(self.units['hp'])

            delta = 2 * h
            if not self.fluffy:
                cores = self.output['u'] == 0
                if (~cores).any():
                    delta[~cores] = (0.75*m[~cores]/(np.pi*rho[~cores]))**0.33333
                del cores

            xmin = np.amin(x - delta) / self.length_scale
            xmax = np.amax(x + delta) / self.length_scale
            ymin = np.amin(y - delta) / self.length_scale
            ymax = np.amax(y + delta) / self.length_scale

            max_coord=max(abs(xmin),abs(xmax), abs(ymin), abs(ymax))
            xmax=max_coord
            ymax=max_coord
            ymin=-max_coord
            xmin=-max_coord
            self.extent = [xmin, xmax, ymin, ymax]
            if self.verbose: print("domain  max: %12.5f" %  max_coord)
            del x, y, m, rho, h, max_coord, delta
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

    @api
    def find_dust(
            self,
            T : list | tuple | np.ndarray | type(None) = None,
            kappa : list | tuple | np.ndarray | type(None) = None,
    ):
        if T is None:
            T = self.output['temperatures'] * float(self.units['temperatures'])
        if kappa is None:
            kappa = self.output['popacity'] * float(self.units['popacity'])
        T = np.asarray(T)
        kappa = np.asarray(kappa)
        
        if self.dust_opacity is None: # No dust
            return np.full(T.shape, False, dtype = bool)

        T_dust_min, T_dust_max = self.dust_Trange
        if T_dust_min is None: T_dust_min = 0
        if T_dust_max is None: T_dust_max = np.nanmax(T[np.isfinite(T)])

        return (T_dust_min < T) & (T < T_dust_max) & (kappa < self.dust_opacity)

    def _apply_dust(self, kappa):
        idx = self.find_dust(kappa = kappa)
        
        # Apply the dust, if there is any
        if idx.any(): kappa[idx] = self.dust_opacity

        return kappa

    def _process_images(self):
        xmin, xmax, ymin, ymax = self.extent
        
        x = self.output['x'] * float(self.units['x']) / self.length_scale
        y = self.output['y'] * float(self.units['y']) / self.length_scale
        z = self.output['z'] * float(self.units['z']) / self.length_scale

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
        flux_min = float('inf')
        flux_max = -float('inf')

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
            flux_min = min(flux_min, self._flux[i])
            flux_max = max(flux_max, self._flux[i])

            for ii, jj in grid_indices[_slice][idx]:
                ray_id[ii, jj].append(i)
                ray_n[ii, jj] += 1

        
        # Much more efficient calculations below
        if self.verbose:
            max_ray = np.amax(ray_n)
            idx_maxray = np.argmax(ray_n.flatten())
            i_maxray, j_maxray = np.unravel_index(idx_maxray, ray_n.shape)

            print("maximum number of particles above the cut off optically thick surface is %d %d %d"%(max_ray,i_maxray,j_maxray))            
            print('minimum tau account for is  %f' % tau_min)
            print('minimum flux account for is %le' % flux_min)
            print('maximum flux account for is %le' % flux_max)
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
                    
                    flux_from_contributors[ir] += f
                    self.spectrum.add(f, self._teff[ir])

                    tau_ray += self._tau[ir]
                    
                if i > 0:
                    attenuation = math.exp(-tau_ray)
                    f = self._flux[i] * math.exp(-tau_ray)
                    for key, image in self.images.items():
                        image.add(i, ii, jj, attenuation)

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
        self.result['image','surf_d'] = surf_d
        self.result['image','surf_id'] = surf_id
        self.result['image','ray_n'] = ray_n
        self.result['image','weighted_averages'] = [image.array for key, image in self.images.items() if 'weightedaverage' in key]

        del xmin,xmax,ymin,ymax,x,y,z,flux_from_contributors
        del shape,ray_id,ray_n,surf_d,surf_id,grid_indices,ijloc_arr
        del rloc,ijminmax_arr,mask,mask1,dtype,array,tau_min,mask2
        del zipped,idx,contributors
        gc.collect()

        



    class Spectrum(object):
        @api
        def __init__(
                self,
                spectrum_size,
                lmax,
                dl_il,
                dl,
                l_range,
                factor1,
                factor2,
                filters,
                ntot,
        ):
            r"""
            Holds spectrum information.
            """
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
            #ltot_spectrum = 0
            # [0.00000000e+00 7.39256413e-21 2.00355212e-07 6.82084616e-01
            #  4.39889814e+03 1.28926397e+06 6.63126098e+07 1.17342608e+09
            #  1.04031610e+10 5.80568647e+10]
            
            self.flux = {
                'total' : np.sum(self.spectrum[:self.il_max - 1] * self.dl * 1e-7),
            }
            for key in self.filters.keys(): self.flux[key] = 0.
            
            max_flux = -10.
            il_flux = 0
            for il in range(self.il_max - 1):
                sp = self.spectrum[il]
                #ltot_spectrum += sp * self.dl * 1e-7 # nanom = 1e-7
                if sp <= 0: continue
                
                sp = np.log10(sp)
                lamb = self.lam[il] / 1e-7 # nanom = 1e-7
                teff_loc = 2.9*1.e6/lamb
                if sp > max_flux:
                    max_flux = sp
                    il_flux = il
                    self.teff = teff_loc
                
                for key, (lamb_low, lamb_hi) in self.filters.items():
                    if lamb_low is not None and lamb < lamb_low: continue
                    if lamb_hi is not None and lamb > lamb_hi: continue

                    self.flux[key] += self.spectrum[il] * self.dl * 1e-7 # nanom = 1e-7
            
            self.teff_min = self.teff
            self.teff_max = self.teff
            min_flux = 0.95 * 10**max_flux
            for il in range(il_flux, 1, -1):
                if self.spectrum[il] < min_flux: break
                self.teff_max = 2.9 * 1.e6 / self.lam[il] * 1.e-7
            
            for il in range(il_flux, self.il_max - 1):
                if self.spectrum[il] < min_flux: break
                self.teff_min = 2.9 * 1.e6 / self.lam[il] * 1.e-7
            
            self.output = []
            b_full = self._sigma / np.pi * self.teff**4
            for il in range(self.il_max - 1):
                sp = self.spectrum[il]
                if sp <= 0: continue
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
        @api
        def __init__(
                self,
                resolution : list | tuple | np.ndarray,
                particle_array : list | tuple | np.ndarray | type(None) = None,
        ):
            r"""
            A class wrapper for individual images.
            """
            self.resolution = resolution
            self.particle_array = np.asarray(particle_array)
            self.array = np.zeros(np.asarray(resolution) + 1)

        def add(self, ID, i, j, attenuation):
            self.array[i, j] += self.particle_array[ID] * attenuation

        def get_total(self):
            # Vectorized. Same as Natasha's way of doing it.
            return np.sum(0.25 * (\
                self.array[  :self.resolution[0]-1,  :self.resolution[1]-1] + \
                self.array[  :self.resolution[0]-1, 1:self.resolution[1]  ] + \
                self.array[ 1:self.resolution[0]  ,  :self.resolution[1]-1] + \
                self.array[ 1:self.resolution[0]  , 1:self.resolution[1]  ]))












@starsmashertools.preferences.use
class FluxResult(starsmashertools.helpers.nesteddict.NestedDict, object):
    def __init__(self, *args, **kwargs):
        self._simulation = None
        self._output = None
        super(FluxResult, self).__init__(*args, **kwargs)
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def save(
            self,
            filename : str = Pref('save.filename'),
            allowed : dict | starsmashertools.helpers.nesteddict.NestedDict = Pref('save.allowed'),
    ):
        r"""
        Save the results to disk as a compressed binary file.
        
        Other Parameters
        ----------
        filename : str, default = ``Pref('save.filename')``
            The name of the file.

        allowed : dict, :class:`~.helpers.nesteddict.NestedDict`\, default = ``Pref('save.allowed')``
            The items to include in the file. Only the keys of the dictionary
            are checked. If ``allowed`` contains a key which is also in this
            FluxResult, then the value is saved to the file. You can specify
            a nested dictionaries in this way, and only the keys given at the
            deepest nesting levels (the "flowers") are considered. See the 
            preferences file for an example.

        See Also
        --------
        :meth:`~.load`
        """
        if isinstance(allowed, dict):
            allowed = starsmashertools.helpers.nesteddict.NestedDict(allowed)
        
        branches = self.branches()
        towrite = starsmashertools.helpers.nesteddict.NestedDict()
        for branch, leaf in allowed.flowers():
            if not leaf: continue
            if branch in self.stems():
                for b, l in self.flowers(stems = (branch,)):
                    towrite[b] = self[b]
            elif branch in branches:
                towrite[branch] = self[branch]

        with gzip.open(filename, 'wb') as f:
            pickle.dump(towrite, f)
        
    @staticmethod
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def load(
            filename : str,
            allowed : dict | starsmashertools.helpers.nesteddict.NestedDict | type(None) = None,
    ):
        r"""
        Load results from disk which were saved by :meth:`~.save`\.

        Parameters
        ----------
        filename : str
            The location of the file on the disk to load.

        Other Parameters
        ----------------
        allowed : dict, :class:`~.helpers.nesteddict.NestedDict`\, None, default = None
            Similar to ``allowed`` in :meth:`~.save`\. If `None` is given, then
            all the values will be loaded into the returned 
            :class:`~.FluxResult` object.

        Returns
        -------
        :class:`~.FluxResult`

        See Also
        --------
        :meth:`~.save`
        """
        if isinstance(allowed, dict):
            allowed = starsmashertools.helpers.nesteddict.NestedDict(allowed)
            
        with gzip.open(filename, 'rb') as f:
            loaded = pickle.load(f)
        
        if allowed:
            for branch, leaf in allowed.flowers():
                if leaf: continue
                if branch not in loaded: continue
                loaded.pop(branch)
        return FluxResult(loaded)
    
    if has_matplotlib:
        @starsmashertools.helpers.argumentenforcer.enforcetypes
        @api
        def plot(
                self,
                ax : matplotlib.axes.Axes | type(None) = None,
                key : str | list | tuple | typing.Callable = 'flux',
                weighted_average : int | type(None) = None,
                log10 : bool = False,
                **kwargs
        ):
            r"""
            Creates a new :class:`~.mpl.artists.FluxPlot` and calls its 
            :meth:`~.mpl.artists.FluxPlot.imshow` method on the given Matplotlib
            :class:`matplotlib.axes.Axes`\.
            
            Other Parameters
            ----------------
            ax : :class:`matplotlib.axes.Axes`\, None, default = None
                The Matplotlib :class:`matplotlib.axes.Axes` to plot on. If 
                `None`\, the plot is created on the axes returned by
                ``plt.gca()``\.

            key : str, list, tuple, :class:`typing.Callable`\, default = 'flux'
                The dictionary key in the ``['image']`` :py:class:`dict` to 
                obtain the data for plotting, or a function which accepts a 
                FluxResult as input and returns a 2D NumPy array and a dict, 
                where the 2D array is the image contents and the dict is the 
                keywords to pass to :meth:`matplotlib.axes.Axes.imshow`\. If 
                ``weighted_average`` is given, this argument is ignored.

            weighted_average : int, None, default = None
                The integer index of the array to plot from the 
                ``weighted_averages`` key in the results. If `None`\, keyword 
                argument ``key`` is ignored.

            log10 : bool, default = False
                If `True`\, :meth:`numpy.log10` will be called on the data.
            
            kwargs
                Other keyword arguments are passed directly to 
                :meth:`~.mpl.artists.FluxPlot.imshow`\. Note, keyword ``origin`` 
                is always set to ``'lower'`` regardless of any value it has in
                ``kwargs``\.

            Returns
            -------
            :class:`matplotlib.axes.AxesImage`\, :class:`~.mpl.artists.FluxPlot`
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
    r"""
    A container for multiple :class:`~.FluxResult` objects. Permits for multiple
    :class:`~.FluxResult` objects to be saved in a single file, which can be 
    convenient when working with many :class:`~.FluxResult` objects.
    """
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(
            self,
            *args,
            allowed : dict | starsmashertools.helpers.nesteddict.NestedDict = Pref('allowed'),
            **kwargs
    ):
        r"""
        Constructor.
        
        Parameters
        ----------
        *args
            Positional arguments are passed directly to
            :meth:`~.helpers.nesteddict.NestedDict.__init__`\.

        Other Parameters
        ----------------
        allowed : dict, :class:`~.helpers.nesteddict.NestedDict`\, default = ``Pref('exclude')``
            Items to include from each added :class:`~.FluxResult`\. This cannot
            be changed after initialization.

        **kwargs
            Other keyword arguments are passed directly to 
            :meth:`~.helpers.nesteddict.NestedDict.__init__`\.
        """
        if isinstance(allowed, dict):
            allowed = starsmashertools.helpers.nesteddict.NestedDict(allowed)
        self._allowed = allowed
        super(FluxResults, self).__init__(*args, **kwargs)

    def is_allowed(self, branch):
        for b, l in self._allowed.flowers():
            if not l: continue
            if branch == b or starsmashertools.helpers.nesteddict.NestedDict.is_child_branch(branch, b):
                return True
        return False

    def add(self, result : str | FluxResult):
        if isinstance(result, str):
            result = FluxResult.load(result)
        
        for branch, leaf in result.flowers():
            if not self.is_allowed(branch): continue
            
            if branch not in self: self[branch] = [leaf]
            else: self[branch] += [leaf]

    def save(self, filename : str):
        import starsmashertools.helpers.path
        if starsmashertools.helpers.path.exists(filename):
            tname = None
            try:
                with tempfile.NamedTemporaryFile(dir = os.getcwd(), delete = False) as output:
                    tname = output.name
                    _input = gzip.GzipFile(mode = 'wb', fileobj = output)
                    pickle.dump(self, _input)
                    _input.flush()
                    _input.close()
            except:
                # temporary files are always local
                if tname is not None and os.path.exists(tname): os.remove(tname)
                raise
            else:
                starsmashertools.helpers.path.rename(tname, filename)
        else:
            try:
                with gzip.open(filename, 'wb') as f:
                    pickle.dump(self, f)
            except:
                # the gzip method is always local
                if os.path.exists(filename): os.remove(filename)
                raise

    @staticmethod
    def load(filename : str):
        with gzip.open(filename, 'rb') as f:
            return pickle.load(f)
