# Included here are useful common functions
import starsmashertools.preferences
from starsmashertools.preferences import Pref
import numpy as np
from starsmashertools.helpers.apidecorator import api
import starsmashertools.helpers.argumentenforcer

try:
    import matplotlib.axes
    has_matplotlib = True
except ImportError:
    has_matplotlib = False

# Given the mass of a collection of particles and coordinates in args, return
# the center of mass in each of those args
@api
def center_of_mass(m, *args):
    total_mass = np.sum(m)
    result = np.zeros(len(args))
    for i, arg in enumerate(args):
        result[i] = np.sum(arg * m) / total_mass
    return result

@api
def center_of_particles(*args):
    total_particles = len(args[0])
    result = np.zeros(len(args))
    for i, arg in enumerate(args):
        result[i] = np.sum(arg) / total_particles
    return result


@api
def period(m1,m2,separation):
    import starsmashertools.lib.units
    G = float(starsmashertools.lib.units.constants['G'])
    return np.sqrt(4 * np.pi**2 / (G * (m1 + m2)) * separation**3)


# The Roche lobe radius
@api
def rocheradius(m1,m2,separation):
    q = m1 / m2
    qonethird = q**(1./3.)
    qtwothirds = qonethird * qonethird
    return 0.49*qtwothirds / (0.6*qtwothirds + np.log(1. + qonethird)) * separation

# Given data 'x' and 'y', linearly interpolate the position(s) 'x0'.
@api
def linear_interpolate(x, y, x0):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape != y.shape:
        raise Exception("Inputs 'x' and 'y' must be the same shape")
    
    if not isinstance(x0, str) and not hasattr(x0, '__iter__'): x0 = [x0]
    x0 = np.asarray(x0)

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]
    
    lenx = len(x)

    ret = np.empty(len(x0))
    
    for i, _x in enumerate(x0):
        idx_closest = np.argmin(np.abs(x - _x))
        xclosest = x[idx_closest]
        if _x > xclosest:
            left = idx_closest
            if idx_closest + 1 > lenx - 1: right = -1
            else: right = idx_closest + 1
        elif _x < xclosest:
            right = idx_closest
            if idx_closest - 1 < 0: left = 0
            else: left = idx_closest - 1
        else:
            ret[i] = y[idx_closest]
            continue
        _x0 = x[left]
        _x1 = x[right]
        _y0 = y[left]
        _y1 = y[right]
        ret[i] = (_y0 * (_x1 - _x) + _y1 * (_x - _x0)) / (_x1 - _x0)
    return ret


@api
def rotate(
        x : list | tuple | np.ndarray,
        y : list | tuple | np.ndarray,
        z : list | tuple | np.ndarray,
        xangle : float | int | np.float_ | np.integer = 0,
        yangle : float | int | np.float_ | np.integer = 0,
        zangle : float | int | np.float_ | np.integer = 0,
):
    """
    Rotate the given ``x``\, ``y``\, and ``z`` points using an Euler rotation.

    An Euler rotation can be understood as follows. Imagine the x, y, and z axes
    as wooden dowels. First the z-axis dowel is rotated `zangle` degrees
    clockwise, then the y-axis dowel is rotated `yangle` degrees clockwise, and 
    finally the x-axis dowel is rotated `xangle` degrees clockwise.

    Parameters
    ----------
    x : list, tuple, :class:`numpy.ndarray`
        The x components.
    
    y : list, tuple, :class:`numpy.ndarray`
        The y components.

    z : list, tuple, :class:`numpy.ndarray`
        The z components.

    xangle : float, int, :class:`numpy.float_`\, :class:`numpy.integer`\, default = 0
        The x component of an Euler rotation in degrees.

    yangle : float, int, :class:`numpy.float_`\, :class:`numpy.integer`\, default = 0
        The y component of an Euler rotation in degrees.

    zangle : float, int, :class:`numpy.float_`\, :class:`numpy.integer`\, default = 0
        The z component of an Euler rotation in degrees.
    
    Returns
    -------
    x, y, z
        A copy of the x, y, z components which were given as inputs, rotated
        by ``xangle``\, ``yangle``\, and ``zangle``\.
    """
    import copy

    x = copy.deepcopy(x)
    y = copy.deepcopy(y)
    z = copy.deepcopy(z)
    
    xanglerad = float(xangle) / 180. * np.pi
    yanglerad = float(yangle) / 180. * np.pi
    zanglerad = float(zangle) / 180. * np.pi

    if zangle != 0: # Rotate about z
        rold = np.sqrt(x * x + y * y)
        phi = np.arctan2(y, x)
        phi -= zanglerad
        x = rold * np.cos(phi)
        y = rold * np.sin(phi)
    if yangle != 0: # Rotate about y
        rold = np.sqrt(z * z + x * x)
        phi = np.arctan2(z, x)
        phi -= yanglerad
        z = rold * np.sin(phi)
        x = rold * np.cos(phi)
    if xangle != 0: # Rotate about x
        rold = np.sqrt(y * y + z * z)
        phi = np.arctan2(z, y)
        phi -= xanglerad
        y = rold * np.cos(phi)
        z = rold * np.sin(phi)

    return x, y, z


def rotate_spherical(
        *args,
        theta : float | int | np.float_ | np.integer = 0,
        phi : float | int | np.float_ | np.integer = 0,
):
    """
    Uses :func:`~.rotate` twice to perform a correct spherical coordinates
    rotation.
    
    Parameters
    ----------
    *args
        Positional arguments are passed directly to :func:`~.rotate`\.

    theta : float, int, :class:`numpy.float_`\, :class:`numpy.integer`\, default = 0
        The polar angle in degrees.

    phi : float, int, :class:`numpy.float_`\, :class:`numpy.integer`\, default = 0
        The azimuthal angle in degrees.
    
    Returns
    -------
    args
        A rotated version of the given positional arguments.
    """
    
    if phi == 0:
        return rotate(
            *args,
            xangle = 0.,
            yangle = theta,
            zangle = 0.,
        )
    if theta == 0:
        return rotate(
            *args,
            xangle = 0.,
            yangle = 0.,
            zangle = -phi,
        )
    
    return rotate(
        *rotate(
            *args,
            xangle = 0.,
            yangle = theta,
            zangle = 0.,
        ),
        xangle = 0.,
        yangle = 0.,
        zangle = -phi,
    )





@starsmashertools.preferences.use
class EffGravPot(object):
    r"""
    A class for working with effective gravitational potentials in an
    :class:`~.lib.output.Output`\, which can come from either a 
    :class:`~.lib.binary.Binary` or from a :class:`~.lib.dynamical.Dynamical`
    prior to the plunge-in. In the presence of a binary in which the individual
    stars are represented as point masses, an individual fluid packet of mass 
    :math:`m` experiences a force :math:`F=-m\nabla\varphi`\, where

    .. math::

        \varphi(\vec{r}) = -\frac{GM_1}{r_1} - \frac{GM_2}{r_2} - \frac{1}{2}\omega^2r_\mathrm{com}^2

    in which :math:`G` is the gravitational constant, :math:`M_1` and 
    :math:`M_2` are the masses of stars 1 and 2, :math:`r_1=|\vec{r}-\vec{r}_1|`
    and :math:`r_2=|\vec{r}-\vec{r}_2|` are the distances of the fluid packet 
    from stars 1 and 2, :math:`\omega` is the oribtal frequency, and 
    :math:`r_\mathrm{com}=|\vec{r}-\vec{r}_\mathrm{com}|` is the distance of the
    fluid packet from the center of mass.
    
    To obtain :math:`\varphi(\vec{r})` without assuming the stars are point 
    masses, specify ``as_point_masses = False`` in  :meth:`~.__init__`\. Here,

    .. math::
    
        \varphi(\vec{r}) = -\sum_i^N\frac{Gm_i}{r_i} - \frac{1}{2}\omega^2r_\mathrm{com}^2

    where :math:`i` represents the SPH particle index and 
    :math:`r_i=|\vec{r}-\vec{r}_i|` is the distance between :math:`\vec{r}` and
    particle :math:`i`\. This method is significantly slower.
    """

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(
            self,
            output : starsmashertools.lib.output.Output,
            as_point_masses : bool = Pref('as_point_masses', True),
            **kwargs
    ):
        r"""
        Class constructor.

        Parameters
        ----------
        output : :class:`~.lib.output.Output`
            The output file to calculate the gravitational potentials for. If
            ``as_point_masses = True``\, the particles are separated based on
            the star they belong to using 
            :meth:`~.lib.binary.Binary.get_primary_IDs` and 
            :meth:`~.lib.binary.Binary.get_secondary_IDs`\. If the output
            belongs to a :class:`~.lib.dynamical.Dynamical` simulation, a search
            will be done to find its parent binary simulation.

            Particles which are detected as ejecta are not processed.

        Other Parameters
        ----------------
        as_point_masses : bool, default = ``Pref('as_point_masses', True)``
            If `True`\, the center of mass for the particles in the primary and
            secondary stars are each calculated and used as the point masses in
            the first equation for :math:`\varphi` above. Otherwise, the
            calculation proceeds using the second equation.

        """
        self._output = None
        self._as_point_masses = as_point_masses

        self._data = {
            'com1' : None,
            'com2' : None,
            'com' : None,
            'M1' : None,
            'M2' : None,
            'omega2' : None,
        }
        self.output = output # Updates the data for the first time

    @property
    def output(self): return self._output
    
    @output.setter
    def output(self, value):
        if self._output == value: return
        self._output = value
        self._update_data()

    @property
    def as_point_masses(self): return self._as_point_masses

    @as_point_masses.setter
    def as_point_masses(self, value : bool):
        if self._as_point_masses == value: return
        if not self._as_point_masses and value:
            self._update_data()
        self._as_point_masses = value

    def _update_data(self):
        import starsmashertools.lib.binary
        import starsmashertools.lib.dynamical
        import starsmashertools

        self._data['omega2'] = float((2*np.pi / self._output.simulation.get_period(
            self._output
        )))**2

        if isinstance(self._output.simulation, starsmashertools.lib.binary.Binary):
            self._data['com1'], self._data['com2'] = self._output.simulation.get_COMs(
                self._output,
            )
            self._data['M1'] = self._output.simulation.get_primary_mass()
            self._data['M2'] = self._output.simulation.get_secondary_mass()
            
        elif isinstance(self._output.simulation, starsmashertools.lib.dynamical.Dynamical):
            binary = self._output.simulation.get_binary()
            
            # Get the primary star COM
            with starsmashertools.mask(
                    self._output,
                    ~self._output['unbound'] & np.isin(
                        self._output['ID'], binary.get_primary_IDs(),
                        assume_unique = True,
                    ),
                    copy = False,
            ) as masked:
                self._data['com1'] = center_of_mass(
                    masked['am'], masked['x'], masked['y'], masked['z'],
                )
                self._data['M1'] = np.sum(masked['am'])
            # Get the secondary star COM
            with starsmashertools.mask(
                    self._output,
                    ~self._output['unbound'] & np.isin(
                        self._output['ID'], binary.get_secondary_IDs(),
                        assume_unique = True,
                    ),
                    copy = False,
            ) as masked:
                self._data['com2'] = center_of_mass(
                    masked['am'], masked['x'], masked['y'], masked['z'],
                )
                self._data['M2'] = np.sum(masked['am'])

            # The rotation frequency can be obtained as v/r where r is the
            # distance from the center of mass of a body with tangential
            # velocity v. We obtain the net tangential velocities of both bodies
            # and check to make sure they agree: they should be almost equal in
            # magnitude and pointing in opposite directions. Otherwise, there is
            # probably no orbit.
            
        else:
            raise TypeError("Effective gravitational potential plots are available only for simulations of type Binary or Dynamical, not '%s'" % type(self._output.simulation).__name__)

        # Get the overall COM
        with starsmashertools.mask(self._output, ~self._output['unbound']) as masked:
            self._data['com'] = center_of_mass(
                masked['am'], masked['x'], masked['y'], masked['z'],
            )

        # convert all to cgs
        for key in ['com1', 'com2', 'com']:
            self._data[key] *= float(self._output.simulation.units.length)
        for key in ['M1', 'M2']:
            self._data[key] *= float(self._output.simulation.units.mass)
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get(
            self,
            x : list | tuple | np.ndarray,
            y : list | tuple | np.ndarray,
            as_point_masses : bool | type(None) = None,
    ):
        r"""
        Parameters
        ----------
        x : list, tuple, :class:`numpy.ndarray`
            The ``x`` positions to get the effective gravitational potential at.
            Must have the same shape as ``y`` and ``z``\. Values must be in
            units of cm.
        
        y : list, tuple, :class:`numpy.ndarray`
            The ``y`` positions to get the effective gravitational potential at.
            Must have the same shape as ``x`` and ``z``\. Values must be in
            units of cm.
        
        z : float, list, tuple, :class:`numpy.ndarray`
            The ``z`` positions to get the effective gravitational potential at.
            Must have the same shape as ``x`` and ``y``\. Values must be in
            units of cm.

        Other Parameters
        ----------------
        as_point_masses : bool, None, default = None
            If `None`\, the value from :meth:`~.__init__` will be used.

        Returns
        -------
        :class:`numpy.ndarray`
            The effective gravitational potentials at positions ``(x, y, z)``\,
            in cgs units.
        """
        from starsmashertools.lib.units import constants
        import warnings
        
        if as_point_masses is None: as_point_masses = self.as_point_masses
        
        G = float(constants['G'])
        com1 = self._data['com1']
        com2 = self._data['com2']
        com = self._data['com']
        M1 = self._data['M1']
        M2 = self._data['M2']
        omega2 = self._data['omega2']
        
        xy = np.stack((x, y), axis = -1)
        result = np.zeros(x.shape)
        
        if not as_point_masses:
            with starsmashertools.mask(
                    self._output, ~self._output['unbound'], copy = False
            ) as masked:
                xyzi = np.column_stack((masked['x'], masked['y'], masked['z']))
                m = masked['am'] * float(self._output.simulation.units['am'])
            xyzi *= float(self._output.simulation.units.length)
            xyi = xyzi[:,:2]

            # Maybe it's fastest to loop over the particles...
            for i, (_xy, _m) in enumerate(zip(xyi, m)):
                result -= G*_m/np.sqrt(np.sum((xy - _xy)**2, axis = -1))
            rcom2 = np.sum((xy - com[:2])**2,axis = -1)
            result += -0.5*omega2*rcom2
            return result
        else:
            r1 = np.sqrt(np.sum((xy - com1[:2])**2, axis = -1))
            r2 = np.sqrt(np.sum((xy - com2[:2])**2, axis = -1))
            rcom2 = np.sum((xy - com[:2])**2, axis = -1)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return -G*M1/r1 - G*M2/r2 - 0.5*omega2*rcom2

    @api
    def get_gradient(self, *args, **kwargs):
        """
        Get the 3D gradient of the effective gravitational potential.

        Other Parameters
        ----------
        *args
            Positional arguments are passed directly to :meth:`~.get`\.
        
        **kwargs
            Keyword arguments are passed directly to :meth:`~.get`\.
        
        Returns
        -------
        :class:`numpy.ndarray`\, :class:`numpy.ndarray`\, :class:`numpy.ndarray`
            The gradient of the effective gravitational potential in the x, y,
            and z directions respectively.

        See Also
        --------
        :meth:`~.get`
        :meth:`numpy.gradient`
        """
        return np.gradient(self.get(*args, **kwargs))


    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_lagrange(
            self,
            n : int,
            resolution : int = 1000,
            return_potential : bool = False
    ):
        """
        Get the n'th Lagrange point

        Parameters
        ----------
        n : int
            The Lagrange point to obtain. Must be 1, 2, 3, 4, or 5.

        resolution : int, default = 1000
            How many positions to search in the parameter space.

        Other Parameters
        ----------------
        return_potential : bool, default = False
            If `True`\, also returns the effective gravitational potential
            corresponding with the Lagrange point.

        Returns
        -------
        :class:`numpy.ndarray`
            A NumPy array of shape (3,) with the x, y, and z positions of the 
            Lagrange point specified by ``n``\.

        float
            The effective gravitational potential at the Lagrange point if
            ``return_potential = True``
        """
        starsmashertools.helpers.argumentenforcer.enforcevalues({
            'n' : [1, 2, 3, 4, 5],
        })

        com1 = self._data['com1']
        com2 = self._data['com2']
        diff = com2 - com1 # destination - origin
        distance = np.sqrt(np.dot(diff, diff))
        direction = diff / distance
        
        if n == 1: # L1 is always located between M1 and M2
            origin = com1 #- 4*distance * direction
            span = np.linspace(0, distance, resolution)
            direc = direction
            #print(positions)
        elif n == 2: # L2 is always located behind M2
            # Take a guess that we should search twice the distance behind M2
            origin = com2
            span = np.linspace(0, 2*distance, resolution)
            direc = direction
        elif n == 3: # L3 is always located behind M1
            # Take a guess to search twice the distance behind M1
            origin = com1
            span = np.linspace(0, 2*distance, resolution)
            direc = -direction
        else: raise NotImplementedError # Not sure yet.

        positions = origin + span[:,None] * direc
        
        pot = self.get(positions[:,0], positions[:,1])

        mpot = np.abs(pot)
        potmin = np.nanmin(mpot[np.isfinite(mpot)])
        potmax = np.nanmax(mpot[np.isfinite(mpot)])
        mpot -= potmin
        mpot /= potmax - potmin
        
        idx = np.isfinite(pot)
        pot = pot[idx]
        positions = positions[idx]

        idx = np.nanargmin(mpot)
        
        if return_potential: return positions[idx], pot[idx]
        return positions[idx]

    if has_matplotlib:
        @starsmashertools.helpers.argumentenforcer.enforcetypes
        @api
        def get_equipotentials(
                self,
                potentials : list | tuple | np.ndarray,
                extent : list | tuple | np.ndarray | type(None) = None,
                normalize : bool = True,
                resolution : list | tuple | type(None) = None,
        ):
            """
            Find the positions of the equipotential contours corresponding with 
            the given effective gravitational potentials.

            Note that this requires Matplotlib.

            Parameters
            ----------
            potentials : list, tuple, :class:`numpy.ndarray`
                The effective gravitational potentials for which to find the 
                contours of effective gravitational equipotentials.
            
            Other Parameters
            ----------------
            extent : list, tuple, :class:`numpy.ndarray`, None, default = None
                Array of ``[xmin, xmax, ymin, ymax]`` for plotting purposes. If
                `None`\, defaults to the current ``axes`` limits.
            
            Returns
            -------
            list
                A list of :class:`numpy.ndarray` objects with the same length as
                the given ``potentials``\. Each element is an (N,3) array of 
                positions.
            """
            import starsmashertools.mpl.axes
            import matplotlib.pyplot as plt

            fig = plt.figure(
                figsize = (8, 8),
            )
            ax = fig.add_axes([0, 0, 1, 1])

            potentials = np.asarray(potentials)

            if extent is None:
                extent = list(ax.get_xlim()) + list(ax.get_ylim())

            if resolution is None:
                res = starsmashertools.mpl.axes.get_resolution(ax)
            else: res = resolution
            
            x =  np.linspace(extent[0], extent[1], res[0])
            y =  np.linspace(extent[2], extent[3], res[1])
            x, y = np.meshgrid(x, y)

            pot = self.get(x, y)

            # Normalizing to enable this to work
            if normalize:
                pot = np.abs(pot)
                idx = np.isfinite(pot)
                potmin = np.nanmin(pot[idx])
                potmax = np.nanmax(pot[idx])
                pot -= potmin
                pot /= potmax - potmin
                potentials = np.abs(potentials)
                potentials -= potmin
                potentials /= potmax - potmin
            
            order = np.argsort(potentials)
            
            cs = ax.contour(
                pot,
                potentials[order],
                origin = 'lower',
                extent = extent,
                cmap = 'tab10',
            )

            plt.close(fig)
            
            ret = []
            for s in order:
                segments = cs.allsegs[s]
                xy = []
                for j, s in enumerate(segments):
                    xy += s.tolist()
                    xy += [[np.nan, np.nan]]
                    
                ret += [np.asarray(xy)]
            return ret
