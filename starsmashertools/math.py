# Included here are useful common functions
import starsmashertools.preferences
from starsmashertools.preferences import Pref
import numpy as np
from starsmashertools.helpers.apidecorator import api
import starsmashertools.helpers.argumentenforcer
import copy
import warnings

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
    r"""
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
    r"""
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

    class MaxItersExceededError(Exception): pass
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(
            self,
            m : list | tuple | np.ndarray,
            x : list | tuple | np.ndarray,
            y : list | tuple | np.ndarray,
            z : list | tuple | np.ndarray,
            period : int | float = 0.,
            G : int | float | type(None) = None,
    ):
        r"""
        Class constructor.

        Parameters
        ----------
        m : list, tuple, :class:`numpy.ndarray`
            Masses of the particles. Must be 1D.

        x : list, tuple, :class:`numpy.ndarray`
            x-positions of the particles. Must be 1D and have the same length as
            ``m``\.

        y : list, tuple, :class:`numpy.ndarray`
            y-positions of the particles. Must be 1D and have the same length as
            ``m``\.

        z : list, tuple, :class:`numpy.ndarray`
            z-positions of the particles. Must be 1D and have the same length as
            ``m``\.
        
        Other Parameters
        ----------------
        period : int, float, default = 0.
            The oribtal period of the binary system.

        G : int, float, :class:`~.lib.units.Unit`\, None, default = None
            The value of the gravitational constant. If `None`\, the value 
            stored in the constants in :mod:`~.starsmashertools.lib.units` is 
            used. If a :class:`~.lib.units.Unit` is given, it is converted to a
            :py:class:`float`\.
        """

        if G is None:
            from starsmashertools.lib.units import constants
            G = constants['G']
        self.G = float(G)

        self.m = np.asarray(m)
        self.xyz = np.column_stack((x,y,z))
        self.period = period

        self._com = center_of_mass(self.m, x, y, z)
        
        self._omega = 2 * np.pi / self.period

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get(
            self,
            positions : list | tuple | np.ndarray,
    ):
        r"""
        Sample the effective gravitational potential :math:`\varphi` field at 
        the given positions.

        Parameters
        ----------
        positions : list, tuple, :class:`numpy.ndarray`
            The x, y, and z positions to get the effective gravitational 
            potential at. Must have a shape (n,3).

        Returns
        -------
        :class:`numpy.ndarray`
            The effective gravitational potential at positions ``(x, y, z)``\. 
            The result is an array of magnitudes with the same length as ``x``\,
            ``y``\, and ``z``\.
        """
        if not isinstance(positions[0], (list, tuple, np.ndarray)):
            positions = np.asarray([positions])
        result = np.full(len(positions), np.nan)
        rcom2s = np.sum((positions - self._com)**2, axis = -1)
        
        for i, pos in enumerate(positions):
            r = np.sqrt(np.sum((pos - self.xyz)**2, axis = -1))
            result[i] = -np.sum(self.m / r)
        return result * self.G - 0.5*self._omega**2*rcom2s

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_gradient(
            self,
            positions : list | tuple | np.ndarray,
    ):
        r"""
        Sample the effective gravitational potential gradient 
        :math:`\nabla\varphi` field at the given positions. This is the same as
        negative the gravitational acceleration vector, since 
        :math:`m\vec{g}=\vec{F}=-m\nabla\varphi`\.
        
        Parameters
        ----------
        positions : list, tuple, :class:`numpy.ndarray`
            The x, y, and z positions to get the effective gravitational 
            potential gradient at. Must have a shape (n,3).

        Returns
        -------
        :class:`numpy.ndarray`
            The gradient of the effective gravitational potential at positions
            ``(x, y, z)``\. The result is an array of 3-vectors, with the same
            length as ``x``\, ``y``\, and ``z``\.
        """
        if not isinstance(positions[0], (list, tuple, np.ndarray)):
            positions = np.asarray([positions])
        
        result = np.full((len(positions), 3), np.nan)
        for i, pos in enumerate(positions):
            dxyz = pos - self.xyz # destination - origin
            
            # distance magnitude squared (row sum)
            r2 = np.sum(dxyz**2, axis = 1)
            r = np.sqrt(r2)

            # divide each x, y, and z component of the offsets by the magnitude
            # of that offset
            directions = dxyz / r[:,None]
            
            # \vec{g} (column sum)
            g = -np.sum(directions * (self.G * self.m / r2)[:, None], axis = 0)
            
            # centripetal acceleration here
            a = -self._omega**2 * (pos - self._com) # destination - origin
            
            result[i] = -(g + a)
        return result
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_lagrange(
            self,
            which : list | tuple | np.ndarray = [1,2,3],
            extent_frac : int | float = 5.,
    ):
        r"""
        Get the n'th Lagrange points.

        Parameters
        ----------
        which : list, tuple, :class:`numpy.ndarray`, default = ``[1,2,3]``
            The Lagrange points to obtain. Currently only :math:`L_1`\, 
            :math:`L_2`\, and :math:`L_3` are supported. 
        
        Other Parameters
        ----------------
        extent_frac : int, float, default = 5.
            The fraction of the maximum extent to search. The extent is the 
            distance of the farthest-away particle. We search beyond that 
            particle by this amount to sample :math:`\varphi`\.

        Returns
        -------
        :class:`numpy.ndarray`
            A NumPy array of shape (n,3) with the x, y, and z positions of the 
            Lagrange points where n is the length of ``which``\.
        """
        import warnings
        
        # Start at the COM, and find the gradient. If it truly is a binary
        # system, then the COM is always somewhere between the two stars. We
        # want to check along the line connecting the two stars. To find that
        # line, we first move radially outward from the COM until we locate two
        # local minima in \phi and a local maximum in \phi
        gradcom = self.get_gradient(self._com)[0]

        ret = np.full((len(which), 3), np.nan)

        if any([i in which for i in range(1, 3 + 1)]):
            # Obtain a bunch of sample points up until the centripetal
            # acceleration is greater than the gravitational acceleration
            r2s = np.sum((self.xyz - self._com)**2, axis=1)

            # The location of the closest star is in the direction of the
            # specific net force, |\vec{F}/m| = |\nabla\varphi|.
            warnings.filterwarnings(action='ignore')
            direction = gradcom / np.sqrt(np.sum(gradcom**2))
            warnings.resetwarnings()

            max_extent = np.sqrt(np.amax(r2s))
            # search around the area
            samples = np.linspace(-max_extent, max_extent, 1000)[:,None] * extent_frac * direction
            phis = self.get(samples)

            # For debugging:
            #import matplotlib.pyplot as plt
            #fig, ax = plt.subplots()
            #ax.plot(samples[:,0], phis)
            #plt.show()
            #quit()

            # Split the searched area into 3 segments, containing the 3 Lagrange
            # points. We mark the separated segments by assigning NaN to some
            # values, and then we split the region on NaNs.

            # Move along the samples to find the places where phi was
            # increasing, but is now starting to decrease.
            boundaries = []
            decreasing = False
            for i, phi in enumerate(phis[:-1]):
                if phis[i+1] <= phi:
                    if decreasing: continue
                    else: boundaries += [i]
                    decreasing = True
                elif phis[i+1] > phi:
                    decreasing = False

            if len(boundaries) != 3:
                raise Exception("Failed to split the region into 3 distinct groups: number of boundaries found != 3")
            
            idx1 = boundaries[0] + np.argmin(phis[boundaries[0]:boundaries[1]])
            idx2 = boundaries[1] + np.argmin(phis[boundaries[1]:boundaries[2]])
            phis[idx1] = np.nan
            samples[idx1] = np.full(3, np.nan)
            phis[idx2] = np.nan
            samples[idx2] = np.full(3, np.nan)
            
            # https://stackoverflow.com/a/14606271
            points = []
            for i in np.ma.clump_unmasked(np.ma.masked_invalid(phis)):
                s = samples[i]
                p = phis[i]

                points += [s[np.argmax(p)]]

            points = np.asarray(points)

            # For debugging:
            #import matplotlib.pyplot as plt
            #fig, ax = plt.subplots()
            #for x,y in zip(points, phis):
            #    ax.scatter(x[0], y)
            #plt.show()
            #quit()
            
            # Thus, the "line" connecting the stars must be parallel to the
            # direction of the gradient. The first 3 Lagrange points are located
            # on that line.

            # Search in the direction first, then negative the direction next to
            # obtain all the Lagrange points. We distinguish between them after.
            # (looking for the 3 maxima on the line)
            #
            # In one direction, phi will first decrease until passing through
            # the COM of one of the stars, where phi = -inf. After, phi will
            # increase to a maximum and then decrease again forever.
            #
            # In the other direction, phi will first increase until a maximum,
            # then decrease until passing through the COM of the other star,
            # where phi = inf. Then phi will increase to another maximum, and
            # thereafter phi decreases again forever.
            
            if len(points) != 3:
                raise Exception("Located a number of Lagrange points != 3:\n" + str(points))

            center = np.mean(points, axis = 0)
            extrema = []
            not_extrema = []
            for i, point in enumerate(points):
                to_center = center - point
                warnings.filterwarnings(action ='ignore')
                to_center /= np.sqrt(np.sum(to_center**2)) # normalize
                warnings.resetwarnings()
                for other in points:
                    if point is other: continue
                    diff = other - point # destination - origin
                    warnings.filterwarnings(action = 'ignore')
                    direc = diff / np.sqrt(np.sum(diff**2))
                    warnings.resetwarnings()
                    # ahat dot bhat = cos(theta)
                    # if cos(theta) == 1, it's the same direction
                    # if cos(theta) == -1, it's the opposite direction
                    if np.dot(direc, to_center) <= 0:
                        not_extrema += [i]
                        break
                else: extrema += [i]

            if len(not_extrema) != 1:
                raise Exception("Expected one non-extrema point")

            # L1 is between L2 and L3
            ret[0] = points[not_extrema[0]]
            
            # At the extrema, phi is larger behind the larger mass (L3)
            idx = np.argsort(self.get(points[extrema]))
            ret[1] = points[extrema[idx[0]]]
            ret[2] = points[extrema[idx[1]]]

        # TODO: find Lagrange points 4 and 5
                
        return ret
