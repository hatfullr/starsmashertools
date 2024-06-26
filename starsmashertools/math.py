# Included here are useful common functions
import numpy as np
from starsmashertools.helpers.apidecorator import api

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


