# Included here are useful common functions
import numpy as np
import starsmashertools.lib.units
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
    G = float(starsmashertools.lib.units.gravconst)
    return np.sqrt(4 * np.pi**2 / (G * (m1 + m2)) * separation**3)


# The Roche lobe radius
@api
def rocheradius(m1,m2,separation):
    q = m1 / m2
    qonethird = q**(1./3.)
    qtwothirds = qonethird * qonethird
    return 0.49*qtwothirds / (0.6*qtwothirds + np.log(1. + qonethird)) * separation

# Given data 'x' and 'y', linearly interpolate the position(s) 'x0'.
# Extrapolates outside the 'x' and 'y' data if 'extrapolate' is True.
# Otherwise, raises an error.
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
        _x0 = x[left]
        _x1 = x[right]
        _y0 = y[left]
        _y1 = y[right]
        ret[i] = (_y0 * (_x1 - _x) + _y1 * (_x - _x0)) / (_x1 - _x0)
    return ret
