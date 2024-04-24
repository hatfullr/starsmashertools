import numpy as np
from starsmashertools.helpers.apidecorator import api
import starsmashertools.helpers.argumentenforcer

class Ray(object):
    """
    A ray consists of a 3D position and a 3D direction.
    """
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(
            self,
            position : list | tuple | np.ndarray,
            direction : list | tuple | np.ndarray,
    ):
        """
        Parameters
        ----------
        position : list, tuple, np.ndarray
            The 3D position vector of the ray, (x, y, z).

        direction : list, tuple, np.ndarray
            The 3D direction vector of the ray, (xhat, yhat, zhat). The 
            direction can be obtained as a subtraction of two vectors A-B, where
            A is the "destination" and B is the "origin". This array is 
            normalized automatically.
        """

        position = np.asarray(position, dtype = float)
        direction = np.asarray(direction, dtype = float)

        if position.shape != (3,):
            raise ValueError("Argument 'position' must have shape (3,), not %s" % str(position.shape))
        if direction.shape != (3,):
            raise ValueError("Argument 'direction' must have shape (3,), not %s" % str(direction.shape))

        # Normalize the direction
        magnitude = np.sqrt(np.sum(direction**2))
        if magnitude == 0:
            raise ValueError("A direction cannot have zero magnitude.")
        direction /= magnitude

        self.position = position
        self.direction = direction

    @api
    def cast(self, *args, **kwargs):
        """
        Cast this ray through a given set of hard spheres, each with positions
        ``x``, ``y``, and ``z`` and radii ``r``.

        Parameters
        ----------
        *args
            Positional arguments are passed directly to 
            :meth:`~.RayCast.__init__`.

        **kwargs
            Keyword arguments are passed directly to :meth:`~.RayCast.__init__`.

        Returns
        -------
        cast : :class:`~.RayCast`
            The RayCast object corresponding to this cast. The intersection 
            points can be accessed with ``cast.points`` and the indices with 
            ``cast.indices``.
        """
        return RayCast(self, *args, **kwargs)
        
        



class RayCast(object):
    """
    An object for holding information about a ray cast operation, where the cast
    is done through a given collection of hard spheres.
    """
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(
            self,
            ray : Ray,
            x : list | tuple | np.ndarray,
            y : list | tuple | np.ndarray,
            z : list | tuple | np.ndarray,
            r : list | tuple | np.ndarray,
    ):
        self.ray = ray
        self.points = np.asarray([])
        self.indices = np.asarray([])
        
        xyz = np.column_stack((x,y,z))
        r = np.asarray(r)

        valid = np.logical_and(
            np.logical_and(np.isfinite(x), np.isfinite(y)),
            np.logical_and(np.isfinite(z), np.isfinite(r)),
        )
        if valid.any():
            indices = np.arange(len(xyz), dtype = int)
            xyz = xyz[valid]
            r = r[valid]
            indices = indices[valid]
            
            D = xyz - self.ray.position # destination - origin
            D_mag2 = np.sum(D**2, axis = 1)
            ahat = self.ray.direction
            s_mag = np.dot(D, ahat)
            h_mag2 = D_mag2 - s_mag**2
            
            r2 = r**2
            idx = h_mag2 < r2
            if idx.any():
                n = sum(idx)
                s_mag = s_mag[idx].reshape(n, 1)
                b_mag2 = r2[idx].reshape(n, 1) - h_mag2[idx].reshape(n, 1)
                b_mag = np.sqrt(b_mag2)
                
                ahat_arr = np.tile(ahat, n).reshape(n, 3)
                p1 = self.ray.position + (s_mag - b_mag) * ahat_arr
                p2 = self.ray.position + (s_mag + b_mag) * ahat_arr
                
                self.points = np.vstack((p1, p2))
                self.indices = np.tile(indices[idx], 2)
                
                sorted_indices = np.argsort(np.sum(self.points**2, axis = 1))
                self.points = self.points[sorted_indices]
                self.indices = self.indices[sorted_indices]
    
