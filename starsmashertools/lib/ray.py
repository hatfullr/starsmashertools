import numpy as np
from starsmashertools.helpers.apidecorator import api
import starsmashertools.helpers.argumentenforcer

class Ray(object):
    r"""
    A ray consists of a 3D position and a 3D direction.
    """
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(
            self,
            position : list | tuple | np.ndarray,
            direction : list | tuple | np.ndarray,
    ):
        r"""
        Parameters
        ----------
        position : list, tuple, :class:`numpy.ndarray`
            The 3D position vector of the ray, (x, y, z).

        direction : list, tuple, :class:`numpy.ndarray`
            The 3D direction vector of the ray, 
            :math:`(\\hat{x},\\hat{y},\\hat{z})`\. The direction can be obtained
            as a subtraction of two vectors :math:`\\vec{A}-\\vec{B}`\, where 
            :math:`\\vec{A}` is the "destination" and :math:`\\vec{B}` is the 
            "origin". This array is normalized automatically.
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
        r"""
        Cast this ray through a given set of hard spheres, each with positions
        ``x``\, ``y``\, and ``z`` and radii ``r``\.

        Parameters
        ----------
        *args
            Positional arguments are passed directly to 
            :meth:`~.RayCast.__init__`\.

        **kwargs
            Keyword arguments are passed directly to 
            :meth:`~.RayCast.__init__`\.

        Returns
        -------
        cast : :class:`~.RayCast`
            The :class:`~.RayCast` object corresponding to this cast. The 
            intersection points can be accessed with :attr:`~.RayCast.points`
            and the indices with :attr:`~.RayCast.indices`\.
        """
        return RayCast(self, *args, **kwargs)
        
        



class RayCast(object):
    r"""
    An object for holding information about a ray cast operation, where the cast
    is done through a given collection of hard spheres.
    """
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(
            self,
            ray : Ray,
            *args,
    ):
        if len(args) == 2:
            xyz = np.asarray(args[0])
        elif len(args) == 4:
            xyz = np.column_stack(args[:-1])
        else:
            raise Exception("Arguments must be either 'x, y, z, r' or 'xyz, r'")
        r = np.asarray(args[-1])
        
        self.ray = ray
        self.points = np.asarray([])
        self.indices = np.asarray([])
        self.distances = np.asarray([])
        valid = np.isfinite(xyz).all(axis = 1) & np.isfinite(r)
        if valid.any():
            indices = np.where(valid)[0]
            
            # destination - origin
            D = xyz[indices] - self.ray.position
            # https://stackoverflow.com/a/19864047
            D_mag2 = np.einsum('...j,...j->...', D, D) # Fastest I could find
            ahat = self.ray.direction
            s_mag = np.dot(D, ahat) # Also equals D_mag * cos(theta)
            
            h_mag2 = D_mag2 - s_mag**2
            
            r2 = r[indices]**2
            idx = h_mag2 < r2
            if idx.any():
                idx = np.where(idx)[0]
                indices = indices[idx]
                n = len(indices)
                s_mag = s_mag[idx].reshape(n, 1)
                b_mag2 = (r2[idx] - h_mag2[idx]).reshape(n, 1)
                b_mag = np.sqrt(b_mag2)
                
                ahat_arr = np.tile(ahat, n).reshape(n, 3)
                p1 = self.ray.position + (s_mag - b_mag) * ahat_arr
                p2 = self.ray.position + (s_mag + b_mag) * ahat_arr
                
                self.points = np.vstack((p1, p2))
                self.indices = np.tile(indices, 2)

                diff = self.points - self.ray.position # destination - origin
                self.distances = np.sqrt(np.einsum('...j,...j->...', diff, diff))
                
                sorted_indices = np.argsort(self.distances)
                
                self.points = self.points[sorted_indices]
                self.indices = self.indices[sorted_indices]
                self.distances = self.distances[sorted_indices]
    

    def __bool__(self):
        r""" Returns `True` if the :attr:`points` array has values. Otherwise, 
        returns `False`\. """
        return len(self.points) > 0
