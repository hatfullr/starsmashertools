import starsmashertools.helpers.argumentenforcer
import numpy as np

class OutOfBoundsError(ValueError, object):
    def __init__(self, interpolator, x, y):
        offenders = np.where(~interpolator.is_in_bounds(x, y))[0]
        message = """
   xmin, xmax = {xmin:f}, {xmax:f}
   ymin, ymax = {ymin:f}, {ymax:f}
   x = {x:s}
   y = {y:s}
   offenders:
      {offenders:s}
        """.format(
            xmin = interpolator.xmin,
            xmax = interpolator.xmax,
            ymin = interpolator.ymin,
            ymax = interpolator.ymax,
            x = str(x),
            y = str(y),
            offenders = '{:>5s} {:>15s}  {:>15s}'.format(
                'index', 'x', 'y'
            ) + '\n' + '\n'.join(['      {i:>5d} {x:>15.7E}, {y:>15.7E}'.format(
                i = i,
                x = float(x[offenders[i]]),
                y = float(y[offenders[i]]),
            ) for i in range(len(offenders))]),
        )
        super(OutOfBoundsError, self).__init__(message.rstrip())

class Interpolator(object):
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def __init__(
            self,
            array, #: float | np.ndarray,
            *axes,
    ):
        axes = list(axes)
        for i, axis in enumerate(axes):
            axes[i] = np.asarray(axis)
        
        # Check errors
        if len(axes) != len(array.shape):
            raise Exception("You must supply a number of axes equal to the dimensions of 'array', whose shape is %s. Received %d axes." % (str(array.shape), len(axes)))
        
        for l, axis in zip(array.shape, axes):
            if len(axis.shape) != 1:
                raise Exception("Input axes must be one-dimensional, but received an array with dimensions %s" % str(axis.shape))
            if len(axis) != l:
                raise Exception("Input axis has length %d but needs length %d" % (len(axis), l))

        # Initialize
        self.array = array
        self.axes = axes
    
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def is_in_bounds(self, *args, **kwargs):
        r"""
        Return a boolean mask that indicates which of the given array elements
        are considered within the bounds of this interpolator.

        This method must be overriden by subclasses.
        """
        raise NotImplementedError
    

class BilinearInterpolator(Interpolator, object):
    def __init__(self, z, x, y):
        super(BilinearInterpolator, self).__init__(z, x, y)
        self.x = self.axes[0]
        self.y = self.axes[1]
        self.z = self.array
        self.xmin = np.nanmin(self.x)
        self.xmax = np.nanmax(self.x)
        self.ymin = np.nanmin(self.y)
        self.ymax = np.nanmax(self.y)
    
    def clamp(self, value, _min, _max):
        value = np.asarray(value)
        if not np.all(np.isfinite(value)):
            raise ValueError("Cannot clamp because not all given values are finite")
        mins = np.full(value.shape, _min, dtype=int)
        maxs = np.full(value.shape, _max, dtype=int)
        valid = np.isfinite(value)
        return np.maximum(np.minimum(value[valid], maxs), mins)

    def get_bounds(self, x, y):
        if not np.all(self.is_in_bounds(x, y)):
            raise OutOfBoundsError(self, x, y)

        x = np.asarray(x)
        y = np.asarray(y)

        iarr = np.zeros(np.prod(x.shape), dtype=int)
        jarr = np.zeros(np.prod(y.shape), dtype=int)
        for i, _x in enumerate(x.flatten()):
            iarr[i] = np.nanargmin(np.abs(self.x - _x))
        for j, _y in enumerate(y.flatten()):
            jarr[j] = np.nanargmin(np.abs(self.y - _y))

        i = iarr.reshape(x.shape)
        j = jarr.reshape(y.shape)
        
        i0 = self.clamp(i, 0, len(self.x) - 2)
        j0 = self.clamp(j, 0, len(self.y) - 2)
        
        return i0, i0 + 1, j0, j0 + 1
        
    def __call__(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        i0, i1, j0, j1 = self.get_bounds(x, y)
        
        x0, y0 = self.x[i0], self.y[j0]
        x1, y1 = self.x[i1], self.y[j1]
        
        Q00 = self.z[i0, j0]
        Q01 = self.z[i0, j1]
        Q10 = self.z[i1, j0]
        Q11 = self.z[i1, j1]

        numerator = Q00*(x1-x)*(y1-y) + Q10*(x-x0)*(y1-y) + Q01*(x1-x)*(y-y0) + Q11*(x-x0)*(y-y0)
        denominator = (x1-x0)*(y1-y0)
        return numerator / denominator

    def is_in_bounds(self, x, y):
        r"""
        Get a boolean mask array that indicates which of the elements in ``x``
        and ``y`` are within the interpolator bounds.
        
        Parameters
        ----------
        x
            The x coordinates of the points.
        y
            The y coordinates of the points.
        
        Returns
        -------
        mask : :class:`numpy.ndarray`
            A boolean mask where `True` indicates that the point is within the
            interpolation bounds and `False` indicates that the point is outside
            the interpolation bounds.
        """
        x = np.asarray(x)
        y = np.asarray(y)
        eps = np.finfo(float).eps
        return (   (self.xmin - eps <= x) & (x <= self.xmax + eps) &
                   (self.ymin - eps <= y) & (y <= self.ymax + eps)   )
