import starsmashertools.helpers.argumentenforcer
import starsmashertools.lib.units
import numpy as np


numeric_types = starsmashertools.lib.units.Unit | int | np.integer | float | np.float_
array_types = [list, tuple, np.ndarray]
    
class NonFiniteValueError(ValueError, object):
    def __init__(self, name, value, *args, **kwargs):
        message = "'%s' must be finite but it is '%s'" % (name, str(value))
        super(NonFiniteValueError, self).__init__(message, *args, **kwargs)
class NegativeValueError(ValueError, object):
    def __init__(self, name, value, *args, **kwargs):
        message = "'%s' must be positive but it is '%s'" % (name, value)
        super(NegativeValueError, self).__init__(message, *args, **kwargs)

class Extents(object):
    """
    Contains information of a bounding box volume.
    """
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def __init__(
            self,
            xmin : numeric_types = -float('inf'),
            xmax : numeric_types = float('inf'),
            ymin : numeric_types = -float('inf'),
            ymax : numeric_types = float('inf'),
            zmin : numeric_types = -float('inf'),
            zmax : numeric_types = float('inf'),
    ):
        self._xmin = xmin
        self._xmax = xmax
        self._ymin = ymin
        self._ymax = ymax
        self._zmin = zmin
        self._zmax = zmax

    def __str__(self): return self.__repr__()

    def __repr__(self):
        return "Extents(x:[%g, %g], y:[%g, %g], z:[%g, %g])" % (self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax)

    @property
    def xmin(self):
        """ Get the minimum bound on the x-axis. """
        return self._xmin
    @property
    def xmax(self):
        """ Get the maximum bound on the x-axis. """
        return self._xmax
    @property
    def ymin(self):
        """ Get the minimum bound on the y-axis. """
        return self._ymin
    @property
    def ymax(self):
        """ Get the maximum bound on the y-axis. """
        return self._ymax
    @property
    def zmin(self):
        """ Get the minimum bound on the z-axis. """
        return self._zmin
    @property
    def zmax(self):
        """ Get the maximum bound on the z-axis. """
        return self._zmax
        
    @property
    def width(self):
        """ Get the size on the x-axis. """
        return self.xmax - self.xmin
    
    @property
    def breadth(self):
        """ Get the size on the y-axis. """
        return self.ymax - self.ymin
    
    @property
    def height(self):
        """ Get the size on the z-axis. """
        return self.zmax - self.zmin

    @property
    def volume(self):
        """ Get the volume of the bounding box. If any limit is infinite then
        the volume will be infinite. """
        return self.width * self.breadth * self.height

    @property
    def min(self):
        """ Get the point (xmin, ymin, zmin) as a NumPy array. """
        return np.array([self.xmin, self.ymin, self.zmin])
    
    @property
    def max(self):
        """ Get the point (xmax, ymax, zmax) as a NumPy array. """
        return np.array([self.xmax, self.ymax, self.zmax])

    @property
    def xy(self):
        """ Return the xy bounds as [xmin, xmax, ymin, ymax]. """
        return np.array([self.xmin, self.xmax, self.ymin, self.ymax])

    @property
    def yx(self):
        """ Return the yx bounds as [ymin, ymax, xmin, xmax]. """
        return np.array([self.ymin, self.ymax, self.xmin, self.xmax])

    @property
    def xz(self):
        """ Return the xz bounds as [xmin, xmax, zmin, zmax]. """
        return np.array([self.xmin, self.xmax, self.zmin, self.zmax])

    @property
    def zx(self):
        """ Return the zx bounds as [zmin, zmax, xmin, xmax]. """
        return np.array([self.zmin, self.zmax, self.xmin, self.xmax])

    @property
    def yz(self):
        """ Return the yz bounds as [ymin, ymax, zmin, zmax]. """
        return np.array([self.ymin, self.ymax, self.zmin, self.zmax])

    @property
    def zy(self):
        """ Return the zy bounds as [zmin, zmax, ymin, ymax]. """
        return np.array([self.zmin, self.zmax, self.ymin, self.ymax])
    
    @property
    def center(self):
        """ Get the center position of the bounding volume. """
        return 0.5 * (self.min + self.max)

    @property
    def radius(self):
        """ Get the maximum distance from the center to the edge of the bounding
        volume. """
        
        xcenter, ycenter, zcenter = self.center
        return max(
            xcenter - self.xmin,
            self.xmax - xcenter,
            ycenter - self.ymin,
            self.ymax - ycenter,
            zcenter - self.zmin,
            self.zmax - zcenter,
        )


    

    
    @xmin.setter
    def xmin(self, value):
        """ Set the minimum bound on the x-axis. """
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'value' : [starsmashertools.lib.units.Unit, int, np.integer, float, np.float_],
        })
        self._xmin = value
        self._check_bad_values()
    @xmax.setter
    def xmax(self, value):
        """ Set the maximum bound on the x-axis. """
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'value' : [starsmashertools.lib.units.Unit, int, np.integer, float, np.float_],
        })
        self._xmax = value
        self._check_bad_values()

    @ymin.setter
    def ymin(self, value):
        """ Set the minimum bound on the y-axis. """
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'value' : [starsmashertools.lib.units.Unit, int, np.integer, float, np.float_],
        })
        self._ymin = value
        self._check_bad_values()
    @ymax.setter
    def ymax(self, value):
        """ Set the maximum bound on the y-axis. """
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'value' : [starsmashertools.lib.units.Unit, int, np.integer, float, np.float_],
        })
        self._ymax = value
        self._check_bad_values()

    @zmin.setter
    def zmin(self, value):
        """ Set the minimum bound on the z-axis. """
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'value' : [starsmashertools.lib.units.Unit, int, np.integer, float, np.float_],
        })
        self._zmin = value
        self._check_bad_values()
    @zmax.setter
    def zmax(self, value):
        """ Set the maximum bound on the z-axis. """
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'value' : [starsmashertools.lib.units.Unit, int, np.integer, float, np.float_],
        })
        self._zmax = value
        self._check_bad_values()
    
        
    @width.setter
    def width(self, value):
        """ Set the size on the x-axis. The limits are expanded/shrunken around
        the center point. """
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'value' : [starsmashertools.lib.units.Unit, int, np.integer, float, np.float_],
        })
        if value < 0:
            raise NegativeValueError('width', value)
        if not np.isfinite(self.xmin):
            raise NonFiniteValueError('xmin', self.xmin)
        if not np.isfinite(self.xmax):
            raise NonFiniteValueError('xmax', self.xmax)
        difference = value - self.width
        self._xmin -= 0.5*difference
        self._xmax += 0.5*difference
        self._check_bad_values()

    @breadth.setter
    def breadth(self, value):
        """ Set the size on the y-axis. The limits are expanded/shrunken around
        the center point. """
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'value' : [starsmashertools.lib.units.Unit, int, np.integer, float, np.float_],
        })
        if value < 0:
            raise NegativeValueError('breadth', value)
        if not np.isfinite(self.ymin):
            raise NonFiniteValueError('ymin', self.ymin)
        if not np.isfinite(self.ymax):
            raise NonFiniteValueError('ymax', self.ymax)
        difference = value - self.breadth
        self._ymin -= 0.5*difference
        self._ymax += 0.5*difference
        self._check_bad_values()

    @height.setter
    def height(self, value):
        """ Set the size on the z-axis. The limits are expanded/shrunken around
        the center point. """
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'value' : [starsmashertools.lib.units.Unit, int, np.integer, float, np.float_],
        })
        if value < 0:
            raise NegativeValueError('height', value)
        if not np.isfinite(self.zmin):
            raise NonFiniteValueError('zmin', self.zmin)
        if not np.isfinite(self.zmax):
            raise NonFiniteValueError('zmax', self.zmax)
        difference = value - self.height
        self._zmin -= 0.5*difference
        self._zmax += 0.5*difference
        self._check_bad_values()

    @volume.setter
    def volume(self, value):
        """ Set the volume of the bounding box by expanding/shrinking the volume
        proportionally in all dimensions around the center. """

        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'value' : [starsmashertools.lib.units.Unit, int, np.integer, float, np.float_],
        })
        if value < 0:
            raise NegativeValueError('volume', value)
        scale_factor = (value / self.volume)**(1./3.)
        self.width *= scale_factor
        self.breadth *= scale_factor
        self.height *= scale_factor
        
    @min.setter
    def min(self, value):
        """ Set the minimum point (xmin, ymin, zmin). """
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'value' : array_types,
        })
        value = np.asarray(value, dtype = self.min.dtype)
        if value.shape != (3,):
            raise ValueError("You must give a list, tuple, or np.ndarray of shape (3,) to set the 'min' property, not '%s'" % str(value))
        self._xmin, self._ymin, self._zmin = value
        self._check_bad_values()

    @max.setter
    def max(self, value):
        """ Set the maximum point (xmax, ymax, zmax). """
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'value' : array_types,
        })
        value = np.asarray(value, dtype = self.max.dtype)
        if value.shape != (3,):
            raise ValueError("You must give a list, tuple, or np.ndarray of shape (3,) to set the 'min' property, not '%s'" % str(value))
        self._xmax, self._ymax, self._zmax = value
        self._check_bad_values()
        
    @xy.setter
    def xy(self, value):
        """ Return the xy bounds as [xmin, xmax, ymin, ymax]. """
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'value' : array_types,
        })
        self._xmin, self._xmax, self._ymin, self._ymax = value
        self._check_bad_values()

    @yx.setter
    def yx(self, value):
        """ Return the yx bounds as [ymin, ymax, xmin, xmax]. """
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'value' : array_types,
        })
        self._ymin, self._ymax, self._xmin, self._xmax = value
        self._check_bad_values()

    @xz.setter
    def xz(self, value):
        """ Return the xz bounds as [xmin, xmax, zmin, zmax]. """
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'value' : array_types,
        })
        self._xmin, self._xmax, self._zmin, self._zmax = value
        self._check_bad_values()

    @zx.setter
    def zx(self, value):
        """ Return the zx bounds as [zmin, zmax, xmin, xmax]. """
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'value' : array_types,
        })
        self._zmin, self._zmax, self._xmin, self._xmax = value
        self._check_bad_values()

    @yz.setter
    def yz(self, value):
        """ Return the yz bounds as [ymin, ymax, zmin, zmax]. """
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'value' : array_types,
        })
        self._ymin, self._ymax, self._zmin, self._zmax = value
        self._check_bad_values()

    @zy.setter
    def zy(self, value):
        """ Return the zy bounds as [zmin, zmax, ymin, ymax]. """
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'value' : array_types,
        })
        self._zmin, self._zmax, self._ymin, self._ymax = value
        self._check_bad_values()
    

    @center.setter
    def center(self, value):
        """ Set the center position. """
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'value' : array_types,
        })
        value = np.asarray(value, dtype = self.center.dtype)
        if value.shape != (3,):
            raise ValueError("You must give a list, tuple, or np.ndarray of shape (3,) to set the 'center' property, not '%s'" % str(value))
        difference = value - self.center
        self._xmin += difference[0]
        self._xmax += difference[0]
        self._ymin += difference[1]
        self._ymax += difference[1]
        self._zmin += difference[2]
        self._zmax += difference[2]
        self._check_bad_values()


    def _check_bad_values(self):
        if self.width < 0: raise NegativeValueError('width', self.width)
        if self.breadth < 0: raise NegativeValueError('breadth', self.breadth)
        if self.height < 0: raise NegativeValueError('height', self.height)

    def get_cli_string(self):
        string  = ["xmin, xmax = %s, %s" % (str(self.xmin), str(self.xmax))]
        string += ["ymin, ymax = %s, %s" % (str(self.ymin), str(self.ymax))]
        string += ["zmin, zmax = %s, %s" % (str(self.zmin), str(self.zmax))]
        string += [""]
        string += ["width (x)   = %s" % str(self.width)]
        string += ["breadth (y) = %s" % str(self.breadth)]
        string += ["height (z)  = %s" % str(self.height)]
        string += [""]
        string += ["radius = %s" % str(self.radius)]
        string += ["volume = %s" % str(self.volume)]
        string += ["center = (%s, %s, %s)" % (
            str(self.center[0]),
            str(self.center[1]),
            str(self.center[2]),
        )]
        return '\n'.join(string)
        


class RadialExtents(Extents, object):
    """
    This type of Extents works only with an :class:`~.lib.output.Output` object.
    """
    def __init__(
            self,
            output : starsmashertools.lib.output.Output,
    ):
        self.output = output
        units = self.output.simulation.units
        x = self.output['x'] * float(units['x'])
        y = self.output['y'] * float(units['y'])
        z = self.output['z'] * float(units['z'])
        radii = 2 * self.output['hp'] * float(units['hp'])

        xmin = starsmashertools.lib.units.Unit(
            np.amin(x - radii),
            units.length.label,
        )
        xmax = starsmashertools.lib.units.Unit(
            np.amax(x + radii),
            units.length.label,
        )
        ymin = starsmashertools.lib.units.Unit(
            np.amin(y - radii),
            units.length.label,
        )
        ymax = starsmashertools.lib.units.Unit(
            np.amax(y + radii),
            units.length.label,
        )
        zmin = starsmashertools.lib.units.Unit(
            np.amin(z - radii),
            units.length.label,
        )
        zmax = starsmashertools.lib.units.Unit(
            np.amax(z + radii),
            units.length.label,
        )

        return super(RadialExtents, self).__init__(
            xmin = xmin, xmax = xmax,
            ymin = ymin, ymax = ymax,
            zmin = zmin, zmax = zmax,
        )

    def __repr__(self):
        return super(RadialExtents, self).__repr__().replace('Extents', 'RadialExtents')

    @property
    def radius(self):
        """ Get the maximum of r + 2*h where r is the radial position of the 
        particles and h is their smoothing lengths."""
        return np.amax(self.output['r'] + 2 * self.output['hp']) * float(self.output.simulation.units.length)
