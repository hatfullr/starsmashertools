import numpy as np
import starsmashertools.helpers.argumentenforcer
from starsmashertools.helpers.apidecorator import api
import starsmashertools.math
import inspect
import sys

""" For integrating the kernel functions:
Imagine you are looking at a particle's kernel. Trace a
path through the kernel perpendicular to your line of
sight, like so:

                   x/h (integration path)
                   ^
             _____ |b
          .#`     `o.
        .`        /| `.
       /     r/h / |   \
      |         /  |a   |
      |        o---o----|-> q/h
      |                 |
       \               /
        .,           ,.
          `#._____.#`

We integrate the dimensionless quantity W*h^3 with respect
to x/h from a/h=0 to b/h = sqrt((r/h)^2 - (q/h)^2). Then we
multiply the result by 2 to integrate through the entire
kernel. Note that the result is a dimensionless quantity.
To use the result in calculations, you need to divide it
by h^2, because int W*h^3 d(x/h) = h^2 int W dx.
"""

@starsmashertools.helpers.argumentenforcer.enforcetypes
@api
def get_by_name(name : str):
    r"""
    Returns the kernel function whose attribute ``name`` matches the given name.
    """
    for _name, member in inspect.getmembers(sys.modules[__name__]):
        if not inspect.isclass(member): continue
        if member is _BaseKernel: continue
        if not issubclass(member, _BaseKernel): continue
        if not hasattr(member, 'name'): continue
        if member.name == name: return member

@starsmashertools.helpers.argumentenforcer.enforcetypes
@api
def get_by_nkernel(nkernel : int):
    r"""
    Returns the kernel function whose attribute ``nkernel`` matches the given 
    ``nkernel``\.
    """
    for _name, member in inspect.getmembers(sys.modules[__name__]):
        if not inspect.isclass(member): continue
        if member is _BaseKernel: continue
        if not issubclass(member, _BaseKernel): continue
        if not hasattr(member, 'nkernel'): continue
        if member.nkernel == nkernel: return member

class _BaseKernel(object):
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(
            self,
            compact_support : int | float = 2,
            tabulated : bool = False,
            ntab : int = 100,
            integrated : bool = False,
            nstp : int = 100,
            interpolate_table : bool = True,
    ):
        r"""
        An SPH kernel function in 3D.

        Other Parameters
        ----------------
        compact_support : int, float, default = 2
            Particle smoothing length :math:`h` will be multiplied by this
            number to implement compact support.

        tabulated : bool, default = False
            If `True`\, then values of this function will be tabulated for quick
            lookup. The number of tabulated values is ``ntab``\.

        ntab : int, default = 1000
            If ``tabulate`` is `True`\, this many values will be stored in
            memory for quick lookup.

        integrated : bool, default = False
            If `True`\, this kernel function will be the 2D integrated kernel
            function. See :meth:`~.set_integrated` for more details. This will
            automatically make the kernel function be tabulated.

        nstp : int, default = 100
            If ``integrated`` is `True`\, then this many integration steps are
            used.

        interpolate_table : bool, default = True
            If `True` and if this kernel function is tabulated, calculations
            will be done by linear interpolating across the table. If `False`\,
            then calculations will be done by simply finding the closest value,
            without interpolating.
        """
        self.compact_support = compact_support

        self._ntab = ntab
        self.table = None
        self._integrated = None
        self.interpolate_table = interpolate_table

        if integrated: self.set_integrated(integrated, nstp)
        elif tabulated:
            self._integrated = False
            self.table = np.empty(self._ntab)
            self._setup_table()

    @property
    def integrated(self): return self._integrated

    @property
    def tabulated(self): return self.table is not None

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __call__(
            self,
            x : int | float | list | tuple | np.ndarray,
            h : int | float | list | tuple | np.ndarray,
    ):
        r"""
        Calculate the value of the kernel function at a given distance ``x`` 
        and smoothing length ``h``\. If ``tabulate`` was set in 
        :meth:`~.__init__`\, a table lookup will be done instead of the full
        calculation.

        Parameters
        ----------
        x : int, float
            The distance from the particle.
        
        h : int, float
            The smoothing length of the particle.
        """
        if self.integrated: return self.scaled(x, h) / h**2
        return self.scaled(x, h) / h**3
    
    def _setup_table(self):
        self._tablex = np.linspace(0, 1, self._ntab)
        for i, _x in enumerate(self._tablex):
            self.table[i] = self._scaled(_x, 1./self.compact_support)
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def _scaled(
            self,
            x : int | float | list | tuple | np.ndarray,
            h : int | float | list | tuple | np.ndarray,
    ):
        r"""
        This must be overridden by classes which inherit from this class.
        """
        raise NotImplementedError

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def scaled(self,
            x : int | float | list | tuple | np.ndarray,
            h : int | float | list | tuple | np.ndarray,
    ):
        r"""
        Calculate the scaled value of the kernel function :math:`W(x,h)h^3` at a
        given distance ``x`` and smoothing length ``h``\. If the function is
        integrated, returns :math:`h^2 \\int W(x,h) dx` instead.
        
        Parameters
        ----------
        x : int, float
            The distance from the particle.
        
        h : int, float
            The smoothing length of the particle.
        """
        if x > self.compact_support * h: return 0
        if self.tabulated:
            v = x / (self.compact_support * h)
            if self.interpolate_table:
                return starsmashertools.math.linear_interpolate(
                    self._tablex,
                    self.table,
                    v,
                )[0]
            else:
                return self.table[int(self._ntab * v)]
        return self._scaled(x, h)

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def set_integrated(self, value : bool, nstp : int = 100):
        r"""
        Set this kernel function to be an integrated kernel function if it isn't
        one already, or a non-integrated kernel function if it is. An integrated
        kernel function has only 2 dimensions instead of 3. The integrated form
        is as though the 3D form were squished down into a flat disk. The 
        distance :math:`x` used in :meth:`~.scaled` and :meth:`~.__call__`
        becomes a 2D distance.

        Parameters
        ----------
        value : bool
        
        Other Parameters
        ----------------
        nstp : int, default = 100
            The number of integration steps to use.
        """
        # Skip when no change needed
        if self.integrated == value: return

        if not self.tabulated:
            self.table = np.full(self._ntab, np.nan)
            self._setup_table()

        if value:
            # Transform into an integrated kernel function
            
            #ctab = float(self._ntab - 1) / self.compact_support**2
            integratedkernel = np.zeros(self._ntab)
            voverh = np.linspace(0, self.compact_support, self._ntab)
            for i, voverhi in enumerate(voverh):
                # Get the step sizes first
                boverh = np.sqrt(self.compact_support**2 - voverhi**2)
                xoverh = np.linspace(0, boverh, nstp)
                
                # Now obtain the (squared) distances from the center of the kernel
                u2 = voverhi**2 + xoverh**2
                
                # Get the corresponding kernel function values
                f = starsmashertools.math.linear_interpolate(
                    self._tablex,
                    self.table,
                    np.sqrt(u2) / self.compact_support,
                )
                #f = kernel[(ctab*u2).astype(int,copy=False)]
                
                # Get the step sizes
                stepsize = boverh/float(len(xoverh))
                
                # Use trapezoidal rule
                integratedkernel[i] = 0.5 * (f[0] + f[-1]) * stepsize
                integratedkernel[i] += np.sum(f[1:-1]) * stepsize
            
            integratedkernel *= 2
            self.table = integratedkernel
        
        else: # Transform into a non-integrated kernel function
            # _setup_table handled the transformation for us already.
            pass
        self._integrated = value



        

class UniformKernel(_BaseKernel, object):
    r"""
    Returns :math:`1/h^3` inside the kernel and 0 otherwise.
    """
    name = 'uniform'
    def _scaled(
            self,
            x : int | float,
            h : int | float,
    ):
        return 1 if x <= self.compact_support * h else 0


class CubicSplineKernel(_BaseKernel, object):
    r"""
    The cubic spline SPH kernel. The usual case of compact support :math:`2h` is
    given in Monaghan (1992):
    
    .. math::
        :nowrap:

        W(x,h) = \\frac{1}{\\pi h^3} \\begin{cases}
            1 - \\frac{3}{2}q^2 + \\frac{3}{4}q^3 & \\text{if\\ }0\\leq \\frac{x}{h} \\leq 1 \\
            \\frac{1}{4}(2 - q)^3 & \\text{if\\ } 1\\leq \\frac{x}{h} \\leq 2 \\
            0 & \\text{otherwise}
        \\end{cases}

    Where :math:`q = x/2h`\. The general case can be written for any compact
    support :math:`C`\.

    .. math::
        :nowrap:
    
        W(x,h) = \\frac{1}{\\pi h^3} \\begin{cases}
            6 q^2 (q - 1) + 1 & \\text{if\\ }0\\leq q \\leq \\frac{1}{2} \\
            2(1 - q)^3 & \\text{if\\ } 0.5\\leq q \\leq 1 \\
            0 & \\text{otherwise}
        \\end{cases}

    where now :math:`q = x / (Ch)`\.
    """
    name = 'cubic spline'
    nkernel = 0
    def _scaled(
            self,
            x : int | float,
            h : int | float,
    ):
        if x > self.compact_support * h: return 0
        
        q = x / (self.compact_support * h)
        
        if q <= 0.5: v = 6 * q**2 * (q - 1) + 1
        else: v = 2 * (1 - q)**3
        return 1./np.pi * v

class WendlandC4Kernel(_BaseKernel, object):
    r"""
    The Wendland :math:`C^4` kernel function.
    """
    name = 'wendland c4'
    nkernel = 2
    def _scaled(
            self,
            x : int | float,
            h : int | float,
    ):
        q = x / (self.compact_support * h)
        if q > 1: return 0
        return 495/(256*np.pi)*(1-q)**6*(35./3.*q**2+6*q+1)
