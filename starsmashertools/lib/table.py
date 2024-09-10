import starsmashertools.helpers.argumentenforcer
from starsmashertools.helpers.apidecorator import api
import numpy as np

class Table(object):
    r"""
    Contains a 2D array of data and supplies methods for looking up values,
    including interpolation.
    """

    class ReadError(Exception): pass
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def __init__(
            self,
            filename : str,
    ):
        import starsmashertools.helpers.path

        if not starsmashertools.helpers.path.isfile(filename):
            raise FileNotFoundError(filename)

        try:
            self.content = self.read_table(filename)
        except Exception as e:
            raise Table.ReadError("Failed to read file '%s'. You may want to check to make sure the file is formatted correctly and hasn't been corrupted.") from e

    def read_table(self, filename : str):
        raise NotImplementedError

    def interpolate(self, *args, **kwargs):
        raise NotImplementedError





class TEOS(Table, object):
    r"""
    The tabulated equation of state (TEOS) used in a StarSmasher simulation. Use
    :meth:`~.interpolate` to obtain values from the TEOS. Note that this class
    is equipped only to read a specific table format. If you are experiencing
    troubles with your table's format, you may need to create your own file 
    reading protocol, likely by creating a subclass. Implement your subclass by
    creating an instance of your subclass and then setting the ``_teos`` 
    attribute of your simulation to that instance. The next time you access the
    ``teos`` attribute, it should return your subclass instance.
    """
    @api
    def __init__(self, *args, verbose : bool = False, **kwargs):
        self.verbose = verbose
        self.info = {}
        self._interpolators = {}
        
        super(TEOS, self).__init__(*args, **kwargs)
        
    def read_table(self, filename : str):
        import starsmashertools.helpers.file
        import starsmashertools.helpers.string
        
        with starsmashertools.helpers.file.open(filename, 'r', lock = False) as f:
            # Read the X, Y, Z, abar, and zbar properties
            for i in range(5):
                value, key = f.readline().strip().split('=')
                self.info[key.strip()] = starsmashertools.helpers.string.parse(value.strip())

            # Read the num, min, max, and step properties
            for i in range(2):
                values, keys = f.readline().strip().split('=')
                keys = keys.split(',')
                for k, key in enumerate(keys[1:]):
                    if i == 0: keys[k + 1] = 'log rho ' + key.strip()
                    elif i == 1: keys[k + 1] = 'log U ' + key.strip()
                    else: raise NotImplementedError("Expected to read only 2 lines of num, min, max, and step information, but read more than that.")

                values = values.split()
                for key, value in zip(keys, values):
                    self.info[key.strip()] = starsmashertools.helpers.string.parse(value)
                    
            f.readline() # skip empty newline

            # This allows for only single spaces in headers
            headers = [h.strip() for h in f.readline().split('  ')]
            headers = [h for h in headers if h not in ['','icode']]

            data = {key:[] for key in headers}
            for i, line in enumerate(f):
                for key, val in zip(headers, line.strip().split()):
                    data[key] += [val]
            
        for key, val in data.items():
            v = val[0]
            result = starsmashertools.helpers.string.parse(v)
            data[key] = np.asarray(val, dtype=object).astype(type(result))

        self._header_order = headers
        
        Nrho = self.info['log rho num']
        Nu = self.info['log U num']
        # Convert all the data into proper 2D arrays
        for key, val in data.items():
            data[key] = val.reshape(Nrho, Nu)

        return data

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    def get_interpolator(
            self,
            which : str | int,
    ):
        if isinstance(which, int): which = self._header_order[which]
        if which not in self._interpolators.keys():
            if self.verbose: print("TEOS: Adding '"+str(which)+"' interpolator")
            import starsmashertools.lib.interpolation
            self._interpolators[which] = starsmashertools.lib.interpolation.BilinearInterpolator(
                self.content[which],
                np.unique(self.content[self._header_order[0]]),
                np.unique(self.content[self._header_order[1]]),
            )
        return self._interpolators[which]
        
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def interpolate(
            self,
            rho : float | list | tuple | np.ndarray,
            u : float | list | tuple | np.ndarray,
            which : str | int,
    ):
        r"""
        Parameters
        ----------
        rho : float, list, tuple, :class:`numpy.ndarray`
            Density values for interpolation, in cgs units.

        u : float, list, tuple, :class:`numpy.ndarray`
            Specific internal energy values for interpolation, in cgs units.

        which : str, int
            The value to obtain by interpolation. If a :py:class:`str` is given,
            it must correspond to one of the table column headers. If a
            :py:class:`int` is given it is interpreted as a column number.
        """
        import starsmashertools.lib.interpolation
        
        if isinstance(which, int): which = self._header_order[which]
        if isinstance(rho, float): rho = np.asarray([rho])
        if isinstance(u, float): u = np.asarray([u])

        rho = np.asarray(rho)
        u = np.asarray(u)

        if rho.shape != u.shape:
            raise Exception("rho and u must have the same dimensions. Found rho with shape %s and u with shape %s" % (str(rho.shape), str(u.shape)))
        
        result = np.full(rho.shape, np.nan)
        idx = np.logical_and(rho != 0, u != 0)
        
        # Create interpolators on-the-fly
        result[idx] = self.get_interpolator(which)(np.log10(rho[idx]), np.log10(u[idx]))
        if isinstance(result, np.ndarray):
            if not result.shape: return float(result)
            if len(result.shape) == 1 and len(result) == 1: return result[0]
        return result



class OpacityTable(Table, object):
    r"""
    Intended for use with simulations that have ncooling = 2 or 3.
    """
    def __init__(self, *args, **kwargs):
        import starsmashertools.lib.interpolation
        super(OpacityTable, self).__init__(*args, **kwargs)
        self.interpolator = starsmashertools.lib.interpolation.BilinearInterpolator(
            self.content['array'],
            self.content['logR'],
            self.content['logT'],
        )
    
    def read_table(self, filename : str):
        import starsmashertools.helpers.file
        data = []
        logT = []
        with starsmashertools.helpers.file.open(filename, 'r', lock = False) as f:
            line = f.readline().strip()
            logR = line.replace('logT,logR', '').split()
            for line in f:
                l = line.strip().split()
                logT += [l[0]]
                data += [l[1:]]
        data = np.asarray(data, dtype=object).astype(float)
        logT = np.asarray(logT, dtype=object).astype(float)
        logR = np.asarray(logR, dtype=object).astype(float)

        return {
            'array' : data,
            'logT' : logT,
            'logR' : logR,
        }

    def interpolate(self, rho, T):
        # Need to give rho and T in cgs.
        logT = np.log10(T)
        logR = np.log10(rho) - 3 * logT + 18
        return self.interpolator(logR, logT)
        
