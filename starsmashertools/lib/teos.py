import numpy as np
import starsmashertools.helpers.file
from starsmashertools.helpers.apidecorator import api
import starsmashertools.helpers.readonlydict
import starsmashertools.helpers.string

# which=0 gives temperature
# which=1 gives mean molecular mass mu
# which=2 gives pressure
# which=3 gives entropy

class TEOS(starsmashertools.helpers.readonlydict.ReadOnlyDict, object):
    @api
    def __init__(self, filename, verbose=False):
        self.verbose = verbose
        if self.verbose: print("TEOS: Reading '"+filename+"'")

        self.info = {}
        with starsmashertools.helpers.file.open(filename, 'r') as f:
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
        
        self._interpolators = {}
        super(TEOS, self).__init__(data)
        


    @api
    def __call__(self, rho, u, which):
        if isinstance(which, int): which = self._header_order[which]

        # Create interpolators on-the-fly
        if which not in self._interpolators.keys():
            if self.verbose: print("TEOS: Adding '"+str(which)+"' interpolator")
            self._interpolators[which] = Interpolator(
                np.unique(self[self._header_order[0]]),
                np.unique(self[self._header_order[1]]),
                self[which],
            )
        if not hasattr(rho, '__iter__'): rho = [rho]
        if not hasattr(u, '__iter__'): u = [u]
        rho = np.asarray(rho)
        u = np.asarray(u)

        if rho.shape != u.shape:
            raise Exception("rho and u must have the same dimensions. Found rho with shape %s and u with shape %s" % (str(rho.shape), str(u.shape)))
        
        result = np.full(rho.shape, np.nan)
        idx = np.logical_and(rho != 0, u != 0)
        result[idx] = self._interpolators[which](np.log10(rho[idx]), np.log10(u[idx]))
        if isinstance(result, np.ndarray):
            if not result.shape: return float(result)
            if len(result.shape) == 1 and len(result) == 1: return result[0]
        return result

    # In case I get confused about how to call
    @api
    def get(self, *args, **kwargs):
        return self(*args, **kwargs)

    



class Interpolator(object):
    class OutOfBoundsError(ValueError, object): pass
    
    def __init__(self, x, y, z):
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.z = np.asarray(z)
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
        x = np.asarray(x)
        y = np.asarray(y)
        eps = np.finfo(float).eps
        inbounds = np.logical_and(
            np.logical_and(self.xmin - eps <= x, x <= self.xmax + eps),
            np.logical_and(self.ymin - eps <= y, y <= self.ymax + eps),
        )
        if not np.all(inbounds):
            raise Interpolator.OutOfBoundsError("\n   xmin, xmax = %f, %f\n   ymin, ymax = %f, %f\n   x = %s\n   y = %s" % (self.xmin, self.xmax, self.ymin, self.ymax, str(x), str(y)))

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
        
