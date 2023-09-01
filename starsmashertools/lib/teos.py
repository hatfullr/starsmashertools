import numpy as np
from scipy.interpolate import LinearNDInterpolator
import collections
import starsmashertools.helpers.file

# which=0 gives temperature
# which=1 gives mean molecular mass mu
# which=2 gives pressure
# which=3 gives entropy

class TEOS:
    def __init__(self, filename, verbose=False):
        self.verbose = verbose
        if self.verbose: print("TEOS: Reading '"+filename+"'")

        f = starsmashertools.helpers.file.open(filename, 'r')
        
        for i in range(5): f.readline()
        line = f.readline().split()
        Nrho = int(line[0])
        rhomin = float(line[1])
        rhomax = float(line[2])
        drho = float(line[3])

        line = f.readline().split()
        Nu = int(line[0])
        umin = float(line[1])
        umax = float(line[2])
        du = float(line[3])

        f.readline()
        headers = f.readline().strip().split("  ")
        headers = [h.strip() for h in headers]
        headers = [h for h in headers if h not in ['','icode']]

        self._tableheaders = headers[2:]
        ncols = len(self._tableheaders)

        coords = np.empty((Nrho*Nu,2) ,dtype=object)
        values = np.empty((Nrho*Nu,ncols), dtype=object)
        for count,line in enumerate(f):
            ls = line.strip().split()
            coords[count] = ls[:2]
            values[count] = ls[2:ncols+2]

        f.close()
                
        self._values = values.astype(float)
        self._coords = coords.astype(float)

        self.interpolators = collections.OrderedDict()

    def __call__(self, rho, u, which):
        if isinstance(which, int): which = self._tableheaders[which]

        # Create interpolators on-the-fly
        keys = list(self.interpolators.keys())
        if which not in keys:
            if self.verbose: print("TEOS: Adding '"+str(which)+"' interpolator")
            self.interpolators[which] = LinearNDInterpolator(self._coords, self._values[:,self._tableheaders.index(which)])

        result = np.full(rho.shape, np.nan)
        idx = np.logical_and(rho != 0, u != 0)
        result[idx] = self.interpolators[which](np.log10(rho[idx]), np.log10(u[idx]))
        return result

    # In case I get confused about how to call
    def get(self, *args, **kwargs):
        return self(*args, **kwargs)

    
