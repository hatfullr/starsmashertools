import starsmashertools.lib.simulation
import starsmashertools.helpers.path
import starsmashertools.helpers.file
import starsmashertools.lib.logfile
from starsmashertools.helpers.apidecorator import api
from starsmashertools.helpers.clidecorator import cli
import starsmashertools.math
import numpy as np
import copy

class Relaxation(starsmashertools.lib.simulation.Simulation, object):
    def __init__(self, *args, **kwargs):
        super(Relaxation, self).__init__(*args, **kwargs)
        self._isPolytrope = None
        self._n = None

        self._profile = None
        profilefile = self.get_profilefile()
        if profilefile is not None:
            if starsmashertools.helpers.file.isMESA(profilefile):
                self._profile = Relaxation.MESA(profilefile)
            else:
                self._profile = Relaxation.Profile(profilefile)

    def _get_children(self, *args, **kwargs):
        return []

    @api
    def get_profilefile(self):
        return starsmashertools.helpers.path.join(self.directory, self['profilefile'])

    @property
    def profile(self): return self._profile

    @property
    def isPolytrope(self):
        if self._isPolytrope is None:
            # Check for a log*.sph file
            try:
                logfiles = starsmashertools.lib.logfile.find(self.directory)
            except FileNotFoundError:
                logfiles = None
            if isinstance(logfiles, list) and len(logfiles) > 0:
                logfile = starsmashertools.lib.logfile.LogFile(sorted(logfiles)[0], self)
                try:
                    line = logfile.get('init: new run, iname=')
                except Exception as e:
                    if 'Failed to find' not in str(e): raise
                    line = None
                
                if line is not None:
                    iname = line.replace('init: new run, iname=', '').strip()
                    self._isPolytrope = iname == '1es'
                    return self._isPolytrope

            # If searching the log files was a bust, try looking for an sph.init file
            initfile = starsmashertools.helpers.path.join(self.directory, 'sph.init')
            if starsmashertools.helpers.path.isfile(initfile):
                with starsmashertools.helpers.file.open(initfile, 'r') as f:
                    f.readline()
                    line = f.readline()

                iname = line.lower().replace('iname=','').replace("'",'').strip()
                self._isPolytrope = iname == '1es'
                return self._isPolytrope
            
            raise Exception("Cannot determine if relaxation is a polytrope because it is missing log files and/or an sph.init file '%s'" % self.directory)

        if self._isPolytrope is None:
            raise Exception("Internal failure. Could not determine if relaxation is a polytrope because it is missing log files and/or an sph.init file '%s'" % self.directory)
        
        return self._isPolytrope

    @cli('starsmashertools')
    def get_n(self):
        if self._n is None:
            # Check for log files
            logfiles = self.get_logfiles()
            if logfiles:
                self._n = int(logfiles[0].get('parent: n=').replace('parent: n=', '').strip())
            else: # If there's no log files, check for output files
                try:
                    data = self.get_data(0)
                except IndexError:
                    raise Exception("Cannot find the number of particles because there are no log files and there are no data files in '%s'" % self.directory)

                header = data.read(return_headers=True, return_data=False)
                self._n = header['ntot']
        return self._n




    class Profile(dict, object):
        @api
        def __init__(self, path, readonly=True):
            self.path = path
            self.readonly = readonly
            self._initialized = False
            self._initializing = False
            super(Relaxation.Profile, self).__init__()

        @property
        def isMESA(self): return isinstance(self, Relaxation.MESA)

        def __copy__(self):
            result = self.__class__.__new__(self.__class__)
            result.__dict__.update(self.__dict__)
            return result
        def __deepcopy__(self, memo):
            result = self.__class__.__new__(self.__class__)
            memo[id(self)] = result
            for k, v in self.__dict__.items():
                setattr(result, k, copy.deepcopy(v, memo))
            return result

        def __getitem__(self, *args, **kwargs):
            if not self._initialized and not self._initializing: self.initialize()
            return super(Relaxation.Profile, self).__getitem__(*args, **kwargs)

        def __setitem__(self, *args, **kwargs):
            if not self._initialized and not self._initializing: self.initialize()
            if not self._initializing and self.readonly:
                raise Exception("Cannot edit a profile marked as readonly")
            return super(Relaxation.Profile, self).__setitem__(*args, **kwargs)

        def keys(self, *args, **kwargs):
            if not self._initialized and not self._initializing: self.initialize()
            return super(Relaxation.Profile, self).keys(*args, **kwargs)

        def values(self, *args, **kwargs):
            if not self._initialized and not self._initializing: self.initialize()
            return super(Relaxation.Profile, self).values(*args, **kwargs)

        def _initialize(self):
            raise NotImplementedError
        
        def initialize(self):
            self._initializing = True
            self._initialize()
            self._initializing = False
            self._initialized = True


    class MESA(Profile, object):
        @api
        def __init__(self, *args, **kwargs):
            super(Relaxation.MESA, self).__init__(*args, **kwargs)
        
        def __eq__(self, other):
            return starsmashertools.helpers.file.compare(self.path, other.path)

        def __repr__(self):
            path = copy.copy(self.path)
            if len(path) > 20:
                path = "..." + path[-17:]
            string = self.__class__.__name__ + "(%s)"
            return string % ("'%s'" % path)

        def __str__(self):
            string = self.__class__.__name__ + "(%s)"
            return string % ("'%s'" % self.path)

        def _initialize(self):
            obj = {}
            with starsmashertools.helpers.file.open(self.path, 'r') as f:
                f.readline() # numbers
                obj = {key:val for key, val in zip(f.readline().strip().split(), f.readline().strip().split())}
                f.readline() # newline
                f.readline() # numbers
                header = f.readline().strip().split()
                for key in header: obj[key] = []

                for line in f:
                    for key, val in zip(header, line.strip().split()):
                        obj[key] += [val]

            for key, val in obj.items():
                if isinstance(val, str):
                    obj[key] = starsmashertools.helpers.string.parse(val)
                elif isinstance(val, list):
                    _type = starsmashertools.helpers.string.parse(val[0])
                    obj[key] = np.asarray(val, dtype=object).astype(type(_type))
                else:
                    raise TypeError("Found MESA file contents that are neither string nor list. This should never happen.")

            for key, val in obj.items(): self[key] = val


        # Compare an Output's particle energy values to this MESA file's
        # energy values by mass and centered on mass value 'minterest'.
        # Returns the the total energy comparison, specific internal
        # energy comparison, and the specific gravitational potential
        # energy comparison.
        def energy_comparison(self, output, minterest=None):
            if not isinstance(output.simulation, starsmashertools.lib.relaxation.Relaxation):
                raise TypeError("Can only do an energy comparison on output files from a Relaxation. Received an output file from a '%s'." % (type(outputfile.simulation).__name__))


            G = starsmashertools.lib.units.gravconst
            rsun = output.simulation.units.length
            msun = output.simulation.units.mass

            # mass = mass coordinate of outer boundary of cell
            # energy = specific internal energy @ center of zone (?)
            # logR = radius at outer boundary of zone

            # I think "@ center" means by mass

            mass = self['mass']
            radius = 10**self['logR']
            epot = G * (mass * msun) / (radius * rsun)
            eint = self['energy']

            mcenter = 0.5 * (mass[:-1] + mass[1:])
            mcenter = np.append(mcenter, 0.5 * mass[-1])

            eint_edge = starsmashertools.math.linear_interpolate(mcenter, eint, mass)
            etot = eint_edge + epot

            r2 = output['r2'] # Don't use this after the sort
            idx = np.argsort(r2)[::-1] # We reverse this because MESA profiles
                                       # are always written backwards
            output.sort(idx)
            mass_SPH = np.cumsum(output['am'][idx[::-1]])[::-1]
            radius_SPH = output['r']
            grpot = output['grpot'] * output.simulation.units['grpot']
            u = output['u'] * output.simulation.units['u']

            eint_SPH = starsmashertools.math.linear_interpolate(mass_SPH, u, mcenter)
            epot_SPH = starsmashertools.math.linear_interpolate(mass_SPH, grpot, mass)
            etot_SPH = starsmashertools.math.linear_interpolate(mass_SPH, u + grpot, mass)

            ret = np.full((len(self), 3), np.nan)
            diff0 = etot - etot_SPH
            diff1 = eint - eint_SPH
            diff2 = epot - epot_SPH

            if minterest is None:
                return diff0 / etot, diff1 / eint, diff2 / epot
            else:
                etot_interest = starsmashertools.math.linear_interpolate(mass, etot, minterest)
                eint_interest = starsmashertools.math.linear_interpolate(mcenter, eint, minterest)
                epot_interest = starsmashertools.math.linear_interpolate(mass, epot, minterest)
                return diff0 / etot_interest, diff1 / eint_interest, diff2 / epot_interest
