import starsmashertools.lib.simulation
import starsmashertools.helpers.path
import starsmashertools.helpers.file
import starsmashertools.lib.logfile
from starsmashertools.helpers.apidecorator import api
from starsmashertools.helpers.clidecorator import cli, clioptions
import starsmashertools.helpers.argumentenforcer
import starsmashertools.math
import starsmashertools.lib.output
import numpy as np
import copy

class Relaxation(starsmashertools.lib.simulation.Simulation, object):
    r"""
    A StarSmasher stellar relaxation, characterized by ``nrelax=1``\.
    """
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
        r"""
        Returns the path to the stellar profile from which this relaxation was
        created, as given by the value of ``profilefile`` in the sph.input file.
        """
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
            initfile = self._get_sphinit_filename()
            #initfile = starsmashertools.helpers.path.join(self.directory, 'sph.init')
            if starsmashertools.helpers.path.isfile(initfile):
                with starsmashertools.helpers.file.open(initfile, 'r', lock = False) as f:
                    f.readline()
                    line = f.readline()

                iname = line.lower().replace('iname=','').replace("'",'').strip()
                self._isPolytrope = iname == '1es'
                return self._isPolytrope
            
            raise Exception("Cannot determine if relaxation is a polytrope because it is missing log files and also missing init file '%s'" % initfile)

        if self._isPolytrope is None:
            raise Exception("Internal failure. Could not determine if relaxation is a polytrope because it is missing log files and init file '%s'" % initfile)
        
        return self._isPolytrope

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @cli('starsmashertools')
    @clioptions(display_name = 'Total number of particles')
    def get_n(self, cli : bool = False):
        if self._n is None:
            # Check for log files
            logfiles = self.get_logfiles()
            if logfiles:
                self._n = int(logfiles[0].get('parent: n=').replace('parent: n=', '').strip())
            else: # If there's no log files, check for output files
                try:
                    data = self.get_output(0)
                except IndexError:
                    raise Exception("Cannot find the number of particles because there are no log files and there are no data files in '%s'" % self.directory)

                header = data.read(return_headers=True, return_data=False)
                self._n = header['ntot']
        if cli: return str(self._n)
        return self._n

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    @cli('starsmashertools')
    @clioptions(display_name = 'Bounding box at end of time integration')
    def get_final_extents(self, cli : bool = False):
        r"""
        Returns the results of :meth:`~.lib.output.Output.get_extents` with
        keyword ``radial = True`` for the final output file in this simulation.
        """
        output = self.get_output(-1)
        extents = output.get_extents(radial = True)
        if cli:
            rmax = np.amax(output['r']) * self.units.length
            string = extents.get_cli_string()
            string += "\n\nmax(r_i) = %s" % str(rmax)
            string += "\n\nlength unit = %s" % str(self.units.length)
            return string
        return extents

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    @cli('starsmashertools')
    @clioptions(display_name = 'Envelope binding energy')
    def get_binding_energy(
            self,
            output : starsmashertools.lib.output.Output | starsmashertools.lib.output.OutputIterator,
            mass_coordinate : int | float | type(None) = None,
            cli : bool = False,
    ):
        r"""
        Returns the binding energy of the star's envelope 
        :math:`E_\\mathrm{SPH}` at some specified mass coordinate :math:`m(r)`\,
        as defined in equation (12) of Hatfull et al. (2021).

        Parameters
        ----------
        output : :class:`~.lib.output.Output`\, :class:`~.output.OutputIterator`
           If a :class:`~.lib.output.Output` is given, the binding energy will
           be calculated just for that output file and a single value will be 
           returned. If a :class:`~.output.OutputIterator` is given then the
           binding energy will be calculated for all 
           :class:`~.output.Output` objects in the iterator and a list of
           values will be returned.
        
        Other Parameters
        ----------
        mass_coordinate : int, float, None, default = None
           The mass coordinate :math:`m(r)` to calculate the binding energy at. 
           Set to `None` to get the total binding energy of the envelope, 
           skipping the core particle if there is one. That is, 
           :math:`m(r) = m_{core\\ particle}`\. Set to 0 to get the binding 
           energy of the entire star, including any core particles (perhaps not
           valid?). A :py:class:`ValueError` is raised if this argument is 
           negative.

        Returns
        -------
        :class:`~.units.Unit` or list
           If ``output`` is of type :class:`~.output.Output` then a 
           :class:`~.units.Unit` is returned in cgs units. If ``output`` is
           of type :class:`~.output.OutputIterator` then a list of 
           :class:`~.units.Unit` objects is returned.
        
        See Also
        --------
        :meth:`~.output.Output.get_core_particles`\, :class:`~.units.Unit`
        """
        import starsmashertools.lib.output
        import starsmashertools.lib.units
        
        if mass_coordinate is not None and mass_coordinate < 0:
            raise ValueError("Positional argument 'mass_coordinate' cannot be negative, but received %s" % str(mass_coordinate))
        
        if isinstance(output, starsmashertools.lib.output.OutputIterator):
            result = []
            for o in output:
                result += [self.get_binding_energy(
                    o,
                    mass_coordinate = mass_coordinate,
                )]
            return result
        
        if mass_coordinate is None:
            cores = output.get_core_particles()
            if len(cores) > 1:
                raise NotImplementedError("Found %d core particles, but expected at most 1" % len(cores))
            if len(cores) == 0: mass_coordinate = 0
            else: mass_coordinate = output['am'][cores[0]]
        
        xyz = np.column_stack((
            output['x'], output['y'], output['z'],
        )) * float(self.units.length)
        r2 = np.sum(xyz**2, axis=-1)
        idx = np.argsort(r2)
        
        m = output['am'] * float(self.units['am'])
        mr = np.cumsum(m[idx])[np.argsort(idx)]
        
        keep = mr >= mass_coordinate
        m = m[keep] 
        u = output['u'][keep] * float(self.units['u'])
        r = np.sqrt(r2[keep]) 
        mr = mr[keep]

        G = float(starsmashertools.lib.units.constants['G'])
        
        ret = np.sum((G * mr / r - u) * m)
        return starsmashertools.lib.units.Unit(ret, self.units.energy.label)

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_initial_shells(
            self,
            output : starsmashertools.lib.output.Output | type(None) = None,
            tolerance : float = 1.e-4,
    ):
        r"""
        Each StarSmasher star is initialized by creating many SPH particles in
        a close-packed hexagonal lattice (Kittel, 1976), a configuration that is
        stable to perturbations (Lombardi et al., 2006). The process creates
        particles that share radial positions with several other particles, 
        forming an analogue to "shells" in stellar evolution code MESA. Each
        particle in a shell is spread out on the shell in some distribution.

        Shells are detected by checking for differences in particle radial 
        positions. First all the particles are sorted by their radial positions,
        then the relative fractional difference 
        :math:`\delta=|r_{i+1} - r_i|/((|r_{i+1}| + |r_i|)/2)` is calculated.
        A shell is detected whenever :math:`\delta > \mathrm{tolerance}`\.
        
        Other Parameters
        ----------------
        output : :class:`~.lib.output.Output`\, ``None``\, default = ``None``
            The output file to obtain shells for. If ``None``\, then the very
            first output file in this relaxation is used.
        
        tolerance : float, default = 1.e-4
            The threshold by which a change in particle radial positions is
            detected as a change in shells.

        Returns
        -------
        shells : list
            Each element is a :class:`~.Shell`\, which contains the
            particle indices associated with each shell and the radial position
            of each shell.
        
        See Also
        --------
        Kittel, C. 1976, Introduction to solid state physics (Wiley)

        Lombardi, J. C., J., Proulx, Z. F., Dooley, K. L., et al. 2006, ApJ, 640, 441, doi: 10.1086/499938
        """
        import starsmashertools
        
        if output is None: output = self.get_output(0)
        r2 = (np.column_stack((output['x'], output['y'], output['z']))**2).sum(axis = 1)
        
        shells = []
        with starsmashertools.mask(output, np.argsort(r2)) as masked:
            # This sorts particles by squared radial positions
            x = masked['x'] * float(self.units['x'])
            y = masked['y'] * float(self.units['y'])
            z = masked['z'] * float(self.units['z'])
            r = np.sqrt((np.column_stack((x,y,z))**2).sum(axis = 1))
            mr = 0 # mass coordinate
            shell_mass = 0
            particles = set()
            for ID, r0, r1, mi in zip(
                    masked['ID'][:-1],
                    r[:-1],
                    r[1:],
                    masked['am'] * float(self.units['am']),
            ):
                particles.update([ID])
                mr += mi
                shell_mass += mi
                delta = 2 * abs(r1 - r0) / (abs(r0) + abs(r1)) # rel frac diff
                if delta > tolerance: # shell detected
                    shells += [Shell(
                        particles,
                        0.5 * (r0 + r1),
                        mr,
                        shell_mass,
                    )]
                    particles = set() # reset
                    shell_mass = 0
            
            # Add one final shell for the outermost particles
            if particles:
                dr = shells[-1].r - shells[-2].r
                shells += [Shell(
                    particles,
                    shells[-1].r + dr,
                    mr,
                    shell_mass,
                )]
        return shells


    
    

            
            
    

    class Profile(dict, object):
        r"""
        A stellar profile from which a relaxation originated.
        """
        @api
        def __init__(self, path : str, readonly : bool = True):
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
            with starsmashertools.helpers.file.open(self.path, 'r', lock = False) as f:
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
            import starsmashertools.lib.units
            
            if not isinstance(output.simulation, starsmashertools.lib.relaxation.Relaxation):
                raise TypeError("Can only do an energy comparison on output files from a Relaxation. Received an output file from a '%s'." % (type(outputfile.simulation).__name__))


            G = starsmashertools.lib.units.constants['G']
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



class Shell(object):
    r"""
    Holds information about particles which form a concentric shell in a 
    relaxation. This generally only applies to the initial state of the
    to-be-relaxed star.

    Attributes
    ----------
    particles : :class:`numpy.ndarray`
        An array of unique particle indices associated with this shell. Particle
        indices are expected to be unique among all shells.

    r : float
        The radial position in cm corresponding to the surface of this shell.
    
    mr : float
        The mass coordinate in grams at the surface of this shell.

    mass : float
        The total mass in grams of the particles assocaited with this shell.
    """
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(
            self,
            particles : list | tuple | set,
            r : float,
            mr : float,
            mass : float,
    ):
        self.particles = np.asarray(list(particles), dtype = int)
        self.r = r
        self.mr = mr
        self.mass = mass

    def __str__(self): return 'Shell(%f, n=%d)' % (self.r, len(self.particles))
    def __repr__(self): return str(self)

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_overlapping_particles(
            self,
            r : list | tuple | np.ndarray,
            radii : list | tuple | np.ndarray,
            which : str = 'both',
    ):
        r"""
        Return the indices of the particles whose kernels intersect the
        imaginary spherical surface defined by this shell's radial position.

        Parameters
        ----------
        r : list, tuple, :class:`numpy.ndarray`
            The radial positions of the particles. Must have the same shape as
            ``radii`` and must be in units of cm.

        radii : list, tuple, :class:`numpy.ndarray`
            The sizes of the particle kernels. Must have the same shape as ``r``
            and must be in units of cm.

        Other Parameters
        ----------------
        which : str, default = ``'both'``
            The kind of overlap to calculate. If ``'inner'``\, then particles
            are only considered overlapping if they are positioned within the
            shell, :math:`r_i < r_\mathrm{shell}`\. If ``'outer'`` then 
            particles are only considered overlapping if they are positioned 
            outside the shell, :math:`r_i > r_\mathrm{shell}`\. If ``'both'``\,
            then no restraints on the overlapping are used.

        Returns
        -------
        is_overlapping : :class:`numpy.ndarray`
            An array of boolean values with the same shape as ``r`` and
            ``radii``\, where ``True`` values indicate an overlap.
        """
        starsmashertools.helpers.argumentenforcer.enforcevalues({
            'which' : ['inner', 'outer', 'both'],
        })
        is_overlapping = np.abs(r - self.r) <= radii
        if which == 'inner': is_overlapping &= r < self.r
        elif which == 'outer': is_overlapping &= r > self.r
        return is_overlapping
