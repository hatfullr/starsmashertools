import starsmashertools.lib.simulation as simulation
import starsmashertools.helpers.path as path
import starsmashertools.lib.relaxation as relaxation
import starsmashertools.preferences as preferences
from starsmashertools.helpers.apidecorator import api
import starsmashertools.helpers.argumentenforcer
import starsmashertools.math
import starsmashertools
import numpy as np

class Binary(simulation.Simulation, object):
    def __init__(self, *args, **kwargs):
        super(Binary, self).__init__(*args, **kwargs)
        self._n1 = None
        self._n2 = None

    # Returns the particles in the output file which correspond
    # to the primary star (usually the donor)
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_primary(self, output : starsmashertools.lib.output.Output):
        if output not in self:
            raise ValueError("Argument 'output' must be an output file from simulation '%s', not '%s'" % (self.directory, output.simulation.directory))
        return starsmashertools.get_particles(self.get_primary_IDs(), output)

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_secondary(self, output : starsmashertools.lib.output.Output):
        if output not in self:
            raise ValueError("Argument 'output' must be an output file from simulation '%s', not '%s'" % (self.directory, output.simulation.directory))
        return starsmashertools.get_particles(self.get_secondary_IDs(), output)

    @api
    def get_n1(self):
        if self._n1 is None: self._get_n1_n2()
        return self._n1

    @api
    def get_n2(self):
        if self._n2 is None: self._get_n1_n2()
        return self._n2

    def _get_n1_n2(self):
        logfiles = self.get_logfiles()
        if logfiles:
            try:
                self._n1 = int(logfiles[0].get('n1=').replace('n1=','').strip())
            except starsmashertools.lib.logfile.LogFile.PhraseNotFoundError:
                # I think this means the primary is a point mass
                pass

            try:
                self._n2 = int(logfiles[0].get('n2=').replace('n2=','').strip())
            except starsmashertools.lib.logfile.LogFile.PhraseNotFoundError:
                # I think this means the primary is a point mass
                pass
        
        if None in [self._n1, self._n2]:
            # If there's no log files then check for sph.start files
            start1u = self.get_start1u()
            start2u = self.get_start2u()

            if path.isfile(start1u):
                header = self.reader.read(start1u, return_headers=True, return_data=False)
                self._n1 = header['ntot']
            else: # Must be a point mass
                self._n1 = 1
            if path.isfile(start2u):
                header = self.reader.read(start2u, return_headers=True, return_data=False)
                self._n2 = header['ntot']
            else: # Must be a point mass
                self._n2 = 1

    # Returns True if the primary is a point mass
    @api
    def isPrimaryPointMass(self): return self.get_n1() == 1
    
    # Returns True if the secondary is a point mass
    @api
    def isSecondaryPointMass(self): return self.get_n2() == 1

    @api
    def get_start1u(self): return path.join(self.directory, "sph.start1u")
    @api
    def get_start2u(self): return path.join(self.directory, "sph.start2u")

    def _get_children(self, *args, **kwargs):
        children = []

        search_directory = kwargs.get('search_directory', preferences.get_default('Simulation', 'search directory'))
        search_directory = path.realpath(search_directory)

        if self.isPrimaryPointMass():
            children += ['point mass']
        else:
            duplicate = path.find_duplicate_file(self.get_start1u(), search_directory, throw_error=True)
            children += [relaxation.Relaxation(path.dirname(duplicate))]
        
        if self.isSecondaryPointMass():
            children += ['point mass']
        else:
            duplicate = path.find_duplicate_file(self.get_start2u(), search_directory, throw_error=True)
            children += [relaxation.Relaxation(path.dirname(duplicate))]
        
        return children

    @api
    def get_primary_IDs(self): return np.arange(self.get_n1())

    @api
    def get_secondary_IDs(self):
        n1, n2 = self.get_n1(), self.get_n2()
        # This needs checked
        return np.arange(n1, n1 + n2)

    # Get the centers of mass of both stars
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_COMs(self, output : starsmashertools.lib.output.Output | starsmashertools.lib.output.OutputIterator):
        if output not in self:
            raise starsmashertools.lib.simulation.Simulation.OutputNotInSimulationError(self, output)

        if isinstance(output, starsmashertools.lib.output.OutputIterator):
            com1 = np.full((len(output), 3), np.nan, dtype=float)
            com2 = np.full((len(output), 3), np.nan, dtype=float)
            for i,o in enumerate(output):
                com1[i], com2[i] = self.get_COMs(o)
            return com1, com2
        
        primary_IDs = self.get_primary_IDs()
        secondary_IDs = self.get_secondary_IDs()
        
        with starsmashertools.mask(output, primary_IDs) as masked:
            xcom1, ycom1, zcom1 = starsmashertools.math.center_of_mass(
                masked['am'],
                masked['x'],
                masked['y'],
                masked['z'],
            )

        with starsmashertools.mask(output, secondary_IDs) as masked:
            xcom2, ycom2, zcom2 = starsmashertools.math.center_of_mass(
                masked['am'],
                masked['x'],
                masked['y'],
                masked['z'],
            )

        return np.array([xcom1, ycom1, zcom1]), np.array([xcom2, ycom2, zcom2])

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_separation(self, output : starsmashertools.lib.output.Output | starsmashertools.lib.output.OutputIterator):
        if output not in self:
            raise starsmashertools.lib.simulation.Simulation.OutputNotInSimulationError(self, output)
        
        if isinstance(output, starsmashertools.lib.output.OutputIterator):
            separation = np.full(len(output), np.nan, dtype=float)
            for i,o in enumerate(output):
                separation[i] = self.get_separation(o)
            return separation
        
        com1, com2 = self.get_COMs(output)
        return np.sqrt(np.sum((com1 - com2)**2, axis=-1))

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_period(self, output : starsmashertools.lib.output.Output | starsmashertools.lib.output.OutputIterator):
        if output not in self:
            raise starsmashertools.lib.simulation.Simulation.OutputNotInSimulationError(self, output)
        
        if isinstance(output, starsmashertools.lib.output.OutputIterator):
            period = np.full(len(output), np.nan, dtype=float)
            for i,o in enumerate(output):
                period[i] = self.get_period(o)
            return period

        separation = self.get_separation(output)
        
        primary_IDs = self.get_primary_IDs()
        secondary_IDs = self.get_secondary_IDs()
        with starsmashertools.mask(output, primary_IDs) as masked:
            m1 = np.sum(masked['am'])
        with starsmashertools.mask(output, secondary_IDs) as masked:
            m2 = np.sum(masked['am'])
        m1 *= self.units.mass
        m2 *= self.units.mass
        separation *= self.units.length
        return starsmashertools.math.period(float(m1), float(m2), float(separation))
        

    # Obtain the Roche lobe filling fraction fRLOF for the given output
    # object. If None, then finds fRLOF for all the output files.
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_fRLOF(self, output : starsmashertools.lib.output.Output | starsmashertools.lib.output.OutputIterator):
        if output not in self:
            raise starsmashertools.lib.simulation.Simulation.OutputNotInSimulationError(self, output)

        if isinstance(output, starsmashertools.lib.output.OutputIterator):
            fRLOF1 = np.full(len(output), np.nan, dtype=float)
            fRLOF2 = np.full(len(output), np.nan, dtype=float)
            for i,o in enumerate(output):
                fRLOF1[i], fRLOF2[i] = self.get_fRLOF(o)
            return fRLOF1, fRLOF2
        
        separation = self.get_separation(output)

        primary = self.get_primary_IDs()
        secondary = self.get_secondary_IDs()

        with starsmashertools.mask(output, primary) as masked:
            m1 = np.sum(masked['am'])
            V1 = np.sum(masked['am'] / masked['rho'])

        with starsmashertools.mask(output, secondary) as masked:
            m2 = np.sum(masked['am'])
            V2 = np.sum(masked['am'] / masked['rho'])

        r_RL1 = starsmashertools.math.rocheradius(float(m1), float(m2), float(separation))
        r_RL2 = starsmashertools.math.rocheradius(float(m2), float(m1), float(separation))
        
        return (0.75 * V1 / np.pi)**(1./3.) / r_RL1, (0.75 * V2 / np.pi)**(1./3.) / r_RL2
        

    @api
    def get_primary_mass(self):
        logfiles = self.get_logfiles()
        if logfiles:
            try:
                return float(logfiles[0].get('mass1=').replace('mass1=','').strip())
            except starsmashertools.lib.logfile.LogFile.PhraseNotFoundError:
                # I think this means the primary is a point mass
                pass
        # If the log files failed, check the first output file
        output = self.get_output(0)
        with starsmashertools.mask(output, self.get_primary_IDs()) as masked:
            return np.sum(masked['am'])

    @api
    def get_secondary_mass(self):
        logfiles = self.get_logfiles()
        if logfiles:
            try:
                return float(logfiles[0].get('mass2=').replace('mass2=','').strip())
            except starsmashertools.lib.logfile.LogFile.PhraseNotFoundError:
                # I think this means the primary is a point mass
                pass
        # If the log files failed, check the first output file
        output = self.get_output(0)
        with starsmashertools.mask(output, self.get_secondary_IDs()) as masked:
            return np.sum(masked['am'])
