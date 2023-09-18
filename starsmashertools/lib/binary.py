import starsmashertools.lib.simulation as simulation
import starsmashertools.helpers.path as path
import starsmashertools.lib.relaxation as relaxation
import starsmashertools.preferences as preferences
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
    def get_primary(self, output):
        if not isinstance(output, starsmashertools.lib.output.Output):
            raise TypeError("Argument 'output' must be of type 'starsmashertools.lib.output.Output', not '%s'" % type(output).__name__)
        if output not in self:
            raise ValueError("Argument 'output' must be an output file from simulation '%s', not '%s'" % (self.directory, output.simulation.directory))
        return starsmashertools.get_particles(self.get_primary_IDs(), output)
    
    def get_secondary(self, output):
        if not isinstance(output, starsmashertools.lib.output.Output):
            raise TypeError("Argument 'output' must be of type 'starsmashertools.lib.output.Output', not '%s'" % type(output).__name__)
        if output not in self:
            raise ValueError("Argument 'output' must be an output file from simulation '%s', not '%s'" % (self.directory, output.simulation.directory))
        return starsmashertools.get_particles(self.get_secondary_IDs(), output)

    def get_n1(self):
        if self._n1 is None: self._get_n1_n2()
        return self._n1

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
    def isPrimaryPointMass(self): return self.get_n1() == 1
    
    # Returns True if the secondary is a point mass
    def isSecondaryPointMass(self): return self.get_n2() == 1


    def get_start1u(self): return path.join(self.directory, "sph.start1u")
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


    def get_primary_IDs(self): return np.arange(self.get_n1())
        
    def get_secondary_IDs(self):
        n1, n2 = self.get_n1(), self.get_n2()
        # This needs checked
        return np.arange(n1, n1 + n2)

    # Get the centers of mass of both stars
    def get_COMs(self, output):
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
        
        with starsmashertools.mask(output, primary_IDs) as masked_output:
            m = masked_output['am']
            m1 = np.sum(m)
            xcom1 = np.sum(masked_output['x'] * m) / m1
            ycom1 = np.sum(masked_output['y'] * m) / m1
            zcom1 = np.sum(masked_output['z'] * m) / m1

        with starsmashertools.mask(output, secondary_IDs) as masked_output:
            m = masked_output['am']
            m2 = np.sum(m)
            xcom2 = np.sum(masked_output['x'] * m) / m2
            ycom2 = np.sum(masked_output['y'] * m) / m2
            zcom2 = np.sum(masked_output['z'] * m) / m2

        return np.array([xcom1, ycom1, zcom1]), np.array([xcom2, ycom2, zcom2])

    def get_separation(self, output):
        if output not in self:
            raise starsmashertools.lib.simulation.Simulation.OutputNotInSimulationError(self, output)
        
        if isinstance(output, starsmashertools.lib.output.OutputIterator):
            separation = np.full(len(output), np.nan, dtype=float)
            for i,o in enumerate(output):
                separation[i] = self.get_separation(o)
            return separation
        
        com1, com2 = self.get_COMs(output)
        return np.sqrt(np.sum((com1 - com2)**2, axis=-1))

    def get_period(self, output):
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
        with starsmashertools.mask(output, primary_IDs) as masked_output:
            m1 = np.sum(masked_output['am'])
        with starsmashertools.mask(output, secondary_IDs) as masked_output:
            m2 = np.sum(masked_output['am'])
        G = self.units.gravconst
        m1 *= self.units.mass
        m2 *= self.units.mass
        separation *= self.units.length

        return np.sqrt(4 * np.pi**2 / (G * (m1 + m2)) * separation**3)
        

    # Obtain the Roche lobe filling fraction fRLOF for the given output
    # object. If None, then finds fRLOF for all the output files.
    def get_fRLOF(self, output):
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

        with starsmashertools.mask(output, primary) as masked_output:
            m1 = np.sum(masked_output['am'])
            V1 = np.sum(masked_output['am'] / masked_output['rho'])

        with starsmashertools.mask(output, secondary) as masked_output:
            m2 = np.sum(masked_output['am'])
            V2 = np.sum(masked_output['am'] / maksed_output['rho'])

        r_RL1 = starsmashertools.math.rocheradius(m1, m2, separation)
        r_RL2 = starsmashertools.math.rocheradius(m2, m1, separation)
        
        return (0.75 * V1 / np.pi)**(1./3.) / r_RL1, (0.75 * V2 / np.pi)**(1./3.) / r_RL2
        

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
        with starsmashertools.mask(output, self.get_primary_IDs()) as masked_output:
            return np.sum(masked_output['am'])

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
        with starsmashertools.mask(output, self.get_secondary_IDs()) as masked_output:
            return np.sum(masked_output['am'])
