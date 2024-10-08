import starsmashertools.preferences
from starsmashertools.preferences import Pref
import starsmashertools.lib.simulation
import starsmashertools.lib.output
from starsmashertools.helpers.apidecorator import api
from starsmashertools.helpers.clidecorator import cli, clioptions
import starsmashertools.helpers.argumentenforcer
import numpy as np

@starsmashertools.preferences.use
class Binary(starsmashertools.lib.simulation.Simulation, object):
    def __init__(self, *args, **kwargs):
        super(Binary, self).__init__(*args, **kwargs)
        self._n1 = None
        self._n2 = None

    # Returns the particles in the output file which correspond
    # to the primary star (usually the donor)
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_primary(self, output : starsmashertools.lib.output.Output):
        import starsmashertools
        if output not in self:
            raise ValueError("Argument 'output' must be an output file from simulation '%s', not '%s'" % (self.directory, output.simulation.directory))
        return starsmashertools.get_particles(output, self.get_primary_IDs())

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_secondary(self, output : starsmashertools.lib.output.Output):
        import starsmashertools
        if output not in self:
            raise ValueError("Argument 'output' must be an output file from simulation '%s', not '%s'" % (self.directory, output.simulation.directory))
        return starsmashertools.get_particles(output, self.get_secondary_IDs())

    @api
    @cli('starsmashertools')
    @clioptions(display_name = 'Primary star number of particles')
    def get_n1(self, cli : bool = False):
        if self._n1 is None: self._get_n1_n2()
        return self._n1

    @api
    @cli('starsmashertools')
    @clioptions(display_name = 'Secondary star number of particles')
    def get_n2(self, cli : bool = False):
        if self._n2 is None: self._get_n1_n2()
        return self._n2

    def _get_n1_n2(self):
        import starsmashertools.helpers.path
        import starsmashertools.lib.logfile
        
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

            if starsmashertools.helpers.path.isfile(start1u):
                header = self.reader.read(start1u, return_headers=True, return_data=False)
                self._n1 = header['ntot']
            else: # Must be a point mass
                self._n1 = 1
            if starsmashertools.helpers.path.isfile(start2u):
                header = self.reader.read(start2u, return_headers=True, return_data=False)
                self._n2 = header['ntot']
            else: # Must be a point mass
                self._n2 = 1

    # Returns True if the primary is a point mass
    @api
    @cli('starsmashertools')
    @clioptions(display_name = 'Show if the primary star is a point mass')
    def isPrimaryPointMass(self, cli : bool = False): return self.get_n1() == 1
    
    # Returns True if the secondary is a point mass
    @api
    @cli('starsmashertools')
    @clioptions(display_name = 'Show if the secondary star is a point mass')
    def isSecondaryPointMass(self, cli : bool = False): return self.get_n2() == 1

    @api
    def get_start1u(self):
        import starsmashertools.helpers.path
        return starsmashertools.helpers.path.join(self.directory, "sph.start1u")
    @api
    def get_start2u(self):
        import starsmashertools.helpers.path
        return starsmashertools.helpers.path.join(self.directory, "sph.start2u")

    def _get_children(
            self,
            verbose : bool = False,
    ):
        import starsmashertools.lib.relaxation
        import starsmashertools.helpers.path
        search_directory = self.get_search_directory()
        
        if self.isPrimaryPointMass():
            children = ['point mass']
        else:
            duplicate = starsmashertools.helpers.path.find_duplicate_file(self.get_start1u(), search_directory, throw_error=True)
            children = [starsmashertools.lib.relaxation.Relaxation(starsmashertools.helpers.path.dirname(duplicate))]

        if self.isSecondaryPointMass():
            children += ['point mass']
        else:
            duplicate = starsmashertools.helpers.path.find_duplicate_file(self.get_start2u(), search_directory, throw_error=True)
            children += [starsmashertools.lib.relaxation.Relaxation(starsmashertools.helpers.path.dirname(duplicate))]

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
        r"""
        Obtain the center of mass of both stars in the simulation for the given
        outputs.

        Parameters
        ----------
        output : :class:`~.lib.output.Output`\, :class:`~.lib.output.OutputIterator`
            The output or outputs for which to get the center of mass of each of
            the two stars.

        Returns
        -------
        :class:`numpy.ndarray`\, :class:`numpy.ndarray`
            Two NumPy arrays of shape ``(3,)`` containing the ``(x, y, z)``
            center of mass positions of the primary and secondary stars 
            respectively.
        """
        import starsmashertools.math
        
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
        
        with starsmashertools.mask(output, primary_IDs, copy = False) as masked:
            xcom1, ycom1, zcom1 = starsmashertools.math.center_of_mass(
                masked['am'],
                masked['x'],
                masked['y'],
                masked['z'],
            )

        with starsmashertools.mask(output, secondary_IDs, copy = False) as masked:
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
        import starsmashertools.math
        import starsmashertools
        
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
        with starsmashertools.mask(output, primary_IDs, copy = False) as masked:
            m1 = np.sum(masked['am'])
        with starsmashertools.mask(output, secondary_IDs, copy = False) as masked:
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
        import starsmashertools.math
        import starsmashertools
        import warnings
        
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

        with starsmashertools.mask(output, primary, copy = False) as masked:
            m1 = np.sum(masked['am'])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                V1 = np.sum(masked['am'] / masked['rho'])

        with starsmashertools.mask(output, secondary, copy = False) as masked:
            m2 = np.sum(masked['am'])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                V2 = np.sum(masked['am'] / masked['rho'])

        r_RL1 = starsmashertools.math.rocheradius(float(m1), float(m2), float(separation))
        r_RL2 = starsmashertools.math.rocheradius(float(m2), float(m1), float(separation))
        
        return (0.75 * V1 / np.pi)**(1./3.) / r_RL1, (0.75 * V2 / np.pi)**(1./3.) / r_RL2

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    @cli('starsmashertools')
    @clioptions(display_name = 'Get Roche lobe overflow factor f_RLOF')
    def get_RLOF(
            self,
            which : str = 'primary',
            threshold : float | int = 1.,
            CLI_output_window : int = Pref('get_RLOF.CLI output window', 6),
            cli : bool = False,
    ):
        r"""
        Obtain the output file that corresponds with the time of Roche lobe
        overflow (RLOF). RLOF is detected using :meth:`~.get_fRLOF` to obtain 
        the fraction of the Roche lobe that is filled fRLOF in the output files.
        The output files are searched to locate the one whose fRLOF value is 
        closest to ``threshold`` and the file just before that one is returned.
        
        Parameters
        ----------
        which : str, default = ``'primary'``
            The star(s) for which to find the RLOF condition. Must be one of the
            following: ``'primary'``\, ``'secondary'``\, or ``'both'``\. If 
            ``'primary'`` or ``'secondary'``, a single 
            :class:`~.lib.output.Output` object is returned, and if ``'both'``
            then two are returned, which are the primary and secondary stars' 
            RLOF times repsectively.

        threshold : float, int,  default = 1.
            The threshold for detecting Roche lobe overflow. This value is
            compared with fRLOF from :meth:`~.get_fRLOF`\.

        CLI_output_window : int, default = ``Pref('get_RLOF.CLI output window', 6)``
            How many output files to show in the CLI. Only applies when running
            starsmashertools in CLI mode.

        Returns
        -------
        :class:`~.lib.output.Output` or None if ``which = 'primary'`` or ``which = 'secondary'``
            The output file corresponding to the time of Roche lobe overflow for
            the star specified by ``which``\. If the star has only a single
            particle, returns `None`\.

        list of length 2 if ``which = 'both'``
            Each element in the list is either a :class:`~.lib.output.Output` or
            `None`. The first element is for the primary star and the second
            element is for the secondary star.

        See Also
        --------
        :meth:`~.get_fRLOF`
        """
        import starsmashertools.helpers.path
        import starsmashertools.helpers.midpoint
        import starsmashertools.helpers.string
        import os
        
        starsmashertools.helpers.argumentenforcer.enforcevalues({
            'which' : ['primary', 'secondary', 'both'],
        })
        outputs = self.get_output()
        midpoint = starsmashertools.helpers.midpoint.Midpoint(outputs)

        output1 = None
        output2 = None
        fRLOF1 = None
        fRLOF2 = None
        
        if which in ['primary', 'both'] and self.get_n1() > 1:
            midpoint.set_criteria(
                lambda output: self.get_fRLOF(output)[0] < threshold,
                lambda output: self.get_fRLOF(output)[0] == threshold,
                lambda output: self.get_fRLOF(output)[0] > threshold,
            )
            output1, index1 = midpoint.get(favor='low', return_index = True)
            fRLOF1 = self.get_fRLOF(output1)[0]
            if fRLOF1 > threshold: output1 = outputs[max(0, index1 - 1)]
        if which in ['secondary', 'both'] and self.get_n2() > 1:
            if self.get_n2() == 1: output2 = None
            midpoint.set_criteria(
                lambda output: self.get_fRLOF(output)[1] < threshold,
                lambda output: self.get_fRLOF(output)[1] == threshold,
                lambda output: self.get_fRLOF(output)[1] > threshold,
            )
            output2, index2 = midpoint.get(favor='low', return_index = True)
            fRLOF2 = self.get_fRLOF(output2)[1]
            if fRLOF2 > threshold: output2 = outputs[max(0, index2 - 1)]
        
        
        if cli:
            # Give some additional fRLOF values in the vicinity of the found output files
            outputs = self.get_output()

            # Get all the output files we will be using
            all_outputs = []
            tops = [None, None]
            bottoms = [None, None]
            if which in ['primary', 'both']:
                idx = outputs.index(output1)
                if idx + CLI_output_window/2 >= len(outputs):
                    tops[0] = len(outputs)
                    bottoms[0] = max(tops[0] - CLI_output_window - 1, 0)
                elif idx - CLI_output_window/2 < 0:
                    bottoms[0] = 0
                    tops[0] = min(CLI_output_window + 1, len(outputs))
                all_outputs += outputs[bottoms[0]:tops[0]]
            if which in ['secondary', 'both']:
                idx = outputs.index(output2)
                if idx + CLI_output_window/2 >= len(outputs):
                    tops[1] = len(outputs)
                    bottoms[1] = max(tops[1] - CLI_output_window - 1, 0)
                elif idx - CLI_output_window/2 < 0:
                    bottoms[1] = 0
                    tops[1] = min(CLI_output_window*2 + 1, len(outputs))
                all_outputs += outputs[bottoms[1]:tops[1]]
            all_outputs = list(set(all_outputs))
            fRLOFs = {'primary':{}, 'secondary':{}}
            for o in all_outputs:
                primary, secondary = self.get_fRLOF(o)
                fRLOFs['primary'][o] = primary
                fRLOFs['secondary'][o] = secondary
            
            widths = [15, 13]
            header_fmt = '{name:>%d}' % (sum(widths) + 1)
            fmt = '{name:>%ds}{pick:1s}{value:%d.%df}' % tuple([widths[0], widths[1], 7])
            strings = [
                [header_fmt.format(name='Primary')],
                [header_fmt.format(name='Secondary')],
            ]
            
            if which in ['primary', 'both']:
                idx = outputs.index(output1)
                for o in outputs[bottoms[0]:tops[0]]:
                    name = starsmashertools.helpers.string.shorten(
                        os.path.basename(o.path),
                        widths[0],
                        where = 'left',
                    )
                    strings[0] += [fmt.format(
                        name=name,
                        value=fRLOFs['primary'][o],
                        pick='*' if o == output1 else '',
                    )]
            if which in ['secondary', 'both']:
                idx = outputs.index(output2)
                for o in outputs[bottoms[1]:tops[1]]:
                    name = starsmashertools.helpers.string.shorten(
                        os.path.basename(o.path),
                        widths[0],
                        where = 'left',
                    )
                    strings[1] += [fmt.format(
                        name=name,
                        value=fRLOFs['secondary'][o],
                        pick='*' if o == output2 else '',
                    )]
            
            if which == 'primary':
                return '\n'.join(strings[0])
            if which == 'secondary':
                return '\n'.join(strings[1])
            if which == 'both':
                # Normalize the lengths
                length = max(len(strings[0]), len(strings[1]))
                for i in range(length):
                    if i >= len(strings[0]): strings[0] += [""]
                    if i >= len(strings[1]): strings[1] += [""]
                string = ""
                for i in range(length):
                    string += strings[0][i] + strings[1][i] + '\n'
                return string
            raise NotImplementedError("Bad which value '%s'" % str(which))
        
        if which == 'primary': return output1
        if which == 'secondary': return output2
        return output1, output2
        
        
    @api
    @cli('starsmashertools')
    @clioptions(display_name = 'Mass of primary star')
    def get_primary_mass(self, cli : bool = False):
        r"""
        Return the mass of the primary star (usually the donor). First the log
        files are searched, and if no log files are found then the first output
        file is checked. If there are no output files then the sph.start1u file
        is checked, and if it doesn't exist then an error is raised.

        Returns
        -------
        float
            The mass of the primary (usually the donor) star.
        """
        import starsmashertools.helpers.path
        import starsmashertools.lib.logfile
        import starsmashertools
        
        logfiles = self.get_logfiles()
        if logfiles:
            try:
                return float(logfiles[0].get('mass1=').replace('mass1=','').strip())
            except starsmashertools.lib.logfile.LogFile.PhraseNotFoundError:
                # I think this means the primary is a point mass
                pass
        # If the log files failed, check the first output file
        output = self.get_output(0)
        if not output:
            # If there's no output files, then try to get sph.start1u
            start1u = self.get_start1u()
            if not starsmashertools.helpers.path.isfile(start1u):
                raise FileNotFoundError("Cannot get the primary's mass because the simulation has no log files, no output files, and the sph.start1u file is missing, in '%s'" % self.directory)
            output = starsmashertools.lib.output.Output(start1u, self)

        with starsmashertools.mask(output, self.get_primary_IDs(), copy = False) as masked:
            return np.sum(masked['am'])

    @api
    @cli('starsmashertools')
    @clioptions(display_name = 'Mass of secondary star')
    def get_secondary_mass(self, cli : bool = False):
        r"""
        Return the mass of the secondary star (usually the accretor). First the
        log files are searched, and if no log files are found then the first
        output file is checked. If there are no output files then the
        sph.start2u file is checked, and if it doesn't exist then an error is
        raised.

        Returns
        -------
        float
            The mass of the secondary (usually the accretor) star.
        """
        import starsmashertools.helpers.path
        import starsmashertools.lib.logfile
        import starsmashertools
        
        logfiles = self.get_logfiles()
        if logfiles:
            try:
                return float(logfiles[0].get('mass2=').replace('mass2=','').strip())
            except starsmashertools.lib.logfile.LogFile.PhraseNotFoundError:
                # I think this means the primary is a point mass
                pass
        # If the log files failed, check the first output file
        output = self.get_output(0)
        if not output:
            # If there's no output files, then try to get sph.start2u
            start2u = self.get_start2u()
            if not starsmashertools.helpers.path.isfile(start2u):
                raise FileNotFoundError("Cannot get the secondary's mass because the simulation has no log files, no output files, and the sph.start2u file is missing, in '%s'" % self.directory)
            output = starsmashertools.lib.output.Output(start2u, self)
        
        with starsmashertools.mask(output, self.get_secondary_IDs(), copy = False) as masked:
            return np.sum(masked['am'])

    @api
    @cli('starsmashertools')
    @clioptions(display_name = 'Mass of primary star core particle')
    def get_primary_core_mass(self, cli : bool = False):
        r"""
        Return the mass of the core particle in the primary (usually the donor)
        star, if it has a core particle. The log files are checked first, which
        is usually faster than checking the outputs. If something goes wrong 
        with checking the log files, the first output file is checked instead,
        using :meth:`~.lib.output.Output.get_core_particles`\.

        Returns
        -------
        float or `None`
            The mass of the core particle in the primary (usually the donor)
            star. If there is no core particle, returns `None`\.

        See Also
        --------
        :meth:`~.get_secondary_core_mass`\, :meth:`~.lib.output.Output.get_core_particles`
        """
        import starsmashertools.helpers.path
        import starsmashertools.lib.logfile
        import starsmashertools
        
        if self.get_n1() == 1: # The primary is a point mass particle
            return self['mbh']
        
        logfiles = self.get_logfiles()
        search_string = 'is a corepoint of mass'
        if logfiles:
            try:
                return float(logfiles[0].get(search_string).replace(search_string, '').strip())
            except starsmashertools.lib.logfile.LogFile.PhraseNotFoundError:
                # I think this means the primary has no core particle
                return None
            
        # If there's no log files, then check for output files
        output = self.get_output(0)
        if not output:
            # If there's no output files, then try to get sph.start1u
            start1u = self.get_start1u()
            if not starsmashertools.helpers.path.isfile(start1u):
                raise FileNotFoundError("Cannot get the primary's core mass because there is more than 1 particle for the primary star and the simulation has no log files, no output files, and the sph.start1u file is missing, in '%s'" % self.directory)
            output = starsmashertools.lib.output.Output(start1u, self)

        cores = output.get_core_particles()
        if len(cores) == 0: return None
        primary_IDs = self.get_primary_IDs()
        for c in cores:
            if c not in primary_IDs: continue
            return output['am'][c]
        raise NotImplementedError("The primary star has multiple core particles")
    
    @api
    @cli('starsmashertools')
    @clioptions(display_name = 'Mass of secondary core particle')
    def get_secondary_core_mass(self, cli : bool = False):
        r"""
        Return the mass of the core particle in the secondary (usually the
        companion) star, if it has a core particle. The log files are checked 
        first, which is usually faster than checking the outputs. If something 
        goes wrong with checking the log files, the first output file is checked
        instead, using :func:`~.lib.output.Output.get_core_particles`\.

        Returns
        -------
        float or `None`
            The mass of the core particle in the secondary (usually the
            companion) star. If there is no core particle, returns `None`\.
        
        See Also
        --------
        :func:`~.get_primary_core_mass`\, :func:`~.lib.output.Output.get_core_particles`
        """
        import starsmashertools.helpers.path
        import starsmashertools

        if self.get_n2() == 1: # The secondary is a point mass particle
            return self['mbh']
        
        # We can't rely on log files for the secondary, because StarSmasher only
        # gives us hints for the primary core particle.
        
        # Check for output files
        output = self.get_output(0)
        if not output:
            # If there's no output files, then try to get sph.start1u
            start2u = self.get_start2u()
            if not starsmashertools.helpers.path.isfile(start2u):
                raise FileNotFoundError("Cannot get the secondary's core mass because there is more than 1 particle for the secondary star, the simulation has no output files, and the sph.start2u file is missing, in '%s'" % self.directory)
            output = starsmashertools.lib.output.Output(start2u, self)

        cores = output.get_core_particles()
        if len(cores) == 0: return None
        secondary_IDs = self.get_secondary_IDs()
        for c in cores:
            if c not in secondary_IDs: continue
            return output['am'][c]
        raise NotImplementedError("The secondary star has multiple core particles")
