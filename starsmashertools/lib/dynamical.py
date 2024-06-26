import starsmashertools.preferences
from starsmashertools.preferences import Pref
import starsmashertools.lib.simulation
from starsmashertools.helpers.apidecorator import api
from starsmashertools.helpers.clidecorator import cli
from starsmashertools.helpers.archiveddecorator import archived
import numpy as np

@starsmashertools.preferences.use
class Dynamical(starsmashertools.lib.simulation.Simulation, object):
    def _get_children(self, *args, **kwargs):
        import starsmashertools.helpers.path
        import starsmashertools.lib.binary
        
        search_directory = self.get_search_directory()
        restartradfile = self.get_start_file()
        duplicate = starsmashertools.helpers.path.find_duplicate_file(
            restartradfile, search_directory, throw_error=True)
        dirname = starsmashertools.helpers.path.dirname(duplicate)
        return [starsmashertools.lib.binary.Binary(dirname)]
    
    @api
    @cli('starsmashertools')
    def get_relaxations(self, cli : bool = False, **kwargs):
        binary = self.get_binary(**kwargs)
        if binary is None: return []
        return binary.get_children(cli = cli, **kwargs)
    
    @api
    @cli('starsmashertools')
    def get_binary(self, cli : bool = False, **kwargs):
        import starsmashertools.lib.binary
        children = self.get_children(**kwargs)
        if len(children) != 1: return None
        if not isinstance(children[0], starsmashertools.lib.binary.Binary):
            return None
        return children[0]
    
    @archived
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    @cli('starsmashertools')
    def get_plunge_time(
            self,
            threshold_frac : int | float | type(None) = Pref('get_plunge_time.threshold_frac', None),
            threshold : int | float | type(None) = Pref('get_plunge_time.threshold', 0.05),
            cli : bool = False,
    ):
        """
        Obtain the plunge-in time for a dynamical simulation which originated
        from a binary simulation. The plunge time here is defined as the time at
        which some amount of mass from the secondary star has entered a distance
        from the primary equal to the primary's original SPH radius (from
        relaxation).

        Parameters
        ----------
        threshold_frac : int, float, None, default = ``Pref('get_plunge_time.threshold_frac', None)``
            The fraction of the secondary's total starting mass that must be
            within the secondary's initial radius to identify the plunge-in
            event.

        threshold : int, float, None, default = ``Pref('get_plunge_time.threshold', 0.05)``
            The mass in solar masses of the secondary star which must be within
            the primary's starting radius in order to detect a plunge-in event.
        
        Returns
        -------
        :class:`~.lib.units.Unit` or None
            The plunge time. Returns `None` if there is not yet any plunge-in
            event in this simulation.
        """
        import starsmashertools.helpers.midpoint
        import starsmashertools.math
        import numpy as np
        import starsmashertools.helpers.path

        if threshold_frac is None and threshold is None:
            raise ValueError("Cannot specify both keywords 'threshold_frac' and 'threshold' as 'None'. You must give at least one.")
        if None not in [threshold_frac, threshold]:
            raise ValueError("At least one of keywords 'threshold_frac' or 'threshold' must be 'None': 'threshold_frac' = %g, 'threshold' = %g" % (threshold_frac, threshold))

        if threshold_frac is not None:
            if threshold_frac <= 0 or threshold_frac > 1:
                raise ValueError("Keyword 'threshold_frac' must be within range (0, 1], not '%g'" % threshold_frac)
        
        binary = self.get_binary()
        if binary is None:
            raise Exception("Cannot obtain the plunge time for a dynamical simulation that did not originate from a Binary simulation: '%s'" % self.directory)

        primary, secondary = binary.get_children()
        
        primary_IDs = binary.get_primary_IDs()
        secondary_IDs = binary.get_secondary_IDs()
        
        bout0 = binary.get_output(0)
        if binary.isPrimaryPointMass():
            RSPH = 2 * bout0['hp'][primary_IDs][0]
        else:
            o = primary.get_output(-1)
            RSPH = np.amax(o['r'] + 2 * o['hp'])

        RSPH2 = RSPH**2

        if threshold_frac is not None:
            if binary.isSecondaryPointMass():
                M2 = bout0['am'][secondary_IDs][0]
            else:
                o = secondary.get_output(-1)
                M2 = np.sum(o['am'])
            threshold = M2 * threshold_frac

        def get_mwithin(output):
            if binary.isPrimaryPointMass():
                com = np.asarray([
                    output['x'][primary_IDs],
                    output['y'][primary_IDs],
                    output['z'][primary_IDs],
                ])
            else:
                with starsmashertools.mask(output, primary_IDs):
                    bound = ~output['unbound']
                    if bound.any():
                        xcom1, ycom1, zcom1 = starsmashertools.math.center_of_mass(
                            output['am'][bound],
                            output['x'][bound],
                            output['y'][bound],
                            output['z'][bound],
                        )
                    else: return 0.
                com = np.asarray([xcom1, ycom1, zcom1])
            if binary.isSecondaryPointMass():
                dist2 = np.sum((com - output['xyz'][secondary_IDs])**2)
                if dist2 <= RSPH2: return output['am'][secondary_IDs]
            else:
                with starsmashertools.mask(output, secondary_IDs):
                    dist2 = np.sum((com - output['xyz'])**2, axis=-1)
                    within = dist2 <= RSPH2
                    if within.any(): return np.sum(output['am'][within])
            return 0.


        # Check for presence of plunge-in
        ret = None
        if get_mwithin(self.get_output(-1)) >= threshold:
            try:
                m = starsmashertools.helpers.midpoint.Midpoint(self.get_output())
                m.set_criteria(
                    lambda output: get_mwithin(output) < threshold,
                    lambda output: get_mwithin(output) == threshold,
                    lambda output: get_mwithin(output) > threshold,
                )
                output = m.get()
                ret = output['t'] * self.units['t']
            except starsmashertools.helpers.argumentenforcer.ArgumentTypeError as e:
                raise Exception("You are likely missing output files") from e
        
        if cli:
            if ret is None: return 'No plunge-in detected yet'
            return str(output) + "\n" + str(ret.auto())
        return ret
            
        


    #@starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def get_orbital_energy(
            self,
            *args,
            origin : tuple | list | np.ndarray = (0, 0, 0),
            filter_unbound : bool = True,
            **kwargs
    ):
        """
        Obtain the sum of the orbital energy of each particle about some origin,
        where origin (0, 0, 0) is the center of mass. Two metrics are returned:
        the orbital energy in the phi hat direction and the theta hat direction,
        where phi is the azimuthal angle and theta is the polar angle. The
        orbital energy of a single particle is its kinetic energy in some given
        direction :math:`mv^2/2`.

        We obtain the velocities by first writing the velocity in spherical
        coordinates

        .. math::

            \\vec{v} = v_r\\hat{r} + r\\dot{\\theta}\\hat{\\theta} + \\dot{\\phi}r\\sin\\phi\\hat{\\phi}

        In this way, the velocity components are 
        :math:`v_\\theta = r\\dot{\\theta}` and 
        :math:`v_\\phi = \\dot{\\phi}r\\sin\\phi`\. To find the angular 
        velocities :math:`\\dot{\\theta}` and :math:`\\dot{\\phi}`\, we use 
        their spherical coordinate relations :math:`\\cos\\theta = z/r` and 
        :math:`\\tan\\phi = y/x`\. Differentiating each with respect to time 
        yields

        .. math::

            \\dot{\\theta} = \\frac{r'}{r^2} \\left( \\frac{zx v_x + zy v_y}{r'^2} - v_z\\right)

        where :math:`r' = (x^2 + y^2)^{1/2}`\, and 
        :math:`\\dot{\\phi} = (x v_y - y v_x) / r'^2`\. Therefore,
        
        .. math::
        
            \\begin{align}
                v_\\theta &= \\frac{y}{r} \\left[ \\frac{zx v_x + zy v_y}{r'^2} - v_z \\right]\\\\
                v_\\phi &= \\frac{r}{r'^2} (x v_y - y v_x)
            \\end{align}

        The corresponding orbital energies of a particle are:
        
        .. math::
        
            \\begin{align}
                E_{orb,i,\\theta} &= \\frac{1}{2} m_i v_{\\theta,i}^2\\\\
                E_{orb,i,\\phi} &= \\frac{1}{2} m_i v_{\\phi,i}^2
            \\end{align}

        Parameters
        ----------
        origin : tuple, list, :class:`numpy.ndarray`\, default = ``(0, 0, 0)``
            The origin relative to the center of mass.

        filter_unbound : bool, default = True
            If `True` then only bound particles are used.
        
        Other Parameters
        ----------------
        *args
            Positional arguments are passed directly to
            :meth:`~.lib.simulation.Simulation.get_output`
        
        **kwargs
            Other keyword arguments are passed directly to
            :meth:`~.lib.simulation.Simulation.get_output`

        Returns
        -------
        list
            The total orbital energy :math:`E_{orb,\\theta}` summed over every 
            particle. If ``filter_unbound`` is `True` then an output that has no
            bound particles has a value of 0.

        list
            The total orbital energy :math:`E_{orb,\\phi}` summed over every 
            particle. If ``filter_unbound`` is `True` then an output that has no
            bound particles has a value of 0.
        """
        outputs = self.get_output(*args, **kwargs)
        if not isinstance(outputs, list): outputs = [outputs]

        Eorbtheta = []
        Eorbphi = []
        for output in outputs:
            if filter_unbound: output.mask(~output['unbound'])
            m = output['am']
            if len(m) == 0:
                Eorbtheta += [0.]
                Eorbphi += [0.]
            
            x, y, z = output['x'], output['y'], output['z']
            x -= origin[0]
            y -= origin[1]
            z -= origin[2]
            
            rprime2 = x**2 + y**2
            r2 = rprime2 + z**2
            rprime = np.sqrt(rprime2)
            r = np.sqrt(r2)
            
            vx = output['vx']
            vy = output['vy']
            vz = output['vz']

            v_theta = y/r * ((z*x*vx + z*y*vy)/rprime2 - vz)
            v_phi = r/rprime2 * (x*vy - y*vx)
            
            Eorbtheta += [0.5 * np.sum(m * v_theta**2)]
            Eorbphi += [0.5 * np.sum(m * v_phi**2)]
        
        if len(outputs) == 1: return Eorbtheta[0], Eorbphi[0]
        return Eorbtheta, Eorbphi
