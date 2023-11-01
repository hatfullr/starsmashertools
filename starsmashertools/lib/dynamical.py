import starsmashertools.lib.simulation
import starsmashertools.lib.binary
import starsmashertools.helpers.path as path
import starsmashertools.preferences as preferences
from starsmashertools.helpers.apidecorator import api
from starsmashertools.helpers.clidecorator import cli
import starsmashertools.helpers.midpoint
import numpy as np


class Dynamical(starsmashertools.lib.simulation.Simulation, object):
    def _get_children(self, *args, **kwargs):
        search_directory = kwargs.get('search_directory', preferences.get_default('Simulation', 'search directory'))
        search_directory = path.realpath(search_directory)
        
        restartradfile = self.get_initialfile()
        duplicate = path.find_duplicate_file(restartradfile, search_directory, throw_error=True)

        return [starsmashertools.lib.binary.Binary(path.dirname(duplicate))]

    @api
    def get_initialfile(self):
        # We assume a dynamical run always begins from a restartrad.sph file,
        # and that the file was copied from a restartrad.sph.orig file.
        filename = path.join(self.directory, preferences.get_default('Dynamical', 'original initial restartrad'))
        if not path.isfile(filename):
            raise FileNotFoundError(filename)
        return filename

    @api
    @cli('starsmashertools')
    def get_relaxations(self, cli : bool = False, **kwargs):
        return self.get_children(**kwargs)[0].get_children(cli=cli, **kwargs)

    @api
    @cli('starsmashertools')
    def get_plunge_time(self, threshold : int | float = 0.005, cli : bool = False):
        """
        Obtain the plunge-in time for a dynamical simulation which originated
        from a binary simulation. The plunge time here is defined as the time at
        which the simulation exceeds a given threshold on total ejected mass.

        Parameters
        ----------
        threshold : int, float, default = 0.005
            The ejected mass threshold in simulation units (default solar
            masses).

        Returns
        -------
        :class:`~.lib.units.Unit` or `None`
            The plunge time. Returns `None` if there is not yet any plunge-in
            event in this simulation.
        """
        children = self.get_children()
        if children and isinstance(children[0], starsmashertools.lib.binary.Binary):
            # Check for presence of plunge-in
            if self.get_output(-1)['mejecta'] < threshold:
                if cli: return 'No plunge-in detected yet'
                return None
            
            m = starsmashertools.helpers.midpoint.Midpoint(self.get_output())
            m.set_criteria(
                lambda output: output['mejecta'] < threshold,
                lambda output: output['mejecta'] == threshold,
                lambda output: output['mejecta'] > threshold,
            )
            output = m.get()
            ret = output['t'] * self.units['t']
            if cli: return str(output) + "\n" + str(ret.auto())
            return ret
            
        raise Exception("Cannot obtain the plunge time for a dynamical simulation that did not originate from a Binary simulation")


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
        direction mv^2/2.

        We obtain the velocities by first writing the velocity in spherical
        coordinates: \vec{v} = v_r\hat{r} + r\dot{\theta}\hat{\theta} + 
        \dot{\phi}r\sin\phi\hat{\phi}. In this way, the velocity components are:
        v_\theta = r\dot{\theta} and v_\phi = \dot{\phi}r\sin\phi. To find the
        angular velocities \dot{\theta} and \dot{\phi}, we use their spherical
        coordinate relations \cos\theta = z/r and \tan\phi = y/x.
        Differentiating each with respect to time yields: \dot{\theta} = 
        (r'/r^2) [ (zx v_x + zy v_y) / r'^2 - v_z], where r' = sqrt(x^2 + y^2),
        and \dot{\phi} = (x v_y - y v_x) / r'^2. Therefore,
        
        v_\theta = y/r [ (zx v_x + zy v_y) / r'^2 - v_z ]
          v_\phi = r/r'^2 (x v_y - y v_x)

        The corresponding orbital energies of a particle are:
        
        E_{orb,i,\theta} = m_i v_{\theta,i}^2 / 2
          E_{orb,i,\phi} = m_i v_{\phi,i}^2 / 2

        Parameters
        ----------
        origin : tuple, list, np.ndarray, default = (0, 0, 0)
            The origin relative to the center of mass.

        filter_unbound : bool, default = True
            If `True` then only bound particles are used.
        
        Other Parameters
        ----------------
        *args
            Positional arguments are passed directly to
            :func:`~.lib.simulation.Simulation.get_output`
        
        **kwargs
            Other keyword arguments are passed directly to
            :func:`~.lib.simulation.Simulation.get_output`

        Returns
        -------
        list
            The total orbital energy E_{orb,\theta} summed over every particle.
            If `filter_unbound` is `True` then an output that has no bound
            particles has a value of 0.

        list
            The total orbital energy E_{orb,\phi} summed over every particle. If
            `filter_unbound` is `True` then an output that has no bound
            particles has a value of 0.
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
