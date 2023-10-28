import starsmashertools.lib.simulation
import starsmashertools.lib.binary
import starsmashertools.helpers.path as path
import starsmashertools.preferences as preferences
from starsmashertools.helpers.apidecorator import api
from starsmashertools.helpers.clidecorator import cli
import starsmashertools.helpers.midpoint


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
            
