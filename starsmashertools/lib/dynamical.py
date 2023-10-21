import starsmashertools.lib.simulation
import starsmashertools.lib.binary
import starsmashertools.helpers.path as path
import starsmashertools.preferences as preferences
from starsmashertools.helpers.apidecorator import api
from starsmashertools.helpers.clidecorator import cli


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
    def get_relaxations(self, *args, cli : bool = False, **kwargs):
        return self.get_children(*args, **kwargs)[0].get_children(*args, cli=cli, **kwargs)
