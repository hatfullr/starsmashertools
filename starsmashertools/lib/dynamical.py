import starsmashertools.lib.simulation as simulation
import starsmashertools.lib.binary as binary
import starsmashertools.helpers.path as path
import starsmashertools.preferences as preferences


class Dynamical(simulation.Simulation, object):
    def __init__(self, *args, **kwargs):
        super(Dynamical, self).__init__(*args, **kwargs)

    def _get_children(self, *args, **kwargs):
        search_directory = kwargs.get('search_directory', preferences.get_default('Simulation', 'search directory'))
        search_directory = path.realpath(search_directory)
        
        restartradfile = self.get_initialfile()
        duplicate = path.find_duplicate_file(restartradfile, search_directory, throw_error=True)

        return [binary.Binary(path.dirname(duplicate))]

    def get_initialfile(self):
        # We assume a dynamical run always begins from a restartrad.sph file,
        # and that the file was copied from a restartrad.sph.orig file.
        filename = path.join(self.directory, preferences.get_default('Dynamical', 'original initial restartrad'))
        if not path.isfile(filename):
            raise FileNotFoundError(filename)
        return filename
        
    def get_relaxations(self, *args, **kwargs):
        print(self.get_children(*args, **kwargs))
        quit()
        return self.get_children(*args, **kwargs)[0].get_children(*args, **kwargs)

