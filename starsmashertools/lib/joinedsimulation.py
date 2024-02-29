import starsmashertools.lib.simulation
import starsmashertools.helpers.argumentenforcer
from starsmashertools.helpers.apidecorator import api
from starsmashertools.helpers.clidecorator import cli

class JoinedSimulation(starsmashertools.lib.simulation.Simulation, object):
    """
    Multiple :py:class:`~.simulation.Simulation` objects merged into one, 
    pulling each time domain into one unified time domain.
    """
    
    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __init__(self, simulations : list | tuple):
        import numpy as np
        
        for simulation in simulations:
            if not isinstance(simulation, starsmashertools.lib.simulation.Simulation):
                raise TypeError("A JoinedSimulation can only consist of type 'Simulation', not '%s'" % type(simulation).__name__)

        if len(simulations) < 2:
            raise Exception("A JoinedSimulation must consist of 2 or more simulations, but received only %d" % len(simulations))

        self._simulations = []
        for simulation in simulations:
            if isinstance(simulation, JoinedSimulation):
                for s in simulation:
                    self._simulations += [s]
            else: self._simulations += [simulation]

        # Order the simulations by times
        times = [simulation.get_output(0)['t'] for simulation in self._simulations]
        times = np.asarray(times, dtype=float)
        self._simulations = [self._simulations[i] for i in np.argsort(times)]
        
        # Update all the simulation archives
        for simulation in self._simulations:
            if simulation.archive is None:
                raise Exception("A JoinedSimulation can only be created on the main thread")

        for simulation in self._simulations:
            others = [s for s in self._simulations if s is not simulation]
            if 'joined simulations' not in simulation.archive:
                simulation.archive.add(
                    'joined simulations',
                    others,
                )
            else:
                l = simulation.archive['joined simulations'].value
                for other in others:
                    if other not in l:
                        l += [other]
                simulation.archive['joined simulations'].value = l
                simulation.archive.save()

        # Remove simulation methods which don't apply to us
        """
        del self.keys
        del self.values
        del self.items
        del self.isRelaxation
        del self.isBinary
        del self.isDynamical
        del self.units
        del self.teos
        del self.compressed
        del self.get_search_directory
        del self.compare_type
        del self._load_children_from_hint_files
        del self._save_children_to_hint_file
        del self._get_compression_filename
        del self._get_sphinit_filename
        del self.get_compressed_properties
        del self.get_children
        del self.get_file
        del self.compress
        del self.decompress
        del self.input
        del self.directory
        """
        

    def __hash__(self):
        return hash(" ".join([simulation.directory for simulation in self]))

    def __len__(self, *args, **kwargs):
        return self._simulations.__len__(*args, **kwargs)
    
    def __getitem__(self, *args, **kwargs):
        return self._simulations.__getitem__(*args, **kwargs)
    
    def __eq__(self, other):
        if not isinstance(other, JoinedSimulation): return False
        if len(self) != len(other): return False
        for i in range(len(self)):
            if self[i] != other[i]: return False
        return True

    @starsmashertools.helpers.argumentenforcer.enforcetypes
    @api
    def __iter__(self, *args, **kwargs):
        return self._simulations.__iter__(*args, **kwargs)

    @api
    def __contains__(self, other):
        if self._simulations.__contains__(other): return True
        for simulation in self:
            if other in simulation: return True
        return False

    @api
    def get_logfiles(self, **kwargs):
        ret = []
        for simulation in self:
            ret += simulation.get_logfiles(**kwargs)
        return ret

    @api
    def get_outputfiles(self, *args, **kwargs):
        ret = []
        for simulation in self:
            ret += simulation.get_outputfiles(*args, **kwargs)
        return ret

    @api
    def get_initialfile(self, *args, **kwargs):
        import numpy as np
        files = [simulation.get_initialfile(*args, **kwargs) for simulation in self]
        # Get the one that has the smallest t value
        times = np.asarray([f['t'] for f in files], dtype=float)
        return files[np.argmin(times)]
    
    @api
    def get_size(self, *args, **kwargs):
        size = 0
        for simulation in self:
            size += simulation.get_size(*args, **kwargs)
        return size

    @api
    def get_output(self, *args, **kwargs):
        pass

    def get_energyfiles(self, *args, **kwargs):
        ret = []
        for simulation in self:
            ret += simulation.get_energyfiles(*args, **kwargs)
        return ret
    
    @api
    def split(self):
        """
        Permanently un-join the simulations which are members of this object.
        This object then becomes `None` to signify it is unusable.
        """
        for simulation in self:
            self.remove(simulation)
        self = None
        
    @api
    def remove(self, simulation : starsmashertools.lib.simulation.Simulation):
        """
        Permanently remove the given simulation from this object.

        Parameters
        ----------
        simulation : starsmashertools.lib.simulation.Simulation
            The simulation to un-join from this object.
        """

        if simulation not in self:
            raise ValueError("%s not in %s" % (simulation, self))

        self._simulations.remove(simulation)

        # Update the simulation archive
        if simulation.archive is None:
            raise Exception("remove can only be called on the main thread")

        l = simulation.archive['joined simulations'].value
        for s in self:
            l2 = s.archive['joined simulations'].value
            l2.remove(s)
            s.archive['joined simulations'].value = l2
            s.archive.save()
            
            l.remove(s)
        
        simulation.archive['joined simulations'].value = l
        simulation.archive.save()
        
