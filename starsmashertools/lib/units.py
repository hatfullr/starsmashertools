import numpy as np
import starsmashertools.preferences
import starsmashertools.helpers.readonlydict
import starsmashertools.helpers.argumentenforcer

# This class is used to convert the raw StarSmasher outputs to cgs units. It
# should never be used for converting StarSmasher outputs to any units other
# than cgs units. You're welcome to manipulate the units in your own code after
# using the values here, but you should never edit the values in this class.
# 
# Note that currently the setting of values 'gram', 'sec', 'cm', and 'kelvin' in
# src/starsmasher.h is not supported. We expect all these values to equal 1.d0
# for now.


class Units(starsmashertools.helpers.readonlydict.ReadOnlyDict, object):
    def __init__(self, simulation):

        # Make sure the given simulation argument is of the right type
        import starsmashertools.lib.simulation
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'simulation' : starsmashertools.lib.simulation.Simulation,
        })
        
        self.simulation = simulation
        
        #self.length = simulation['runit']
        #self.mass = simulation['munit']

        obj = {
            # Header units
            'hco' : self.length,
            'hfloor' : self.length,
            'sep0' : self.length,
            'tf' : self.time,
            'dtout' : self.time,
            't' : self.time,
            'tjumpahead' : self.time,
            'trelax' : self.time,
            'dt' : self.time,
            'omega2' : self.frequency * self.frequency,
            'erad' : self.specificenergy,
            'displacex' : self.length,
            'displacey' : self.length,
            'displacez' : self.length,

            

            # Output file units
            'x' : self.length,
            'y' : self.length,
            'z' : self.length,
            'am' : self.mass,
            'hp' : self.length,
            'rho' : self.density,
            'vx' : self.velocity,
            'vy' : self.velocity,
            'vz' : self.velocity,
            'vxdot' : self.acceleration,
            'vydot' : self.acceleration,
            'vzdot' : self.acceleration,
            'u' : self.specificenergy,
            'udot' : self.specificluminosity,
            'grpot' : self.specificenergy,
            'meanmolecular' : 1.,
            'cc' : 1,
            'divv' : self.velocity / self.length, # If this is divergence of velocity
            'ueq' : self.specificenergy,
            'tthermal' : self.time,


            
            # Extra units. You can add your own here if you want more units, but
            # it's probably best to use the preferences.py file instead.
            'popacity' : self.opacity,
            'uraddot' : self.specificluminosity,
            'temperature' : 1,
            'tau' : 1,
            'dEemergdt' : self.luminosity,
            'dEdiffdt' : self.luminosity,
            'dEmaxdiffdt' : self.luminosity,
            'uraddotcool' : self.specificluminosity,
            'uraddotheat' : self.specificluminosity,
        }

        _locals = {}
        for attr in dir(self):
            _locals[attr] = getattr(self, attr)
        
        for key, val in starsmashertools.preferences.get_default('Units', 'extra').items():
            if isinstance(val, (float, int)):
                obj[key] = val
            elif isinstance(val, str):
                obj[key] = eval(val, {}, _locals)
            else:
                raise TypeError("All values declared in preferences.py in defaults['Units']['extras'] must be type 'float', 'int', or 'str', not '%s'" % type(val).__name__)
        
        super(Units, self).__init__(obj)

    @property
    def length(self): return self.simulation['runit']

    @property
    def mass(self): return self.simulation['munit']
        
    @property
    def time(self): return np.sqrt(self.length**3 / (self.gravconst * self.mass))

    @property
    def frequency(self): return 1. / self.time
        
    @property
    def gravconst(self): return 6.67390e-08 # This comes from src/starsmasher.h

    @property
    def area(self): return self.length * self.length

    @property
    def volume(self): return self.area * self.length
    
    @property
    def energy(self): return self.gravconst * self.mass * self.mass / self.length

    @property
    def velocity(self): return self.length / self.time
    
    @property
    def acceleration(self): return self.velocity / self.time

    @property
    def density(self): return self.mass / self.volume

    @property
    def opacity(self): return self.length**2 / self.mass

    @property
    def luminosity(self): return self.energy / self.time

    @property
    def flux(self): return self.luminosity / self.area

    @property
    def specificenergy(self): return self.energy / self.mass

    @property
    def specificluminosity(self): return self.luminosity / self.mass
