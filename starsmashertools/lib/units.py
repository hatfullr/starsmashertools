import numpy as np
import starsmashertools.preferences
import starsmashertools.helpers.readonlydict
import starsmashertools.helpers.argumentenforcer
import copy


class Unit(float, object):
    exception_message = "Operation '%s' is disallowed on 'Unit' type objects. Please convert to 'float' first using, e.g., 'float(unit)'."
    
    def __new__(self, value, *args, **kwargs):
        return float.__new__(self, value)
    
    def __init__(self, value, label):
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'value' : [float],
            'label' : [str, Unit.Label],
        })
        
        if isinstance(label, str):
            self.label = Unit.Label(label)
        else: self.label = label
        
        float.__init__(value)

    # Returns a factor to multiply with to do a unit conversion
    @staticmethod
    def get_conversion_factor(old_unit, new_unit=None):
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'old_unit' : [str],
            'new_unit' : [str, None],
        })
        conversions = starsmashertools.preferences.get_default('Units', 'unit conversions', throw_error=True)
        for key, val in conversions.items():
            if new_unit is None:
                # Here we simply convert back to a base unit
                if old_unit in val.keys():
                    return val[old_unit], key
            else:
                if old_unit == key and new_unit in val.keys():
                    return 1. / val[new_unit]
                if old_unit in val.keys() and new_unit == key:
                    return val[old_unit]
        raise ValueError("Failed to find conversion ['%s', '%s'] in the 'unit conversions' dict in preferences.py" % old_unit, new_unit)

    # Return a copy of this Unit converted into another Unit specified by 'string'
    def convert(self, old_unit, new_unit):
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'old_unit' : [str],
            'new_unit' : [str],
        })
        factor = Unit.get_conversion_factor(old_unit, new_unit)
        total_factor = 1.
        for val in self.label.left:
            if val == old_unit: total_factor *= factor
        for val in self.label.right:
            if val == old_unit: total_factor /= factor
        return Unit(float(self) * total_factor, self.label.convert(old_unit, new_unit))

    # Returns a copy of this object in exclusively 'cm', 'g', and 's' units.
    def get_base(self):
        ret = copy.copy(self)
        for i, unit in enumerate(ret.label.left):
            if unit in ['cm', 'g', 's']: continue
            factor, new_unit = Unit.get_conversion_factor(unit)
            ret *= factor
            ret.label.left[i] = new_unit
        for i, unit in enumerate(ret.label.right):
            if unit in ['cm', 'g', 's']: continue
            factor, new_unit = Unit.get_conversion_factor(unit)
            ret /= factor
            ret.label.right[i] = new_unit
        return ret

    def __repr__(self): return 'Unit(%g, %s)' % (float(self), str(self.label))
    def __str__(self): return '%g %s' % (float(self), str(self.label))

    def __eq__(self, other):
        starsmashertools.helpers.argumentenforcer.enforcetypes({'other' : [Unit]})
        return super(Unit, self).__eq__(other) and self.label == other.label

    def __reduce__(self):
        # https://docs.python.org/3/library/pickle.html#object.__reduce__
        return (Unit, (float(self), self.label))
    def __reduce_ex__(self, *args, **kwargs):
        return self.__reduce__()

    # self * other = self.__mul__(other)
    def __mul__(self, other):
        starsmashertools.helpers.argumentenforcer.enforcetypes({'other' : [float, Unit]})
        if isinstance(other, Unit):
            return Unit(float(self) * float(other), self.label * other.label)
        return super(Unit, self).__mul__(other)

    # other * self = self.__rmul__(other) if other doesn't have __mul__
    def __rmul__(self, other):
        starsmashertools.helpers.argumentenforcer.enforcetypes({'other' : [float, Unit]})
        return self.__mul__(other)

    # self / other = self.__truediv__(other)
    def __truediv__(self, other):
        starsmashertools.helpers.argumentenforcer.enforcetypes({'other' : [float]})
        if isinstance(other, Unit):
            return Unit(float(self) / float(other), self.label / other.label)
        return super(Unit, self).__truediv__(other)

    # other / self = self.__rtruediv__(other) if other doesn't have __truediv__
    def __rtruediv__(self, other):
        starsmashertools.helpers.argumentenforcer.enforcetypes({'other' : [float]})
        if isinstance(other, Unit):
            return Unit(float(other) / float(self), other.label / self.label)
        return Unit(other / float(self), 1 / self.label)
    

    

    def __add__(self, other):
        starsmashertools.helpers.argumentenforcer.enforcetypes({'other' : [Unit]})
        if self.label != other.label:
            raise Exception("Adding Units can only be done when they have the same Labels.")
        return Unit(float(self) + float(other), self.label)
    def __radd__(self, other):
        starsmashertools.helpers.argumentenforcer.enforcetypes({'other' : [Unit]})
        if self.label != other.label:
            raise Exception("Adding Units can only be done when they have the same Labels.")
        return Unit(float(other) + float(self), other.label)
    def __sub__(self, other):
        starsmashertools.helpers.argumentenforcer.enforcetypes({'other' : [Unit]})
        if self.label != other.label:
            raise Exception("Subtracting Units can only be done when they have the same Labels.")
        return Unit(float(self) - float(other), self.label)
    def __rsub__(self, other):
        starsmashertools.helpers.argumentenforcer.enforcetypes({'other' : [Unit]})
        if self.label != other.label:
            raise Exception("Subtracting Units can only be done when they have the same Labels.")
        return Unit(float(other) - float(self), other.label)

    def __pow__(self, value):
        starsmashertools.helpers.argumentenforcer.enforcetypes({'value' : [float, int]})
        return Unit(float(self)**value, self.label**value)
    
    # Outright disallow the following magic methods
    def __abs__(self, *args, **kwargs):
        raise Exception(Unit.exception_message % 'abs')
    def __bool__(self, *args, **kwargs):
        raise Exception(Unit.exception_message % 'bool')
    def __ceil__(self, *args, **kwargs):
        raise Exception(Unit.exception_message % 'ceil')
    def __divmod__(self, *args, **kwargs):
        raise Exception(Unit.exception_message % 'divmod')
    def __floor__(self, *args, **kwargs):
        raise Exception(Unit.exception_message % 'floor')
    def __floordiv__(self, *args, **kwargs):
        raise Exception(Unit.exception_message % 'floordiv')
    def __int__(self, *args, **kwargs):
        raise Exception(Unit.exception_message % 'int')
    def __mod__(self, *args, **kwargs):
        raise Exception(Unit.exception_message % 'mod')
    def __neg__(self, *args, **kwargs):
        raise Exception(Unit.exception_message % 'neg')
    def __pos__(self, *args, **kwargs):
        raise Exception(Unit.exception_message % 'pos')
    def __rdivmod__(self, *args, **kwargs):
        raise Exception(Unit.exception_message % 'rdivmod')
    def __pos__(self, *args, **kwargs):
        raise Exception(Unit.exception_message % 'pos')
    def __rfloordiv__(self, *args, **kwargs):
        raise Exception(Unit.exception_message % 'rfloordiv')
    def __rmod__(self, *args, **kwargs):
        raise Exception(Unit.exception_message % 'rmod')
    def __round__(self, *args, **kwargs):
        raise Exception(Unit.exception_message % 'round')
    def __rpow__(self, *args, **kwargs):
        raise Exception(Unit.exception_message % 'rpow')
    def __trunc__(self, *args, **kwargs):
        raise Exception(Unit.exception_message % 'trunc')
    def conjugate(self, *args, **kwargs):
        raise Exception(Unit.exception_message % 'conjugate')
    def fromhex(self, *args, **kwargs):
        raise Exception(Unit.exception_message % 'fromhex')
    def hex(self, *args, **kwargs):
        raise Exception(Unit.exception_message % 'hex')
    def imag(self, *args, **kwargs):
        raise Exception(Unit.exception_message % 'imag')
    def real(self, *args, **kwargs):
        raise Exception(Unit.exception_message % 'real')


    class Label(object):
        # Principles:
        #    1. A Label has both a 'long' form and a 'short' form. For example, a
        #       Label with short form 'erg' has long form 'cm*g*g/s*s'.
        #       Notice that there is only 1 '/' symbol. The '/' symbol cannot be
        #       repeated in a Label and all symbols after it are considered division.
        #    2. All labels must contain only units 'cm', 'g', 's', or '1' in the long
        #       forms, and can have only operations '*' and '/'. Each of 'cm', 'g',
        #       and 's' must be written in that order ('cgs') on other sides of the
        #       '/' symbol if there is one. The value '1' can only be written on the
        #       left side of the '/' if there are no other symbols there.
        #    3. Only multiplication and division is allowed on Labels.
        #    4. Whenever a long form is changed, it is then simplified. For example,
        #       if Label a has long form 'cm/s' and Label b has 's' then multiplying
        #       a with b gives 'cm*s/s'. Then we count the number of 'cm' on the
        #       left of the '/' sign and compare it with the number of 'cm' on the
        #       right side. If the two are equal then we remove the first instance of
        #       each from either side. Then we proceed with 'g' and finally 's'.
        #    5. A long form is converted to a short form starting from left-to-right.
        #       For example, a Label with long form 'cm*cm*g*g/cm*s*s'
        #       first simplifies the expression to 'cm*cm*g*g/s*s'. Then
        #       it checks the abbreviations list in order to find conversions. For
        #       example, the 'erg' abbreviation is defined as 'cm*g*g/s*s'.
        #       Thus we look for the presence of 1 'cm' and 2 'g' on the left side of
        #       the '/' and for 2 's' on the right side. If we find enough of such
        #       symbols then we remove those symbols and insert 'erg' on the left-most
        #       side, 'erg * cm'.

        conversions = [
            ['erg', 'cm*g*g/s*s'],
            ['erg/g', 'cm*g/s*s'],
        ]

        def __init__(self, value):
            super(Unit.Label, self).__init__()
            self.left = []
            self.right = []
            self.set(value)

        @staticmethod
        def split(string):
            lhs, rhs = string, []
            if '/' in string: lhs, rhs = string.split('/')
            if lhs == '1': lhs = []
            if lhs: lhs = lhs.split('*')
            if rhs: rhs = rhs.split('*')
            return lhs, rhs

        @staticmethod
        def check(string):
            empty = string.replace('cm', '').replace('g','').replace('s','').replace('*','').replace('/','').replace('1','')
            if len(empty) > 0:
                raise Exception("Argument 'string' contains characters other than 'cm', 'g', 's', '*', '/', and '1'")
        
        @property
        def short(self):
            conversions = starsmashertools.preferences.get_default('Units', 'label conversions')
            if conversions is not None:
                for short, values in Unit.Label.conversions:
                    lhs, rhs = Unit.Label.split(values)
                    left = copy.copy(self.left)
                    right = copy.copy(self.right)
                    for l in lhs:
                        if l in left: left.remove(l)
                        else: break
                    else: # If the left side was successful
                        for r in rhs:
                            if r in right: right.remove(r)
                            else: break
                        else: # If the right side was successful
                            self.left = [short] + left
                            self.right = right
            return self.long
            
        @property
        def long(self):
            if not self.left and self.right:
                string = "1"
            else:
                string = "*".join(self.left)
            if self.right: string += "/"+"*".join(self.right)            
            return string

        # Return a copy of this label with changes
        def convert(self, old_unit, new_unit):
            starsmashertools.helpers.argumentenforcer.enforcetypes({
                'old_unit' : [str],
                'new_unit' : [str],
            })
            ret = copy.copy(self)
            for i, val in enumerate(ret.left):
                if val == old_unit: ret.left[i] = new_unit
            for i, val in enumerate(ret.right):
                if val == old_unit: ret.right[i] = new_unit
            return ret
        
        def simplify(self):
            for item in ['cm', 'g', 's']:
                lc, rc = self.left.count(item), self.right.count(item)
                while item in self.left and item in self.right:
                    self.left.remove(item)
                    self.right.remove(item)
                    
                    plc = copy.copy(lc)
                    prc = copy.copy(rc)
                    lc = self.left.count(item)
                    rc = self.right.count(item)
                    if lc == plc and rc == prc: break

        def set(self, string):
            Unit.Label.check(string)
            self.left, self.right = Unit.Label.split(string)
            self.organize()
            self.simplify()
                    
        def organize(self):
            newleft = []
            newright = []
            for item in ['cm', 'g', 's']:
                if item in self.left:
                    newleft += [item]*self.left.count(item)
                if item in self.right:
                    newright += [item]*self.right.count(item)
            self.left = newleft
            self.right = newright
        
        def __str__(self): return self.short
        def __repr__(self): return self.long
        
        def __eq__(self, other):
            starsmashertools.helpers.argumentenforcer.enforcetypes({'other' : [Unit.Label, str]})
            if isinstance(other, Unit.Label):
                return self.long == other.long
            return self.long == other

        def __mul__(self, other):
            starsmashertools.helpers.argumentenforcer.enforcetypes({'other' : [Unit.Label, int]})
            if isinstance(other, Unit.Label):
                ret = Unit.Label("")
                ret.left = self.left + other.left
                ret.right = self.right + other.right
            else:
                ret = copy.copy(self)
                for i in range(1, other): ret *= self
            ret.organize()
            ret.simplify()
            return ret

        def __rmul__(self, other):
            starsmashertools.helpers.argumentenforcer.enforcetypes({'other' : [int, Unit.Label]})
            return self.__mul__(other)
        
        def __truediv__(self, other):
            starsmashertools.helpers.argumentenforcer.enforcetypes({'other' : [Unit.Label]})
            ret = Unit.Label("")
            ret.left = self.left + other.right
            ret.right = self.right + other.left
            ret.organize()
            ret.simplify()
            return ret
            
        def __rtruediv__(self, other):
            starsmashertools.helpers.argumentenforcer.enforcetypes({'other' : [Unit.Label, int]})
            if isinstance(other, int) and other != 1:
                raise Exception("When dividing an 'int' by a 'Unit.Label', the int must be equal to '1', not '%d'" % other)
            
            ret = Unit.Label("")
            if isinstance(other, Unit.Label):
                ret.left = self.right + other.left
                ret.right = self.left + other.right
            else:
                if self.left: ret.right = self.left
                if self.right: ret.left = self.right
            ret.organize()
            ret.simplify()
            return ret

        def __pow__(self, value):
            starsmashertools.helpers.argumentenforcer.enforcetypes({'value' : [int, float]})
            ret = ""
            if isinstance(value, float):
                num, denom = value.as_integer_ratio()

                if denom > 4:
                    raise Exception("Cannot raise Unit.Label '%s' to the power of '%s'. The maximum whole number denominator is 4." % (str(self), str(value)))
                
                print(num, denom)
                ret = 1
                for i in range(num): ret *= self

                for item in ['cm', 'g', 's']:
                    if ((item in ret.left and denom > ret.left.count(item)) or
                        (item in ret.right and denom > ret.right.count(item))):
                        raise Exception("Cannot raise Unit.Label '%s' to the power of '%s'" % (str(self), str(value)))
                    if item in ret.left:
                        for i in range(denom):
                            if ret.left.count(item) == 1: break
                            ret.left.remove(item)
                    if item in ret.right:
                        for i in range(denom):
                            if ret.right.count(item) == 1: break
                            ret.right.remove(item)
            else:
                if value > 0:
                    ret = 1
                    for i in range(value): ret *= self
                elif value < 0:
                    ret = 1
                    for i in range(abs(value)): ret /= self
            return ret
    
    

class Units(starsmashertools.helpers.readonlydict.ReadOnlyDict, object):
# This class is used to convert the raw StarSmasher outputs to cgs units. It
# should never be used for converting StarSmasher outputs to any units other
# than cgs units. You're welcome to manipulate the units in your own code after
# using the values here, but you should never edit the values in this class.
# 
# Note that currently the setting of values 'gram', 'sec', 'cm', and 'kelvin' in
# src/starsmasher.h is not supported. We expect all these values to equal 1.d0
# for now.
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
    def length(self): return Unit(self.simulation['runit'], 'cm')

    @property
    def mass(self): return Unit(self.simulation['munit'], 'g')

    # This comes from src/starsmasher.h
    @property
    def gravconst(self): return Unit(6.67390e-08, 'cm*cm*cm/g*s*s') 
        
    @property
    def time(self): return np.sqrt(self.length**3 / (self.gravconst * self.mass))

    @property
    def frequency(self): return 1. / self.time
        
    

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



    @staticmethod
    def get_best_time_unit(value):
        f = float(value)
        if f <= 100: return f, 's'
        f /= 60.
        if f <= 100: return f, 'min'
        f /= 60.
        if f <= 100: return f, 'hr'
        f /= 24.
        if f <= 100: return f, 'days'
        f /= 365.25
        return f, 'yrs'

    
