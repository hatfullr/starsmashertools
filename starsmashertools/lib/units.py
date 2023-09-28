import numpy as np
import starsmashertools.preferences
import starsmashertools.helpers.readonlydict
import starsmashertools.helpers.argumentenforcer
import copy




class Unit(float, object):
    exception_message = "Operation '%s' is disallowed on 'Unit' type objects. Please convert to 'float' first using, e.g., 'float(unit)'."
    
    def __new__(self, value, *args, **kwargs):
        return float.__new__(self, value)
    
    def __init__(self, value, label, base=['cm', 'g', 's']):
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'value' : [float],
            'label' : [str, Unit.Label],
        })

        self.base = base
        
        if isinstance(label, str):
            self.label = Unit.Label(label, self.base)
        else: self.label = label
        
        float.__init__(value)

    # Decide on units that give the cleanest value
    def auto(self, threshold=100, conversions=None):
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'threshold' : [float, int],
        })
        if threshold <= 0: raise ValueError("Argument 'threshold' must be > 0, not '%s'" % str(threshold))
        
        # Get all the available conversions
        if conversions is None:
            conversions = starsmashertools.preferences.get_default('Units', 'unit conversions')
        if conversions is None: return self

        base = self.get_base(conversions=conversions)
        if float(base) < threshold: return base

        left, right = Unit.Label.split(self.label.long)
        label = left + right

        arr = []
        for base_unit, conversion_dict in conversions.items():
            if base_unit not in self.base: continue
            if base_unit in label:
                for new_unit, value in conversion_dict.items():
                    arr += [[base_unit, new_unit, value]]
        values = [a[2] for a in arr]
        idx = np.argsort(values)
        arr = [arr[i] for i in idx]

        for a in arr:
            old_unit, new_unit = a[0], a[1]
            b = copy.deepcopy(base)
            b = b.convert(old_unit, new_unit, conversions=conversions)
            if float(b) < threshold: return b
        return self
        
    # Returns a factor to multiply with to do a unit conversion
    # If new_unit is None, then we are simply converting old_unit back to a base
    # unit ('cm', 'g', 's').
    def get_conversion_factor(self, old_unit, new_unit=None, conversions=None):
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'old_unit' : [str],
            'new_unit' : [str, type(None)],
        })
        if conversions is None:
            conversions = starsmashertools.preferences.get_default('Units', 'unit conversions', throw_error=True)

        expected_values = self.label.left + self.label.right

        if len(expected_values) == 0:
            raise Exception("Cannot obtain a conversion factor for Unit with an empty Label")
        
        starsmashertools.helpers.argumentenforcer.enforcevalues({
            'old_unit' : expected_values,
        })

        # Converting base to base unit, but the old unit is already a base unit.
        # Then we don't need to do any conversion!
        if new_unit is None and old_unit in self.base:
            raise ValueError("Cannot get the conversion factor for '%s' to base units %s because it is already a base unit" % (old_unit, starsmashertools.helpers.string.list_to_string(self.base)))

        ret_new_unit = None
        for base_unit, conversion_dict in conversions.items():
            if new_unit is None:
                #if base_unit not in self.base: continue
                # Here we simply convert back to a base unit
                if old_unit not in conversion_dict.keys(): continue
                factor = conversion_dict[old_unit]
                ret_new_unit = base_unit
                break
            else:
                if old_unit == base_unit and new_unit in conversion_dict.keys():
                    factor = 1. / conversion_dict[new_unit]
                    break
                elif old_unit in conversion_dict.keys() and new_unit == base_unit:
                    factor = conversion_dict[old_unit]
                    break
        else: # If we didn't find the units in the conversions dict
            raise ValueError("Failed to find conversion '%s' to '%s'" % (old_unit, new_unit))


        total_factor = 1.
        if old_unit in self.label.left:
            total_factor *= self.label.left.count(old_unit) * factor
        if old_unit in self.label.right:
            total_factor /= self.label.right.count(old_unit) * factor
        
        if ret_new_unit is not None:
            return total_factor, ret_new_unit
        return total_factor
        

    # Return a copy of this Unit converted into another Unit specified by
    # 'string'
    def convert(self, old_unit, new_unit, conversions=None):
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'old_unit' : [str],
            'new_unit' : [str],
        })
        factor = self.get_conversion_factor(old_unit, new_unit, conversions=conversions)
        return Unit(float(self) * factor,self.label.convert(old_unit, new_unit))

    # Returns a copy of this object in exclusively 'cm', 'g', and 's' units.
    def get_base(self, conversions=None):
        ret = copy.deepcopy(self)
        for i, unit in enumerate(ret.label.left):
            factor, new_unit = self.get_conversion_factor(unit, conversions=conversions)
            ret *= factor
            ret.label.left[i] = new_unit
        for i, unit in enumerate(ret.label.right):
            factor, new_unit = self.get_conversion_factor(unit, conversions=conversions)
            ret *= factor
            ret.label.right[i] = new_unit
        return ret

    def __repr__(self):
        string = self.__class__.__name__ + "(%g, %s)"
        return string % (float(self), str(self.label))
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
        starsmashertools.helpers.argumentenforcer.enforcetypes({'other' : [float, int, Unit]})
        if isinstance(other, Unit):
            return Unit(float(self) * float(other), self.label * other.label)
        return Unit(float(self) * other, self.label)

    # other * self = self.__rmul__(other) if other doesn't have __mul__
    def __rmul__(self, other):
        starsmashertools.helpers.argumentenforcer.enforcetypes({'other' : [float, int, Unit]})
        return self.__mul__(other)

    # self / other = self.__truediv__(other)
    def __truediv__(self, other):
        starsmashertools.helpers.argumentenforcer.enforcetypes({'other' : [float, int, Unit]})
        if isinstance(other, Unit):
            return Unit(float(self) / float(other), self.label / other.label)
        return Unit(float(self) / other, self.label)

    # other / self = self.__rtruediv__(other) if other doesn't have __truediv__
    def __rtruediv__(self, other):
        starsmashertools.helpers.argumentenforcer.enforcetypes({'other' : [float, int, Unit]})
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
        #    1. A Label has both a 'long' form and a 'short' form. For example,
        #       a Label with short form 'erg' has long form 'cm*g*g/s*s'.
        #       Notice that there is only 1 '/' symbol. The '/' symbol cannot be
        #       repeated in a Label and all symbols after it are considered
        #       division.
        #    2. All labels must contain only units 'cm', 'g', 's', or '1' in the
        #       long forms, and can have only operations '*' and '/'. Each of
        #       'cm', 'g', and 's' must be written in that order ('cgs') on
        #       other sides of the '/' symbol if there is one. The value '1' can
        #       only be written on the
        #       left side of the '/' if there are no other symbols there.
        #    3. Only multiplication and division is allowed on Labels.
        #    4. Whenever a long form is changed, it is then simplified. For
        #       example, if Label a has long form 'cm/s' and Label b has 's'
        #       then multiplying a with b gives 'cm*s/s'. Then we count the
        #       number of 'cm' on the left of the '/' sign and compare it with
        #       the number of 'cm' on the right side. If the two are equal then
        #       we remove the first instance of each from either side. Then we
        #       proceed with 'g' and finally 's'.
        #    5. A long form is converted to a short form starting from
        #       left-to-right. For example, a Label with long form
        #       'cm*cm*g*g/cm*s*s' first simplifies the expression to
        #       'cm*cm*g*g/s*s'. Then it checks the abbreviations list in order
        #       to find conversions. For example, the 'erg' abbreviation is
        #       defined as 'cm*g*g/s*s'. Thus we look for the presence of 1 'cm'
        #       and 2 'g' on the left side of the '/' and for 2 's' on the right
        #       side. If we find enough of such symbols then we remove those
        #       symbols and insert 'erg' on the left-most side, 'erg * cm'.

        def __init__(self, value, base=['cm','g','s']):
            self.base = base
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

        def check(self):
            empty = self.long
            symbols = self.base + ['*','/','1']
            for symbol in symbols:
                empty = empty.replace(symbol, '')
            if len(empty) > 0:
                s = ["'%s'" % k for k in symbols]
                if len(symbols) == 1: string = "'%s'" % s[0]
                else:
                    string = ", ".join(s[:-1])
                    if len(symbols) > 2: string += ", "
                    string += " and " + s[-1]
                raise Exception("Unit.Label has a long form which contains characters other than %s: '%s'" % (string, self.long))
        
        @property
        def short(self):
            new_left = copy.deepcopy(self.left)
            new_right = copy.deepcopy(self.right)
            
            # Search for unit conversions and then apply those conversions
            conversions = starsmashertools.preferences.get_default('Units', 'label conversions')
            if conversions is not None:
                for short, values in conversions:
                    short_lhs, short_rhs = Unit.Label.split(short)
                    lhs, rhs = Unit.Label.split(values)

                    had_left = False
                    left = copy.deepcopy(new_left)
                    for l in lhs:
                        if l in left: left.remove(l)
                        else: break
                    else: # No break means success
                        had_left = True
                    if not had_left: continue
                    
                    had_right = False
                    right = copy.deepcopy(new_right)
                    for r in rhs:
                        if r in right: right.remove(r)
                        else: break
                    else: # No break means success
                        had_right = True
                    if not had_right: continue

                    new_left = short_lhs + left
                    new_right = short_rhs + right
            
            return Unit.Label.get_string(new_left, new_right)
            
        @property
        def long(self):
            return Unit.Label.get_string(self.left, self.right)

        @staticmethod
        def get_string(left, right):
            starsmashertools.helpers.argumentenforcer.enforcetypes({
                'left' : [list, tuple],
                'right' : [list, tuple],
            })
            if not left and right:
                string = "1"
            else:
                string = "*".join(left)
            if right: string += "/"+"*".join(right)
            return string

        # Return a copy of this label with changes
        def convert(self, old_unit, new_unit):
            starsmashertools.helpers.argumentenforcer.enforcetypes({
                'old_unit' : [str],
                'new_unit' : [str],
            })
            ret = copy.deepcopy(self)
            for i, val in enumerate(ret.left):
                if val == old_unit: ret.left[i] = new_unit
            for i, val in enumerate(ret.right):
                if val == old_unit: ret.right[i] = new_unit
            return ret
        
        def simplify(self):
            for item in self.base:
                lc, rc = self.left.count(item), self.right.count(item)
                while item in self.left and item in self.right:
                    self.left.remove(item)
                    self.right.remove(item)
                    
                    plc = copy.deepcopy(lc)
                    prc = copy.deepcopy(rc)
                    lc = self.left.count(item)
                    rc = self.right.count(item)
                    if lc == plc and rc == prc: break

        def set(self, string):
            self.left, self.right = Unit.Label.split(string)

            # Break down conversions as needed
            conversions = starsmashertools.preferences.get_default('Units', 'label conversions')
            if conversions is not None:
                for short, value in conversions:
                    lhs, rhs = Unit.Label.split(value)
                    left = copy.deepcopy(self.left)
                    right = copy.deepcopy(self.right)
                    for val in self.left:
                        if val == short:
                            idx = left.index(val)
                            left = left[:idx] + lhs + left[idx+1:]
                            right += rhs
                    for val in self.right:
                        if val == short:
                            idx = right.index(val)
                            right = right[:idx] + rhs + right[idx+1:]
                            left += lhs
                    self.left = left
                    self.right = right
            self.organize()
            self.simplify()
            self.check()
                    
        def organize(self):
            newleft = []
            newright = []
            for item in self.base:
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
                if self.base != other.base:
                    raise Exception("Cannot combine Unit.Labels of different bases: '%s' and '%s'" % (str(self.base), str(other.base)))
                ret = Unit.Label("", self.base)
                ret.left = self.left + other.left
                ret.right = self.right + other.right
            else:
                ret = copy.deepcopy(self)
                for i in range(1, other): ret *= self
            ret.organize()
            ret.simplify()
            ret.check()
            return ret

        def __rmul__(self, other):
            starsmashertools.helpers.argumentenforcer.enforcetypes({'other' : [int, Unit.Label]})
            return self.__mul__(other)
        
        def __truediv__(self, other):
            starsmashertools.helpers.argumentenforcer.enforcetypes({'other' : [Unit.Label]})
            if self.base != other.base:
                raise Exception("Cannot combine Unit.Labels of different bases: '%s' and '%s'" % (str(self.base), str(other.base)))
            ret = Unit.Label("", self.base)
            ret.left = self.left + other.right
            ret.right = self.right + other.left
            ret.organize()
            ret.simplify()
            ret.check()
            return ret
            
        def __rtruediv__(self, other):
            starsmashertools.helpers.argumentenforcer.enforcetypes({'other' : [Unit.Label, int]})
            if isinstance(other, int) and other != 1:
                raise Exception("When dividing an 'int' by a 'Unit.Label', the int must be equal to '1', not '%d'" % other)
            
            ret = Unit.Label("", self.base)
            if isinstance(other, Unit.Label):
                if self.base != other.base:
                    raise Exception("Cannot combine Unit.Labels of different bases: '%s' and '%s'" % (str(self.base), str(other.base)))
                ret.left = self.right + other.left
                ret.right = self.left + other.right
            else:
                if self.left: ret.right = self.left
                if self.right: ret.left = self.right
            ret.organize()
            ret.simplify()
            ret.check()
            return ret

        def __pow__(self, value):
            starsmashertools.helpers.argumentenforcer.enforcetypes({'value' : [int, float]})
            ret = ""
            if isinstance(value, float):
                num, denom = value.as_integer_ratio()

                if denom > 4:
                    raise Exception("Cannot raise Unit.Label '%s' to the power of '%s'. The maximum whole number denominator is 4." % (str(self), str(value)))
                
                ret = 1
                for i in range(num): ret *= self

                for item in self.base:
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
            ret.check()
            return ret
    




# This comes from src/starsmasher.h
gravconst = Unit(6.67390e-08, 'cm*cm*cm/g*s*s')





        

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
        
    @property
    def time(self): return (self.length**3 / (gravconst * self.mass))**0.5

    @property
    def frequency(self): return 1. / self.time

    @property
    def area(self): return self.length * self.length

    @property
    def volume(self): return self.area * self.length
    
    @property
    def energy(self): return gravconst * self.mass * self.mass / self.length

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

    
