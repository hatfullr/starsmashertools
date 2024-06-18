# See the bottom of this file for definitions of constants.

import starsmashertools.preferences
import starsmashertools.helpers.readonlydict
import starsmashertools.helpers.argumentenforcer
import starsmashertools.helpers.string
from starsmashertools.helpers.apidecorator import api
import numpy as np
import re
import fractions
import copy
import operator

def get_all_labels():
    conversions = Unit.conversions
    labels = []
    for conversion in conversions:
        for key, val in conversion.items():
            if isinstance(val, str):
                if val not in labels: labels += [val]
            elif isinstance(val, (list, tuple)):
                for l, v in val:
                    if l in labels: continue
                    labels += [l]
            else:
                raise NotImplementedError("Unrecognized type in 'unit conversions' in preferences.py: '%s'" % type(val).__name__)
    return labels

@starsmashertools.preferences.use
class Unit(object):
    """
    To use string formatters with a Unit, you can either specify the usual
    formatting as integer or float types, i.e., ``'{:5.3f}'``\, or you can
    specify a formatter for both the value and the label, i.e. ``'{:5.3f 6s}'``
    (see examples below). Any whitespace between the value and label formatters 
    is discarded during processing.
    
    Example
    -------
        
    .. code-block:: python

        import starsmashertools.lib.units
        u = starsmashertools.lib.units.Unit(1.234567, 'day')
    
        # Values only:
        print('{:f}'.format(u))
        # '1.234567'
        print('{:10f}'.format(u))
        # '  1.234567'    
        print('{:10.2f}'.format(u))
        # '      1.23'

        # Values and labels
        print('{:fs}'.format(u))
        # '1.234567day'
        print('{:f4s}'.format(u))
        # '1.234567day '
        print('{:f 4s}'.format(u))
        # '1.234567day '
        print('{:f>4s}'.format(u))
        # '1.234567 day'
        print('{:f >4s}'.format(u))
        # '1.234567 day'
        print('{:7.3f >4s}'.format(u))
        # '  1.235 day'

    """
    _format_re = re.compile(r'[^[a-zA-Z]*[a-zA-Z]{1}')
    
    conversions = None
    
    class InvalidLabelError(Exception, object): pass
    class InvalidTypeConversionError(Exception, object): pass
    
    @api
    def __init__(self, *args, base = ['cm', 'g', 's', 'K']):
        if len(args) == 1 and isinstance(args[0], str):
            # Convert string to Unit
            string = args[0]
            string = string.replace('Unit', '').replace('(','').replace(')','')
            try:
                value, label = string.split(',')
                value = starsmashertools.helpers.string.parse(value.strip())
                label = label.strip().replace('"','').replace("'",'')
            except Exception as e:
                raise Unit.InvalidTypeConversionError("Invalid str to Unit conversion: '%s'" % string) from e
        elif len(args) == 2:
            value, label = args
        else: raise ValueError("Unit takes at most 2 positional arguments, not %d" % len(args))

        if not isinstance(value, (float, int)):
            raise TypeError(value)
        if not isinstance(label, (str, Unit.Label)):
            raise TypeError(label)
        
        #starsmashertools.helpers.argumentenforcer.enforcetypes({
        #    'value' : [float, int],
        #    'label' : [str, Unit.Label],
        #})

        self.base = base
        if isinstance(label, str):
            self.label = Unit.Label(label, base = self.base)
        else: self.label = label
        self.value = value
    
    @staticmethod
    def get_conversions():
        if Unit.conversions is None:
            Unit.conversions = []
            for key, c in Unit.preferences.get('conversions').items():
                value, base = c.split()
                value = float(value)

                for obj in Unit.conversions:
                    if obj['base'] == base:
                        obj['conversions'] += [[key, value]]
                        break
                else:
                    Unit.conversions += [{
                        'base' : base,
                        'conversions' : [[key, value]],
                    }]
        return Unit.conversions
    
    
    # Decide on units that give the cleanest value
    @api
    def auto(self, threshold=100, conversions = None):
        import starsmashertools
        
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'threshold' : [float, int],
        })
        if threshold <= 0: raise ValueError("Argument 'threshold' must be > 0, not '%s'" % str(threshold))
        
        # Get all the available conversions
        if conversions is None:
            conversions = Unit.get_conversions()
            if not conversions: return self
        
        base = self.get_base(conversions = conversions)
        if float(base) < threshold: return base

        # Get a list of all the possible conversions
        label = base.label.left + base.label.right
        possible_results = []
        for conversion in conversions:
            if conversion['base'] not in label: continue
            for name, value in conversion['conversions']:
                # Replace all the instances of the base unit with this new
                # converted unit
                new_left = base.label.left.copy()
                new_right = base.label.right.copy()
                for i, item in enumerate(new_left):
                    if item == conversion['base']: new_left[i] = name
                for i, item in enumerate(new_right):
                    if item == conversion['base']: new_right[i] = name
                l = Unit.Label._get_string(new_left, new_right)
                possible_results += [base.convert(l, conversions=conversions)]
        
        possible_results = sorted(
            possible_results,
            key=lambda x: float('inf') if x.value > threshold else threshold - x.value,
        )
        return possible_results[0]

    @api
    def get_conversion_factor(self, new_label, conversions = None):
        # Convert this Unit into a compatible new Unit
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'new_label' : [str, Unit.Label],
        })
        if isinstance(new_label, str): new_label = Unit.Label(
                new_label, base = self.label.base
        )

        # Quickly make sure that the labels make sense. Otherwise, zip won't
        # work as we expect.
        #assert len(self.label.left) == len(new_label.left)
        #assert len(self.label.right) == len(new_label.right)
        
        if conversions is None:
            conversions = Unit.get_conversions()
            if not conversions: conversions = None
        
        # In the comments we consider the case of converting '10 km/hr' to
        # 'm/min', where our current unit is 'km/hr' and the new_label='m/min'.
        # The correct answer is '166.666667 m/min'.

        # Convert from '10 km/hr' to '277.7778 cm/s'
        base = self.get_base(conversions = conversions)
        
        # Search the conversions to get from '277.7778 cm/s' to '166.6667 m/min'
        # First we work with the left-side labels and then we work with the
        # right-side labels
        factor = 1.
        
        for conversion in conversions:
            for i, (short, value) in enumerate(conversion['conversions']):
                for item, new_item in zip(self.label.left, new_label.left):
                    if (item, new_item) == (short, short): continue
                    if item == short: factor *= value
                    if new_item == short: factor /= value
                for item, new_item in zip(self.label.right, new_label.right):
                    if (item, new_item) == (short, short): continue
                    if item == short: factor /= value
                    if new_item == short: factor *= value
        return factor
    
    @api
    def convert(
            self,
            new : str | type(None) = None,
            to : list | tuple | type(None) = None,
            **kwargs
    ):
        """
        Return a copy of this :class:`~.Unit` converted to a new unit.
        
        Other Parameters
        ----------
        new : str, :class:`~.Unit.Label`\, :class:`~.Unit`\, None, default = None
            If a :py:class:`str` is given it will be converted to a 
            :class:`~.Unit.Label`\. If a :class:`~.Unit` is given, it must have
            the same dimensions as this Unit. If `None`\, you must specify 
            keyword ``to`` (see below).
        
        to : list, tuple, :class:`~.Units`\, None, default = None
            An iterable of :py:class:`str` unit labels. For each given string, 
            the :class:`~.Unit.Label` is searched for matches. Then, each match
            is converted to the base units (cgs) before being converted into the
            unit of the given string. If `None`\, you must specify keyword 
            ``new`` (see above).
        
        **kwargs
            Keywords arguments are passed directly to 
            :meth:`~.Unit.get_conversion_factor`\.
        
        Returns
        -------
        :class:`~.Unit`
            The newly converted unit.
        
        Examples
        --------
        This example converts 1 cm/s to 1 km/hr::
        
            >>> unit = Unit(1, 'cm/s')
            >>> unit.convert('km/hr')
            Unit(0.036, km/hr)
        
        
        See Also
        --------
        :meth:`~.get_conversion_factor`
        """
        import starsmashertools
        
        if to is None:
            starsmashertools.helpers.argumentenforcer.enforcetypes({
                'new' : [str, Unit.Label, Unit],
            })
        elif new is None:
            starsmashertools.helpers.argumentenforcer.enforcetypes({
                'to' : [list, tuple],
            })
            for t in to:
                if not isinstance(t, str):
                    raise TypeError("Keyword argument 'to' must contain only str types. Received '%s' type." % type(t).__name__)
        else:
            raise TypeError("One of keywords 'new' or 'to' must be None")
        if to is None and new is None:
            raise TypeError("One of keywords 'new' or 'to' must not be None")
        
        
        
        if new is None:
            base = self.get_base()
            new = copy.deepcopy(base.label)
            conversions = Unit.get_conversions()
            all_bases = [c['base'] for c in conversions]
            all_names = [[c[0] for c in conversion['conversions']] for conversion in conversions]
            
            for conversion in conversions:
                # If the base label doesn't contain this base name ('cm', 'g',
                # 's', etc.), then we can skip it.
                bname = conversion['base']
                if bname not in base.label.left + base.label.right: continue

                for name, value in conversion['conversions']:
                    if name not in to: continue

                    # We get here if the conversion name is included in the "to"
                    # list. Now we need to replace the corresponding base name
                    # in the new label with the value in the "to" list.
                    for i, item in enumerate(new.left):
                        if item != bname: continue
                        new.left[i] = name
                    for i, item in enumerate(new.right):
                        if item != bname: continue
                        new.right[i] = name
        
        if isinstance(new, str): new = Unit.Label(new, base = self.label.base)
        if isinstance(new, Unit): label = new.label
        else:
            label = new
        if not self.label.is_compatible(label):
            raise Unit.InvalidLabelError("Cannot convert unit because labels '%s' and '%s' are incompatible" % (self.label, label))
        factor = self.get_conversion_factor(label, **kwargs)
        if isinstance(new, Unit): factor /= new.value
        
        for conversion in Unit.get_conversions():
            if not ('/' in conversion['base'] or '*' in conversion['base']):
                continue
            b = Unit.Label(conversion['base'], base = label.base)
            change_made = False
            for short, _ in conversion['conversions']:
                for item in label.left:
                    if item != short: continue
                    label.left.remove(item)
                    label.left += b.left
                    label.right += b.right
                    change_made = True
                for item in label.right:
                    if item != short: continue
                    label.right.remove(item)
                    label.left += b.left
                    label.right += b.right
                    change_made = True
            #if change_made:
            #    print("HELLO")
                #label = Unit.Label(label.long, base = label.base)
            #    print(label.long)
        
        return Unit(self.value * factor, label)
                    
                

    # Returns new Unit in the base units
    @api
    def get_base(self, conversions = None):
        if conversions is None:
            conversions = Unit.get_conversions()
        
        ret = copy.deepcopy(self)
        for conversion in conversions:
            base_label = Unit.Label(conversion['base'], base = self.label.base)
            conversion_names, conversion_values = [], []
            for name, value in conversion['conversions']:
                conversion_names += [name]
                conversion_values += [value]
            for i, name in enumerate(ret.label.left):
                if name in self.base: continue
                if name in conversion_names:
                    ret.value *= conversion_values[conversion_names.index(name)]
                    ret.label.left[i] = conversion['base']
            for i, name in enumerate(ret.label.right):
                if name in self.base: continue
                if name in conversion_names:
                    ret.value /= conversion_values[conversion_names.index(name)]
                    ret.label.right[i] = conversion['base']

        if ret.label.left and not ret.label.right:
            ret.label = Unit.Label('*'.join(ret.label.left), base = self.label.base)
        elif not ret.label.left and ret.label.right:
            ret.label = Unit.Label('1/' + '*'.join(ret.label.right), base = self.label.base)
        elif ret.label.left and ret.label.right:
            ret.label = Unit.Label('*'.join(ret.label.left) + '/' + '*'.join(ret.label.right), base = self.label.base)

        return ret

    def sqrt(self, *args, **kwargs):
        return Unit(self.value**0.5, self.label**0.5)

    def __repr__(self): return self.__class__.__name__ + '(' + str(self) + ')'
    def __str__(self): return '{:g}'.format(self.value) + ' ' + str(self.label)
    def __format__(self, format_spec):
        matches = Unit._format_re.findall(format_spec)
        if not matches: return self.value.__format__(format_spec) + str(self.label)
        if len(matches) == 1: return self.value.__format__(format_spec)
        return self.value.__format__(matches[0]) + str(self.label).__format__(matches[1].lstrip())
    
    @api
    def __eq__(self, other):
        if isinstance(other, Unit):
            if not self.label.is_compatible(other.label): return False
            #return self.value == other.convert(self.label).value
            base = self.get_base()
            other_base = other.get_base()
            if base.label == other_base.label:
                return base.value == other_base.value
            return self.value == other.value and self.label == other.label
        return self.value == other
    @api
    def __gt__(self, other):
        if isinstance(other, Unit):
            base = self.get_base()
            other_base = other.get_base()
            if base.label != other_base.label: raise Unit.InvalidLabelError
            return base.value > other_base.value
        return self.value > other
    @api
    def __ge__(self, other):
        if isinstance(other, Unit):
            base = self.get_base()
            other_base = other.get_base()
            if base.label != other_base.label: raise Unit.InvalidLabelError
            return base.value >= other_base.value
        return self.value >= other
    @api
    def __lt__(self, other):
        if isinstance(other, Unit):
            base = self.get_base()
            other_base = other.get_base()
            if base.label != other_base.label: raise Unit.InvalidLabelError
            return base.value < other_base.value
        return self.value < other
    @api
    def __le__(self, other):
        if isinstance(other, Unit):
            base = self.get_base()
            other_base = other.get_base()
            if base.label != other_base.label: raise Unit.InvalidLabelError
            return base.value <= other_base.value
        return self.value <= other
    
    def __reduce__(self):
        # https://docs.python.org/3/library/pickle.html#object.__reduce__
        return (Unit, (self.value, self.label))
    def __reduce_ex__(self, *args, **kwargs):
        return self.__reduce__()

    # Note that __float__ and __int__ do not convert the Unit to its base before
    # returning. This is the most likely expected outcome.
    @api
    def __float__(self): return float(self.value)
    @api
    def __int__(self): return int(self.value)
    
    # self * other = self.__mul__(other)
    @api
    def __mul__(self, other):
        if isinstance(other, Unit):
            base = self.get_base()
            other_base = other.get_base()
            if base.label == other_base.label:
                return Unit(base.value * other_base.value, base.label**2).convert(self.label**2)
            return Unit(self.value * other.value, self.label * other.label)
        return Unit(self.value * other, self.label)

    @api
    def __rmul__(self, other): return Unit(other * self.value, self.label)

    # self / other = self.__truediv__(other)
    @api
    def __truediv__(self, other):
        if isinstance(other, Unit):
            base = self.get_base()
            other_base = other.get_base()
            if base.label == other_base.label:
                # Convert down to the base units. Result is dimensionless
                return base.value / other_base.value
            return Unit(self.value / other.value, self.label / other.label)
        return Unit(self.value / other, self.label)

    # other / self = self.__rtruediv__(other)
    @api
    def __rtruediv__(self, other):
        return Unit(other / self.value, 1 / self.label)
    
    @api
    def __add__(self, other):
        if isinstance(other, Unit):
            base = self.get_base()
            other_base = other.get_base()
            if base.label != other_base.label: raise Unit.InvalidLabelError
            return Unit(base.value + other_base.value, base.label).convert(self.label)
        return Unit(self.value + other, self.label)
    @api
    def __radd__(self, other): return Unit(other + self.value, self.label)
    @api
    def __sub__(self, other):
        if isinstance(other, Unit):
            base = self.get_base()
            other_base = other.get_base()
            if base.label != other_base.label: raise Unit.InvalidLabelError
            return Unit(base.value - other_base.value, base.label).convert(self.label)
        return Unit(self.value - other, self.label)
    @api
    def __rsub__(self, other): return Unit(other - self.value, self.label)
    @api
    def __pow__(self, value): return Unit(self.value**value, self.label**value)

    @api
    def __abs__(self): return Unit(abs(self.value), self.label)
    @api
    def __neg__(self): return Unit(-self.value, self.label)
    @api
    def __ceil__(self): return Unit(self.value.__ceil__(), self.label)
    @api
    def __floor__(self): return Unit(self.value.__floor__(), self.label)
    @api
    def __floordiv__(self, other): return (self / other).__floor__()
    @api
    def __mod__(self, other): return self - (self / other).__floor__() * other
    @api
    def __pos__(self): return self
    @api
    def __rfloordiv__(self, other): return (other / self).__floor__()
    @api
    def __rmod__(self, other):
        if isinstance(other, Unit):
            return other - (other / self).__floor__() * self
        return Unit((other - (other / self).__floor__() * self).value, self.label)
    @api
    def __round__(self, *args, **kwargs): return Unit(self.value.__round__(*args, **kwargs), self.label)
    @api
    def __trunc__(self): return Unit(self.value.__trunc__(), self.label)

    @api
    def __index__(self): return self.value.__index__()

    @starsmashertools.preferences.use
    class Label(object):
        # Principles:
        #    1. A Label has both a 'long' form and a 'short' form. For example,
        #       a Label with short form 'erg' has long form 'cm*g*g/s*s'.
        #       Notice that there is only 1 '/' symbol. The '/' symbol cannot be
        #       repeated in a Label and all symbols after it are considered
        #       division.
        #    2. All labels must contain only units 'cm', 'g', 's', or '1' in the
        #       long forms, and can have only operations '*' and '/'. The value
        #       '1' can only be written on the left side of the '/' if there are
        #       no other symbols there.
        #    3. Only *, /, and ** are allowed on Labels.
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

        conversions = None

        def __init__(
                self,
                value : str | list | tuple,
                base = ['cm','g','s','K'],
        ):
            self.base = base
            if isinstance(value, str):
                self.left = []
                self.right = []
                self.set(value)
            elif isinstance(value, (list, tuple)):
                self.left, self.right = value

        def __deepcopy__(self, *args, **kwargs):
            return Unit.Label(self.long, base = self.base)
                
        @property
        def isCompound(self):
            # "Compound" means this Label contains more than a single value,
            # such as 'cm/s' for example, or '1/s'.
            return ((len(self.left) == 1 and not self.right) or
                    len(self.left) == 0 and len(self.right) == 1)

        @staticmethod
        def split(string):
            # the string could contain '/' or not
            # if it does, either sides on the '/' could have '*'
            # if they do, split on '*' on each side
            # otherwise, check the left side for '1'
            # if it has '1', return the left side as []
            if '/' in string:
                lhs, rhs = string.split('/')
                return [] if lhs == '1' else lhs.split('*'), rhs.split('*')
            else:
                return [] if string == '1' else string.split('*'), []

        @staticmethod
        def get_conversions():
            if Unit.Label.conversions is None:
                Unit.Label.conversions = [
                    [val, key] for key,val in Unit.Label.preferences.get('conversions').items()
                ]
            yield from Unit.Label.conversions
        
        @property
        def short(self):
            new_left = self.left.copy()
            new_right = self.right.copy()
            
            # Search for unit conversions and then apply those conversions
            for long, value in Unit.Label.get_conversions():
                long_lhs, long_rhs = Unit.Label.split(long)
                lhs, rhs = Unit.Label.split(value)

                had_left = False
                left = new_left.copy()
                for l in lhs:
                    if l in left: left.remove(l)
                    else: break
                else: # No break means success
                    had_left = True
                if not had_left: continue

                had_right = False
                right = new_right.copy()
                for r in rhs:
                    if r in right: right.remove(r)
                    else: break
                else: # No break means success
                    had_right = True
                if not had_right: continue

                new_left = long_lhs + left
                new_right = long_rhs + right
            
            return Unit.Label._get_string(new_left, new_right)

        @property
        def long(self):
            return Unit.Label._get_string(self.left, self.right)
        
        @staticmethod
        def _get_string(left, right):
            if left:
                if right: return '*'.join(left) + '/' + '*'.join(right)
                return '*'.join(left)
            if right: return '1/' + '*'.join(right)
            return ''

        def get_base_string(self):
            """ Return this label's string with all units converted down to the
            base units. """
            if Unit.get_conversions() is None: return self.long

            #if isinstance(self.left, str): newleft = [self.left]
            #else: newleft = self.left.copy()
            #if isinstance(self.right, str): newright = [self.right]
            #else: newright = self.right.copy()

            newleft = self.left.copy()
            newright = self.right.copy()
            
            for conversion in Unit.get_conversions():
                base = conversion['base']
                
                for name, _ in conversion['conversions']:
                    # This is as fast as we can go
                    newleft[:] = [base if item == name else item for item in newleft]
                    newright[:] = [base if item == name else item for item in newright]

            ret = Unit.Label._get_string(newleft, newright)
            for value, long in Unit.Label.get_conversions():
                if ret != value: continue
                ret = long
                break
            return ret
        
        def is_compatible(self, other):
            """
            Returns `True` if this Label can be safely converted to the other
            Label.
            """
            if not isinstance(other, (str, Unit.Label)): return False
            if isinstance(other, str): other = Unit.Label(other, base = self.base)

            #print(self.get_base_string(), other.get_base_string())
            
            # Comparing the labels as they are in their base forms tells us if
            # they are compatible, regardless of how they have been converted.
            return self.get_base_string() == other.get_base_string()

        # Return a copy of this label with changes
        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def convert(self, old_label : str, new_label : str):
            return Unit.Label(
                (
                    [new_label if val == old_label else val for val in self.left],
                    [new_label if val == old_label else val for val in self.right],
                 ),
            )

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
            for short, long in Unit.Label.get_conversions():
                lhs, rhs = Unit.Label.split(long)
                # str are immutable, so this will create a copy
                left = self.left
                right = self.right
                if not isinstance(left, str): left = left.copy()
                if not isinstance(right, str): right = right.copy()
                
                for val in self.left:
                    if val != short: continue
                    idx = left.index(val)
                    left = left[:idx] + lhs + left[idx+1:]
                    right += rhs
                for val in self.right:
                    if val != short: continue
                    idx = right.index(val)
                    right = right[:idx] + rhs + right[idx+1:]
                    left += lhs
                self.left = left
                self.right = right
            
            self.organize()
            self.simplify()

        def organize(self):
            # Sort the left and right arrays starting with the base values first
            # and then any other values after.
            self.left = sorted(
                [b for b in self.left if b in self.base],
                key = lambda x: self.base.index(x),
            ) + sorted(
                [b for b in self.left if b not in self.base],
            )
            self.right = sorted(
                [b for b in self.right if b in self.base],
                key = lambda x: self.base.index(x),
            ) + sorted(
                [b for b in self.right if b not in self.base],
            )
        
        def __str__(self): return self.short
        def __repr__(self): return self.long

        def __eq__(self, other):
            if isinstance(other, Unit.Label):
                return self.long == other.long
            elif isinstance(other, str):
                return self.long == other
            return False

        def __mul__(self, other):
            starsmashertools.helpers.argumentenforcer.enforcetypes(
                {'other' : [Unit.Label, int]}
            )
            ret = copy.deepcopy(self)
            if isinstance(other, Unit.Label):
                if self.base != other.base:
                    raise Exception("Cannot combine Unit.Labels of different bases: '%s' and '%s'" % (str(self.base), str(other.base)))
                ret.left += other.left
                ret.right += other.right
            else:
                ret.left = ret.left * other
                ret.right = ret.right * other
            ret.organize()
            ret.simplify()
            return ret

        def __rmul__(self, *args, **kwargs):
            return self.__mul__(*args, **kwargs)

        def __truediv__(self, other):
            starsmashertools.helpers.argumentenforcer.enforcetypes(
                {'other' : [Unit.Label, str]}
            )
            if self.base != other.base:
                raise Exception("Cannot combine Unit.Labels of different bases: '%s' and '%s'" % (str(self.base), str(other.base)))
            ret = copy.deepcopy(self)

            ret.left += other.right
            ret.right += other.left
            
            ret.organize()
            ret.simplify()
            return ret

        def __rtruediv__(self, other):
            starsmashertools.helpers.argumentenforcer.enforcetypes(
                {'other' : [Unit.Label, int]}
            )
            if isinstance(other, int) and other != 1:
                raise Exception("When dividing an 'int' by a 'Unit.Label', the int must be equal to '1', not '%d'" % other)

            ret = copy.deepcopy(self)
            if isinstance(other, Unit.Label):
                if self.base != other.base:
                    raise Exception("Cannot combine Unit.Labels of different bases: '%s' and '%s'" % (str(self.base), str(other.base)))
                ret.left += other.left
                ret.right += other.right
            else:
                right = ret.right.copy()
                ret.right = ret.left.copy()
                ret.left = right
            ret.organize()
            ret.simplify()
            return ret
        
        def __pow__(self, value):
            starsmashertools.helpers.argumentenforcer.enforcetypes(
                {'value' : [int, float]}
            )
            ret = ""
            if isinstance(value, float):
                frac = fractions.Fraction(value).limit_denominator(16)
                num, denom = frac.numerator, frac.denominator

                if round(abs(num/denom - value), 7) > 0:
                    raise Exception("(%s)**%s exponent has reduced fraction of %d/%d = %g. Consider changing the precision of the exponent." % (str(self), str(value), num, denom, num/denom))
                
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
            return ret

        def __reduce__(self):
            return (self.__class__, (self.long, self.base))









        
@starsmashertools.preferences.use
class Units(starsmashertools.helpers.readonlydict.ReadOnlyDict, object):
    """
    This class represents the units which a :class:`~.lib.simulation.Simulation`
    uses, as set in the StarSmasher code.
    """
    
    @api
    def __init__(self, simulation):
        import starsmashertools
        import starsmashertools.lib.simulation
        
        # Make sure the given simulation argument is of the right type
        starsmashertools.helpers.argumentenforcer.enforcetypes({
            'simulation' : [str, starsmashertools.lib.simulation.Simulation],
        })
        if isinstance(simulation, str):
            simulation = starsmashertools.get_simulation(simulation)
        
        self.simulation = simulation
        self.constants = copy.deepcopy(constants)
        
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
            'temperatures' : self.temperature,
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
        
        for key, val in self.preferences.get('extra').items():
            if isinstance(val, (float, int)):
                obj[key] = val
            elif isinstance(val, str):
                obj[key] = eval(val, {}, _locals)
            else:
                raise NotImplementedError
        
        super(Units, self).__init__(obj)

    # Pickling stuff:
    def __reduce__(self):
        # https://docs.python.org/3/library/pickle.html#object.__reduce__
        return (Units, (self.simulation.directory,),self.__getstate__(),)
    
    def __setstate__(self, state):
        self.__init__(state['simulation directory'])
        self.constants = state['constants']
    
    def __getstate__(self):
        return {
            'simulation directory' : self.simulation.directory,
            'constants' : self.constants,
        }
    
    # Simulation units
    @property
    def length(self): return Unit(self.simulation['runit'], 'cm')

    @property
    def mass(self): return Unit(self.simulation['munit'], 'g')

    @property
    def temperature(self): return Unit(1., 'K')
        
    @property
    def time(self): return (self.length**3 / (self.constants['G'] * self.mass))**0.5
    
    @property
    def frequency(self): return 1. / self.time

    @property
    def area(self): return self.length * self.length

    @property
    def volume(self): return self.area * self.length
    
    @property
    def energy(self): return self.constants['G'] * self.mass * self.mass / self.length

    @property
    def velocity(self): return self.length / self.time
    
    @property
    def acceleration(self): return self.velocity / self.time

    @property
    def force(self): return self.acceleration * self.mass

    @property
    def pressure(self): return self.force / self.area

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




# A few fundamental physical constants are defind here. Derived constants which
# rely on these constants are defined in the Units class.
#
# NOTE: Each Units class has its own version of this dictionary, automatically
#       converted to cgs where needed.
#

constants = {
    # The following are from NIST Database 121. Last Update to Data Content: May
    # 2019. https://physics.nist.gov/cuu/Constants/index.html
    
    # Gravitational constant
    'G' : Unit(
        6.67430e-8,      # Standard uncertainty = 0.00015 x 10-11 m^3 kg^-1 s^-2
        'cm*cm*cm/g*s*s',
    ),
    
    # Planck constant
    'planck' : Unit(
        6.62607015e-27,  # Standard uncertainty = (exact)
        'cm*cm*g/s',     # erg*s
    ),
    
    # Speed of light
    'c' : Unit(
        2.99792458e10,   # Standard uncertainty = (exact)
        'cm/s',
    ),
    
    # Boltzmann constant
    'boltzmann' : Unit(
        1.380649e-16,    # Standard uncertainty = (exact)
        'cm*cm*g/s*s*K', # erg/K
    ),

    # The following is from IAU 2015 Resolution B3 (10.48550/arXiv.1510.07674)
    'Lsun' : Unit(
        3.828e33,
        'cm*cm*g/s*s*s', # erg/s
    ),
}
constants['sigmaSB'] = 2 * np.pi**5 * constants['boltzmann']**4 / \
    (15 * constants['c']**2 * constants['planck']**3)
constants['a'] = 4 * constants['sigmaSB'] / constants['c']

