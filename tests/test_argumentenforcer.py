import unittest
import starsmashertools.helpers.argumentenforcer
import numpy as np
import itertools
import basetest

class CustomType: pass

# https://docs.python.org/3/library/stdtypes.html
types = {
    'type(None)' : 'None',
    # numeric
    'int' : 'int()',
    'float' : 'float()',
    'complex' : 'complex()',
    # boolean
    'bool' : 'bool()',
    # sequence
    'list' : 'list()',
    'tuple' : 'tuple()',
    'range' : 'range(0)',
    # text sequence
    'str' : 'str()',
    # binary sequence
    'bytes' : 'bytes()',
    'bytearray' : 'bytearray()',
    'memoryview' : 'memoryview(bytes())',
    # set
    'set' : 'set()',
    'frozenset' : 'frozenset()',
    # mapping
    'dict' : 'dict()',
    # Custom classes
    'CustomType' : 'CustomType()',
}

# NumPy types
for key, val in np.sctypes.items():
    for v in val:
        try: v()
        except: continue
        if v.__name__ in types.keys(): continue
        # Handle un-silenceable deprecation warnings...
        if v.__name__ in [
                'object',
        ]: continue
        types['np.%s' % v.__name__] = 'np.%s()' % v.__name__

type_combinations = list(itertools.combinations(types.keys(), 3))





class TestArgumentEnforcer(basetest.BaseTest):
    def test_individual_types(self):
        """
        Test all the types individually
        """
        for _type, value in types.items():
            string = """
@starsmashertools.helpers.argumentenforcer.enforcetypes
def f(argument : {_type}): return None
f({value})
            """.format(_type=_type, value=value)
            exec(string)

    def test_type_combinations(self):
        """
        Test all the types in combinations of length 3
        """
        for combo in type_combinations:
            string = ["@starsmashertools.helpers.argumentenforcer.enforcetypes"]
            string += ['def f(%s): return None' % ", ".join(["argument"+str(i)+" : {}" for i in range(len(combo))])]
            string += ['f(%s)' % ", ".join([types[c] for c in combo])]
            string = "\n".join(string)
            inputs = [c for c in combo]
            inputs += [types[c] for c in combo]
            string = string.format(*inputs)
            exec(string)

    def test_individual_values(self):
        for _type, value in types.items():
            string = """
a = {value}
starsmashertools.helpers.argumentenforcer.enforcetypes({{'a' : [{_type}]}})
            """.format(_type = _type, value= value)
            exec(string)

    def test_numpy_arrays(self):
        """
        An error should be raised if a NumPy array argument doesn't contain elements of the correct dtype, where the dtype is one of the accepted types for that argument. For example, "def f(a: int | np.ndarray):". This function should accept both `int` values and `np.array` values, so long as the dtype of the array is `int`.
        """
        error = starsmashertools.helpers.argumentenforcer.ArgumentTypeError
        
        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def f(a : int | np.ndarray): pass
        f(int())
        f(np.array([], dtype=int))
        with self.assertRaises(error):
            f(np.int32())
            f(np.array([np.int32()], dtype=np.int32))
            f(np.array([float()]))
            f(True)

        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def f(a : np.ndarray): pass
        f(np.array([]))
        with self.assertRaises(error): f(int())

        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def f(a : np.int32 | np.int16 | np.ndarray): pass
        f(np.int16())
        f(np.int32())
        with self.assertRaises(error):
            f(int())
            f(np.array([float()]))

        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def f(a : int | float | np.ndarray): pass
        f(int())
        f(float())
        f(np.int32())
        f(np.float16())
        #with self.assertRaises(error):
        #    f(np.int32())
        #    f(np.float16())

        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def f(a : int | float | list | tuple | np.ndarray): pass
        f(list())
        f(tuple())
        f(int())
        f(float())
        f(np.array([float()]))
        f(np.array([int()]))
        f(np.int32())
        f(np.float64())
        f(np.array([int()], dtype=np.int32))
        f(np.array([float()], dtype=np.float64))
        with self.assertRaises(error):
            f(np.array([[]], dtype=list))

        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def f(a : int = 0): pass
        with self.assertRaises(error): f(np.array([float()]))

        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def f(a : int | np.ndarray = 0): pass
        f()
        f(a=np.array([int()]))
        with self.assertRaises(error): f(np.array([float()]))

        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def f(a : int | np.ndarray | type(None) = None): pass
        f()
        f(a=int())
        f(a=np.array([int()]))
        with self.assertRaises(error): f(np.array([float()]))

        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def f(a : list | tuple | np.ndarray | type(None) = None): pass
        f()
        f(list())
        f(tuple())
        f(np.array([]))
        with self.assertRaises(error): f(True)

        # Test custom types with NumPy arrays
        class OtherType: pass
        
        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def f(a : list | tuple | CustomType | np.ndarray): pass
        f(list())
        f(tuple())
        f(np.array([]))
        f(np.asarray([CustomType()], dtype = object))
        f(np.asarray([CustomType()], dtype = CustomType))
        f(np.asarray([], dtype = object))
        f(np.asarray([], dtype = CustomType))
        with self.assertRaises(error): f(OtherType())
        with self.assertRaises(error):
            f(np.asarray([OtherType()], dtype = object))
        with self.assertRaises(error):
            f(np.asarray([OtherType()], dtype = OtherType))
        with self.assertRaises(error):
            f(np.asarray([CustomType(), OtherType()], dtype = object))
        
    # This is just a basic test for now. I'm not sure how to automate the task
    # across all combinations of types
    def test_enforce_basic_values(self):
        thing = int()
        starsmashertools.helpers.argumentenforcer.enforcetypes(
            { 'thing' : [int] },
        )
        starsmashertools.helpers.argumentenforcer.enforcetypes(
            { 'thing' : [str, int] },
        )
        starsmashertools.helpers.argumentenforcer.enforcetypes(
            { 'thing' : [str, bool, float, int] },
        )

        type_err = starsmashertools.helpers.argumentenforcer.ArgumentTypeError
        for _type in [str, float, bool]:
            with self.assertRaises(type_err):
                starsmashertools.helpers.argumentenforcer.enforcetypes({
                    'thing' : [_type],
                })

        val_err = starsmashertools.helpers.argumentenforcer.ArgumentValueError
        for _value in [[int()], [int(), int()], [float()], [bool()]]:
            def func(value):
                starsmashertools.helpers.argumentenforcer.enforcevalues({
                    'value' : _value,
                })
            with self.assertRaises(val_err):
                func(None)

if __name__ == "__main__":
    unittest.main(failfast=True)
