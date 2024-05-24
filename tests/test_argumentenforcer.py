import unittest
import starsmashertools.helpers.argumentenforcer
import numpy as np
import itertools
import basetest
import typing

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
    def test_simple(self):
        error = starsmashertools.helpers.argumentenforcer.ArgumentTypeError

        class A: pass
        class B: pass
        class C: pass
        
        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def f(a): pass
        f(A())
        
        
        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def f(a : A): pass
        f(A())
        with self.assertRaises(error): f(B())
        with self.assertRaises(error): f(C())

        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def f(a : A | B): pass
        f(A())
        f(B())
        with self.assertRaises(error): f(C())
        
        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def f(a : A, b : B): pass
        f(A(), B())
        with self.assertRaises(error): f(B(), B())
        with self.assertRaises(error): f(A(), A())
        with self.assertRaises(error): f(B(), A())
        
        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def f(a : A, b : B = B()): pass
        f(A())
        f(A(), b = B())
        with self.assertRaises(error): f(B())
        with self.assertRaises(error): f(A(), b = A())
        with self.assertRaises(error): f(B(), b = A())
        
        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def f(a : A, b : B, c : C = C()): pass
        f(A(), B())
        f(A(), B(), c = C())
        with self.assertRaises(error): f(B(), B())
        with self.assertRaises(error): f(A(), A())
        with self.assertRaises(error): f(B(), A())
        with self.assertRaises(error): f(B(), B(), c = C())
        with self.assertRaises(error): f(A(), A(), c = C())
        with self.assertRaises(error): f(B(), A(), c = C())
        with self.assertRaises(error): f(A(), B(), c = A())
        with self.assertRaises(error): f(A(), B(), c = B())

        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def f(a : A, b : B = B(), c : C = C()): pass
        f(A())
        f(A(), b = B())
        f(A(), c = C())
        f(A(), b = B(), c = C())
        with self.assertRaises(error): f(B())
        with self.assertRaises(error): f(A(), b = A())
        with self.assertRaises(error): f(A(), c = A())
        with self.assertRaises(error): f(A(), b = C())
        with self.assertRaises(error): f(A(), c = B())
        with self.assertRaises(error): f(B(), b = C())
        with self.assertRaises(error): f(B(), b = B())
        with self.assertRaises(error): f(B(), c = B())
        with self.assertRaises(error): f(B(), c = C())
        with self.assertRaises(error): f(A(), b = A(), c = C())
        with self.assertRaises(error): f(A(), b = B(), c = A())
        with self.assertRaises(error): f(A(), b = A(), c = A())
        with self.assertRaises(error): f(A(), b = C(), c = C())
        
    
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
        An error should be raised if a NumPy array argument doesn't contain elements of the correct dtype, where the dtype is one of the accepted types for that argument. For example, "def f(a: int | np.ndarray):". This function should accept both `int` values and `np.ndarray` values.
        """
        error = starsmashertools.helpers.argumentenforcer.ArgumentTypeError
        
        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def f(a : int | np.ndarray): pass
        f(int())
        f(np.array([]))
        f(np.int32())
        with self.assertRaises(error): f(True)
        
        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def f(a : np.ndarray): pass
        f(np.array([]))
        with self.assertRaises(error): f(int())

        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def f(a : np.int32 | np.int16): pass
        f(np.int16())
        f(np.int32())
        with self.assertRaises(error): f(int())
        
        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def f(a : int | float | np.ndarray): pass
        f(int())
        f(float())
        f(np.int32())
        f(np.float16())
        f(np.array([]))

        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def f(a : int | float | list | tuple | np.ndarray): pass
        f(int())
        f(float())
        f(list())
        f(tuple())
        f(np.array([]))

        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def f(a : int = 0): pass
        with self.assertRaises(error): f(np.array([]))

        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def f(a : int | np.ndarray = 0): pass
        f()
        f(a = np.array([]))
        with self.assertRaises(error): f(1.)
        
        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def f(a : int | np.ndarray | type(None) = None): pass
        f()
        f(a=int())
        f(a=np.array([]))
        f(a = None)
        with self.assertRaises(error): f(True)

        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def f(a : list | tuple | np.ndarray | type(None) = None): pass
        f()
        f(list())
        f(tuple())
        f(np.array([]))
        f(a = None)
        with self.assertRaises(error): f(True)
        
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

    def test_typing(self):
        error = starsmashertools.helpers.argumentenforcer.ArgumentTypeError
        
        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def f(a : typing.Callable): pass
        f(lambda:0)
        with self.assertRaises(error): f(0)

        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def f(a : typing.Callable | int): pass
        f(lambda:0)
        f(0)
        with self.assertRaises(error): f(1.)

        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def f(a : typing.Callable | type(None) = None): pass
        f()
        f(a = lambda:0)
        f(a = None)
        with self.assertRaises(error): f(0)


    def test_subclasses(self):
        error = starsmashertools.helpers.argumentenforcer.ArgumentTypeError
        
        class A: pass
        class B(A): pass
        
        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def f(a : A): pass
        f(B())
        
        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def f(a : A | B): pass
        f(A())
        f(B())

        

class TestLoader(unittest.TestLoader, object):
    def getTestCaseNames(self, obj):
        return [
            # Very basic functionality
            'test_simple',
            'test_enforce_basic_values',
            'test_numpy_arrays',
            'test_subclasses',
            
            # More advanced functionality
            'test_individual_types',
            'test_individual_values',
            'test_type_combinations',
        ]

                

if __name__ == "__main__":
    unittest.main(failfast=True, testLoader = TestLoader())
