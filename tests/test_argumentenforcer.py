import unittest
import starsmashertools.helpers.argumentenforcer

class TestArgumentEnforcer(unittest.TestCase):
    def test_enforcetypes(self):

        @starsmashertools.helpers.argumentenforcer.enforcetypes
        def func(
                string : str,
                integer : int,
                number : float,
                boolean : bool,
        ):
            return None

        self.assertRaises(
            starsmashertools.helpers.argumentenforcer.ArgumentTypeError,
            func,
            (0, 0, 1.0, True),
        )
        self.assertRaises(
            starsmashertools.helpers.argumentenforcer.ArgumentTypeError,
            func,
            ('', '', 1.0, True),
        )
        self.assertRaises(
            starsmashertools.helpers.argumentenforcer.ArgumentTypeError,
            func,
            ('', 0, '', True),
        )
        self.assertRaises(
            starsmashertools.helpers.argumentenforcer.ArgumentTypeError,
            func,
            ('', 0, 1.0, ''),
        )

    def test_enforcevalues(self):
        thing = 0
        starsmashertools.helpers.argumentenforcer.enforcetypes(
            { 'thing' : [int] },
        )
        starsmashertools.helpers.argumentenforcer.enforcetypes(
            { 'thing' : [str, int] },
        )
        starsmashertools.helpers.argumentenforcer.enforcetypes(
            { 'thing' : [str, bool, float, int] },
        )

        for _type in [str, float, bool]:
            self.assertRaises(
                starsmashertools.helpers.argumentenforcer.ArgumentTypeError,
                starsmashertools.helpers.argumentenforcer.enforcetypes,
                ( { 'thing' : [_type] } ),
            )
        
        for _value in [[0], [0, 1], [1.0], [False]]:
            def func(value):
                starsmashertools.helpers.argumentenforcer.enforcevalues({
                    'value' : _value,
                })
            self.assertRaises(
                starsmashertools.helpers.argumentenforcer.ArgumentValueError,
                func,
                (None),
            )
        

if __name__ == "__main__":
    unittest.main()
