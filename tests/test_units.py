import starsmashertools.lib.units
import starsmashertools
import starsmashertools.helpers.argumentenforcer
import unittest

class TestUnits(unittest.TestCase):
    def test(self):
        simulation = starsmashertools.get_simulation("data")
        units = starsmashertools.lib.units.Units(simulation)
        
        self.assertRaises(
            starsmashertools.helpers.argumentenforcer.ArgumentTypeError,
            starsmashertools.lib.units.Units,
            (None),
        )
        
        def f(val):
            units['x'] = val
        self.assertRaises(
            Exception,
            f,
            (0),
        )

if __name__ == "__main__":
    unittest.main()
