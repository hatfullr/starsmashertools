import starsmashertools.lib.units
import starsmashertools
import starsmashertools.helpers.argumentenforcer
import unittest

class TestUnits(unittest.TestCase):
    def test_labels(self):
        l0 = starsmashertools.lib.units.Unit.Label('cm')
        l1 = starsmashertools.lib.units.Unit.Label('cm')
        self.assertEqual(l0, l1)
        L = l0 * l1
        self.assertEqual(L.long, "cm*cm")
        self.assertNotEqual(l0, L)
        self.assertNotEqual(l1, L)
        
        l1 = starsmashertools.lib.units.Unit.Label('s')
        L = l0 * l1
        self.assertEqual(L.long, "cm*s")

        l0 = starsmashertools.lib.units.Unit.Label('cm/cm')
        l0.simplify()
        self.assertEqual(l0.long, '')

        l0 = starsmashertools.lib.units.Unit.Label('cm*cm/cm*cm')
        l0.simplify()
        self.assertEqual(l0.long, '')

        l0 = starsmashertools.lib.units.Unit.Label('cm*cm/cm*s')
        l0.simplify()
        self.assertEqual(l0.long, 'cm/s')

        l0 = starsmashertools.lib.units.Unit.Label('cm*g*s/g*s')
        l0.simplify()
        self.assertEqual(l0.long, 'cm')

        l0 = starsmashertools.lib.units.Unit.Label('cm')
        l1 = starsmashertools.lib.units.Unit.Label('s')
        L = l0 / l1
        self.assertEqual(L.long, 'cm/s')
        L = l1 / l0
        self.assertEqual(L.long, 's/cm')

        l1 = starsmashertools.lib.units.Unit.Label('cm')
        L = l0 / l1
        self.assertEqual(L.long, '')

        l0 = starsmashertools.lib.units.Unit.Label('cm*g*s/cm*cm*cm*cm')
        l1 = starsmashertools.lib.units.Unit.Label('g*s/cm')
        L = l0 / l1
        self.assertEqual(L.long, '1/cm*cm')
        L = L * l1
        self.assertEqual(L.long, 'g*s/cm*cm*cm')

        
        l0 = starsmashertools.lib.units.Unit.Label('cm')
        L = 1 / l0
        self.assertEqual(L.long, '1/cm')
        self.assertEqual(L.short, '1/cm')

        l0 = starsmashertools.lib.units.Unit.Label('cm*g*g/s*s')
        self.assertEqual(l0.short, 'erg')

        l0 = starsmashertools.lib.units.Unit.Label('cm*g/s*s')
        self.assertEqual(l0.short, 'erg/g')

        l0 = starsmashertools.lib.units.Unit.Label('cm*g*g/s*s*s')
        self.assertEqual(l0.short, 'erg/s')

        l0 = starsmashertools.lib.units.Unit.Label('cm/s')
        l1 = starsmashertools.lib.units.Unit.Label('g*g/s')
        L = l0 * l1
        self.assertEqual(L.short, 'erg')

    def test_unit(self):
        u0 = starsmashertools.lib.units.Unit(1.0, 'cm')
        u1 = starsmashertools.lib.units.Unit(1.0, 's')
        u2 = starsmashertools.lib.units.Unit(1.0, 'cm/s')
        U = u0 / u1
        self.assertEqual(U, u2)
        self.assertEqual(U.label, 'cm/s')

        u3 = u0**2
        self.assertEqual(u3.label, 'cm*cm')

        u4 = u0**(-2)
        self.assertEqual(u4.label, '1/cm*cm')

        self.assertEqual((u4 * u0**4).label, u3.label)

        u3 = u2**2
        self.assertEqual(u3.label, 'cm*cm/s*s')

        u3 = (u2**2)**0.5
        self.assertEqual(u3.label, 'cm/s')

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
