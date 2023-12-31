import starsmashertools.lib.units
import starsmashertools
import starsmashertools.preferences
import starsmashertools.helpers.argumentenforcer
import unittest
import numpy as np

class TestUnits(unittest.TestCase):
    def test_labels(self):
        l = starsmashertools.lib.units.Unit.Label("cm")
        self.assertTrue(l.is_compatible('m'))
        l = starsmashertools.lib.units.Unit.Label("cm/s")
        self.assertTrue(l.is_compatible('m/hr'))
        l = starsmashertools.lib.units.Unit.Label("yr/cm*cm")
        self.assertTrue(l.is_compatible('s/km*m'))
        
        l = starsmashertools.lib.units.Unit.Label('cm*s/cm')
        l.simplify()
        self.assertEqual(str(l), 's')

        l = starsmashertools.lib.units.Unit.Label('s*g*cm')
        l.organize()
        self.assertEqual(str(l), 'cm*g*s')
        l = starsmashertools.lib.units.Unit.Label('1/s*g*cm')
        l.organize()
        self.assertEqual(str(l), '1/cm*g*s')
        
        
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
        
        
        l0 = starsmashertools.lib.units.Unit.Label('cm/s')
        l0 = l0.convert('cm', 'm')
        self.assertEqual(l0.short, 'm/s')
        self.assertEqual(l0.long, 'm/s')
        
        
        
        l = starsmashertools.lib.units.Unit.Label('cm')**2
        self.assertEqual(l, 'cm*cm')
        
        l = starsmashertools.lib.units.Unit.Label('cm')**(-2)
        self.assertEqual(l, '1/cm*cm')

        l = starsmashertools.lib.units.Unit.Label('cm')**(-2)
        l *= starsmashertools.lib.units.Unit.Label('cm')**3
        self.assertEqual(l, 'cm')
        
        l = starsmashertools.lib.units.Unit.Label('cm/s')**2
        self.assertEqual(l, 'cm*cm/s*s')
        
        l = l**(0.5)
        self.assertEqual(l, 'cm/s')


        l = starsmashertools.lib.units.Unit.Label('cm*g*g/s*s')
        l = l.convert('cm', 'm')
        self.assertEqual(l, 'm*g*g/s*s')
        
        l = l.convert('m', 'cm')
        self.assertEqual(l.short, 'erg')

        l = starsmashertools.lib.units.Unit.Label('erg')
        self.assertEqual(l.short, 'erg')
        self.assertEqual(l.long, 'cm*g*g/s*s')

        

        

    def test_unit(self):
        u0 = starsmashertools.lib.units.Unit(1.0, 'cm')
        u1 = starsmashertools.lib.units.Unit(1.0, 's')
        u2 = starsmashertools.lib.units.Unit(1.0, 'cm/s')
        U = u0 / u1
        self.assertEqual(U, u2)
        self.assertEqual(U.label, 'cm/s')

        u = starsmashertools.lib.units.Unit(10, 'cm')
        u = u.get_base()
        self.assertAlmostEqual(float(u), 10)

        u = starsmashertools.lib.units.Unit(0.01, 'cm')
        u = u.get_base()
        self.assertAlmostEqual(float(u), 0.01)

        u = starsmashertools.lib.units.Unit(1, 'm')
        u = u.get_base()
        self.assertAlmostEqual(float(u), 100)

        
        u = starsmashertools.lib.units.Unit(1, 'km/hr')
        u = u.get_base()
        self.assertAlmostEqual(float(u), 1e5 / 3600)
        

        u = starsmashertools.lib.units.Unit(2, 'cm')
        u = u.convert('m')
        self.assertAlmostEqual(float(u), 0.02)
        u = u.convert('cm')
        self.assertAlmostEqual(float(u), 2)

        u = starsmashertools.lib.units.Unit(4.0, 'cm*cm')
        u = u.convert('m*m')
        self.assertAlmostEqual(float(u), 4. / (100 * 100))

        u = starsmashertools.lib.units.Unit(1.0, 'cm/s')
        u = u.convert('km/min')
        self.assertAlmostEqual(float(u), 1e-5 * 60)
        u = u.get_base()
        self.assertAlmostEqual(float(u), 1)
        self.assertEqual(u.label, 'cm/s')

        u = starsmashertools.lib.units.Unit(3600*24, 's')
        u = u.convert('day')
        self.assertAlmostEqual(float(u), 1)

        u = starsmashertools.lib.units.Unit(1.0, '')
        self.assertRaises(
            Exception,
            u.convert,
            ('km'),
        )


        u = starsmashertools.lib.units.Unit(3600*24, 's')
        factor = u.get_conversion_factor('day')
        u *= factor
        self.assertAlmostEqual(float(u), 1)
        
        u = starsmashertools.lib.units.Unit(86400, 'hr')
        u = u.auto()
        self.assertAlmostEqual(float(u), 86400 / (24*365.25))
        self.assertEqual(u.label, 'yr')

        u = starsmashertools.lib.units.Unit(86400, 's')
        u = u.auto()
        self.assertAlmostEqual(float(u), 24)
        self.assertEqual(u.label, 'hr')


        gravconst = starsmashertools.lib.units.Unit(6.67390e-08, 'cm*cm*cm/g*s*s')
        runit = starsmashertools.lib.units.Unit(6.9599e10, 'cm')
        munit = starsmashertools.lib.units.Unit(1.9891e33, 'g')

        simulation = starsmashertools.get_simulation('data')
        expected = simulation['runit']**3 / (6.67390e-08 * simulation['munit'])
        
        u = runit**3 / (gravconst * munit)
        self.assertAlmostEqual(float(u), expected)
        self.assertEqual(u.label, 's*s')
        u = np.sqrt(u)
        self.assertEqual(u.label, 's')
        self.assertAlmostEqual(float(u), np.sqrt(expected))
        

    def test_units(self):
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

        #print(simulation['runit']**3 / (simulation.units.gravconst * simulation['munit']))
        #expected = np.sqrt(simulation['runit']**3 / (simulation.units.gravconst * simulation['munit']))
        #self.assertAlmostEqual(float(simulation.units['t']), expected)
        #u = simulation.units['t'].convert('day')
        #self.assertAlmostEqual(float(u), expected / (3600 * 24))

        

if __name__ == "__main__":
    unittest.main(failfast=True)
