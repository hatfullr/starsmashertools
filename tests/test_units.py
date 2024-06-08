import starsmashertools.lib.units
import starsmashertools
import starsmashertools.helpers.argumentenforcer
import unittest
import numpy as np
import basetest
import math

class TestUnits(basetest.BaseTest):
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

        # Test label cancellation
        l1 = starsmashertools.lib.units.Unit.Label('cm')
        self.assertEqual((l0 / l1).long, '')

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

        l0 = starsmashertools.lib.units.Unit.Label('cm*cm*g/s*s')
        self.assertEqual(l0.short, 'erg')
        
        l0 = starsmashertools.lib.units.Unit.Label('cm*cm/s*s')
        self.assertEqual(l0.short, 'erg/g')
        
        l0 = starsmashertools.lib.units.Unit.Label('cm*cm*g/s*s*s')
        self.assertEqual(l0.short, 'erg/s')
        
        l0 = starsmashertools.lib.units.Unit.Label('cm*cm/s')
        l1 = starsmashertools.lib.units.Unit.Label('g/s')
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


        l = starsmashertools.lib.units.Unit.Label('cm*cm*g/s*s')
        l = l.convert('cm', 'm')
        self.assertEqual(l, 'm*m*g/s*s')
        
        l = l.convert('m', 'cm')
        self.assertEqual(l.short, 'erg')

        l = starsmashertools.lib.units.Unit.Label('erg')
        self.assertEqual(l.short, 'erg')
        self.assertEqual(l.long, 'cm*cm*g/s*s')

        # Ordering of labels should not matter.
        labels = [
            starsmashertools.lib.units.Unit.Label('cm*s*g'),
            starsmashertools.lib.units.Unit.Label('cm*g*s'),
            starsmashertools.lib.units.Unit.Label('g*cm*s'),
            starsmashertools.lib.units.Unit.Label('s*cm*g'),
            starsmashertools.lib.units.Unit.Label('cm*s*g'),
            starsmashertools.lib.units.Unit.Label('g*s*cm'),
        ]
        for label in labels:
            self.assertEqual(label.long, 'cm*g*s')
            for label2 in labels:
                self.assertEqual(label, label2)

    def test_conversions(self):
        # Simple conversions
        u = starsmashertools.lib.units.Unit(1, 's')
        u.convert('min')
        self.assertEqual(u.label, 's')
        self.assertEqual(u.value, 1)
        
        u = u.convert('min')
        self.assertEqual(u.label, 'min')
        self.assertAlmostEqual(u.value, 1./60)
        
        u = u.convert('s')
        self.assertEqual(u.label, 's')
        self.assertAlmostEqual(u.value, 1)
        
        with self.assertRaises(TypeError):
            u.convert(new_label = None, to = None)

        u = starsmashertools.lib.units.Unit(1, 'min')
        self.assertEqual(
            u.convert('s'),
            starsmashertools.lib.units.Unit(60, 's'),
            msg = 'conversion down',
        )
        u = starsmashertools.lib.units.Unit(1, 'km')
        self.assertEqual(
            u.convert('m'),
            starsmashertools.lib.units.Unit(1000, 'm'),
        )
            
        
        # A more complex conversion
        u = starsmashertools.lib.units.Unit(1, 'cm/s')
        u = u.convert('km/hr')
        self.assertEqual(u.label, 'km/hr')
        self.assertAlmostEqual(u.value, 1. / 1e5 * 3600)
        
        # Conversion of only specific units
        u = starsmashertools.lib.units.Unit(1, 'cm*cm/g*s')
        u = u.convert(to = ['km', 'hr'])
        self.assertEqual(u.label, 'km*km/g*hr')
        self.assertAlmostEqual(u.value, (1/1e5) * (1/1e5) / (1/3600))

        # Lsun conversion
        u = starsmashertools.lib.units.Unit(1, 'Lsun')
        base = u.get_base()
        self.assertEqual(base.label.short, 'erg/s')
        self.assertEqual(base.value, starsmashertools.lib.units.constants['Lsun'])

        # Dividing by the same base units should result in conversions, always
        # converting to the base
        u = starsmashertools.lib.units.Unit(1, 'cm')
        u2 = starsmashertools.lib.units.Unit(1, 'm')
        self.assertEqual(u/u2, starsmashertools.lib.units.Unit(0.01, 'cm'))
        self.assertEqual(u2/u, starsmashertools.lib.units.Unit(100, 'cm'))
    
    def test_integers(self):
        u = starsmashertools.lib.units.Unit(1, 's')

        u *= 10
        self.assertEqual(u, 10)
        self.assertEqual(u.label, 's')
        u /= 10
        self.assertEqual(u, 1 * 10 / 10)
        u -= 1
        self.assertEqual(u, 0)
        self.assertEqual(u.label, 's')
        u += 1
        self.assertEqual(u, 1)
        self.assertEqual(u.label, 's')

        u.value = 1
        self.assertEqual(u, 1)
        self.assertEqual(u.label, 's')

        u = 10 / u
        self.assertEqual(u, 10)
        self.assertEqual(u.label, '1/s')
        u.value = 1
        u = 10 * u
        self.assertEqual(u, 10)
        self.assertEqual(u.label, '1/s')
        u = 1 - u
        self.assertEqual(u, 1 - 10)
        self.assertEqual(u.label, '1/s')
        u = 1 + u
        self.assertEqual(u, 1 - 10 + 1)
        self.assertEqual(u.label, '1/s')

    def test_floats(self):
        u = starsmashertools.lib.units.Unit(1., 's')
        u *= 1.5
        self.assertEqual(u, 1. * 1.5)
        self.assertEqual(u.label, 's')
        u /= 3.
        self.assertEqual(u, 1.5 / 3.)
        u -= 0.1
        self.assertEqual(u, 1.5 / 3. - 0.1)
        self.assertEqual(u.label, 's')
        u += 0.1
        self.assertEqual(u, 1.5 / 3. - 0.1 + 0.1)
        self.assertEqual(u.label, 's')

        u.value = 2.
        self.assertEqual(u, 2.)
        self.assertEqual(u.label, 's')

        u = 1.5 / u
        self.assertEqual(u, 1.5 / 2.)
        self.assertEqual(u.label, '1/s')
        u.value = 1.
        u = 10.5 * u
        self.assertEqual(u, 10.5)
        self.assertEqual(u.label, '1/s')
        u = 1.5 - u
        self.assertEqual(u, 1.5 - 10.5)
        self.assertEqual(u.label, '1/s')
        u = 1.5 + u
        self.assertEqual(u, 1.5 - 10.5 + 1.5)
        self.assertEqual(u.label, '1/s')

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
        with self.assertRaises(starsmashertools.lib.units.Unit.InvalidLabelError):
            u.convert('km')
        
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


        # Test operations with same bases but different labels
        u1 = starsmashertools.lib.units.Unit(1, 'cm')
        u2 = starsmashertools.lib.units.Unit(1, 'm')
        self.assertEqual(
            u1 + u2,
            starsmashertools.lib.units.Unit(101, 'cm'),
            msg = '__add__',
        )
        self.assertEqual(
            u1 * u2,
            starsmashertools.lib.units.Unit(100, 'cm*cm'),
            msg = '__mul__',
        )
        u1 = starsmashertools.lib.units.Unit(1, 'm')
        u2 = starsmashertools.lib.units.Unit(1, 'km')
        self.assertEqual(
            u1 * u2,
            starsmashertools.lib.units.Unit(1000, 'm*m'),
            msg = '__mul__',
        )
        

        # Test some weird operators
        for i in range(0, 10):
            u = starsmashertools.lib.units.Unit(i, 's')
            for j in range(0, 10):
                if j != 0:
                    self.assertEqual(u % j, i % j, msg = '%d %d' % (i,j))
                    self.assertEqual((u % j).label, 's', msg = '%d %d' % (i,j))
                if i != 0:
                    self.assertEqual(j % u, j % i, msg = '%d %d' % (i,j))
                    self.assertEqual((j % u).label, 's', msg = '%d %d' % (i,j))
        u = starsmashertools.lib.units.Unit(1, 's')
        self.assertEqual(
            -starsmashertools.lib.units.Unit(1, 's'),
            starsmashertools.lib.units.Unit(-(1), 's'),
            msg = '__neg__',
        )
        self.assertEqual(
            +starsmashertools.lib.units.Unit(-1, 's'),
            starsmashertools.lib.units.Unit(+(-1), 's'),
            msg = '__pos__'
        )
        self.assertEqual(
            abs(starsmashertools.lib.units.Unit(-1, 's')),
            starsmashertools.lib.units.Unit(1, 's'),
            msg = '__abs__',
        )
        self.assertEqual(
            round(starsmashertools.lib.units.Unit(0.8, 's')),
            starsmashertools.lib.units.Unit(1, 's'),
            msg = '__round__'
        )
        self.assertEqual(
            math.trunc(starsmashertools.lib.units.Unit(0.8, 's')),
            starsmashertools.lib.units.Unit(0, 's'),
            msg = '__trunc__'
        )


    def test_units(self):
        import starsmashertools.helpers.readonlydict
        simulation = starsmashertools.get_simulation("data")
        units = starsmashertools.lib.units.Units(simulation)
        
        with self.assertRaises(starsmashertools.helpers.argumentenforcer.ArgumentTypeError):
            starsmashertools.lib.units.Units(None)

        
        with self.assertRaises(starsmashertools.helpers.readonlydict.ReadOnlyDict.EditError):
            units['x'] = 0

    def test_formatting(self):
        u = starsmashertools.lib.units.Unit(1000, 'day')
        s = '{:10.4f >4s}'.format(u)
        self.assertEqual(s, ' 1000.0000 day')
        s = '{:<10.4f >4s}'.format(u)
        self.assertEqual(s, '1000.0000  day')
        s = '{:5.1f >4s}'.format(u)
        self.assertEqual(s, '1000.0 day')


    def test_different_label_comparisons(self):
        # Equalities
        self.assertEqual(
            starsmashertools.lib.units.Unit(1, 's'),
            starsmashertools.lib.units.Unit(1, 's'),
        )
        self.assertNotEqual(
            starsmashertools.lib.units.Unit(1, 's'),
            starsmashertools.lib.units.Unit(2, 's'),
        )
        self.assertLessEqual(
            starsmashertools.lib.units.Unit(1, 's'),
            starsmashertools.lib.units.Unit(1, 's'),
        )
        self.assertGreaterEqual(
            starsmashertools.lib.units.Unit(1, 's'),
            starsmashertools.lib.units.Unit(1, 's'),
        )
        self.assertAlmostEqual(
            starsmashertools.lib.units.Unit(    1,   's'),
            starsmashertools.lib.units.Unit(1./60, 'min'),
        )
        self.assertAlmostEqual(
            starsmashertools.lib.units.Unit(      1,     's*s'),
            starsmashertools.lib.units.Unit(1./3600, 'min*min'),
        )

        # Inequalities
        self.assertLess(
            starsmashertools.lib.units.Unit(1, 's'),
            starsmashertools.lib.units.Unit(2, 's'),
        )
        self.assertGreater(
            starsmashertools.lib.units.Unit(2, 's'),
            starsmashertools.lib.units.Unit(1, 's'),
        )
        self.assertLess(
            starsmashertools.lib.units.Unit(    1,   's'),
            starsmashertools.lib.units.Unit(2./60, 'min'),
        )
        self.assertGreater(
            starsmashertools.lib.units.Unit(2./60, 'min'),
            starsmashertools.lib.units.Unit(    1,   's'),
        )

        # Error checking
        u1 = starsmashertools.lib.units.Unit(1, 's')
        u2 = starsmashertools.lib.units.Unit(1, 'g')
        with self.assertRaises(starsmashertools.lib.units.Unit.InvalidLabelError):
            u1 > u2
        with self.assertRaises(starsmashertools.lib.units.Unit.InvalidLabelError):
            u1 >= u2
        with self.assertRaises(starsmashertools.lib.units.Unit.InvalidLabelError):
            u1 < u2
        with self.assertRaises(starsmashertools.lib.units.Unit.InvalidLabelError):
            u1 <= u2
            
        
    def test_different_label_operations(self):
        s = starsmashertools.lib.units.Unit(1,'s')
        m = starsmashertools.lib.units.Unit(2./60,'min')
        g = starsmashertools.lib.units.Unit(1,'g')
        self.assertEqual(m - s, s)
        with self.assertRaises(starsmashertools.lib.units.Unit.InvalidLabelError):
            g - s
        
        self.assertEqual(s + m, 3 * s)
        with self.assertRaises(starsmashertools.lib.units.Unit.InvalidLabelError):
            g + s
        self.assertEqual(m // s, 2)
        self.assertEqual(s // m, 0)
        self.assertEqual(s % m, starsmashertools.lib.units.Unit(1, 's'))
        self.assertEqual(m % s, starsmashertools.lib.units.Unit(0, 'min'))


        
        

if __name__ == "__main__":
    unittest.main(failfast=True)
