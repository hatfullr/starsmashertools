import unittest
import os
import starsmashertools.lib.simulation
import starsmashertools.lib.output
import starsmashertools
import warnings
import basetest

# Test the functionality of the Simulation class
# This test depends on 

class TestRelaxation(basetest.BaseTest):
    def setUp(self):
        curdir = os.path.dirname(__file__)
        simdir = os.path.join(curdir, 'data')
        self.simulation = starsmashertools.get_simulation(simdir)

    def test_get_n(self):
        self.assertEqual(self.simulation.get_n(), 1038)

    def test_get_final_extents(self):
        extents = self.simulation.get_final_extents()
        self.assertAlmostEqual(
            float(extents.radius)/float(self.simulation.units.length),
            0.876398091009529,
        )

    def test_isPolytrope(self):
        self.assertTrue(self.simulation.isPolytrope)

    def test_get_children(self):
        warnings.filterwarnings(action='ignore')
        self.assertEqual(self.simulation.get_children(), [])
        warnings.resetwarnings()

    def test_get_binding_energy(self):
        import starsmashertools.lib.units
        import numpy as np
        
        output = self.simulation.get_output(0)
        Ebind = self.simulation.get_binding_energy(output)
        self.assertTrue(isinstance(Ebind, starsmashertools.lib.units.Unit))
        self.assertEqual(5.463207228488362e+46, Ebind.value)
        self.assertEqual(Ebind.label, 'cm*cm*g/s*s')

        G = float(self.simulation.units.gravconst)

        units = self.simulation.units
        IDs = np.arange(output['ntot'])
        
        xyz = np.column_stack((output['x'], output['y'], output['z']))
        r2 = np.sum(xyz**2, axis=-1)
        idx = np.argsort(r2)
        xyz = xyz[idx]
        r2 = r2[idx]
        m = output['am'][idx] * float(units['am'])
        mr = np.cumsum(m)
        r = np.sqrt(r2) * float(units.length)
        u = output['u'][idx] * float(units['u'])
        IDs = IDs[idx]
        
        for i, (mri, ri, ui, mi, ID) in enumerate(zip(mr, r, u, m, IDs)):
            Ebind = np.sum((G * mr[i:] / r[i:] - u[i:]) * m[i:])
            Ebind_found = float(self.simulation.get_binding_energy(
                output, mass_coordinate = mri,
            ))
            self.assertGreater(Ebind_found, 0)
            self.assertAlmostEqual(Ebind / Ebind_found, 1., msg = "At mass coordinate m(r) = %15.7E Msun, particle index %d" % (mri / float(units['am']), ID))

if __name__ == "__main__":
    unittest.main(failfast=True)
