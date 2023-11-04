import unittest
import os
import starsmashertools.lib.input



class TestInput(unittest.TestCase):
    def setUp(self):
        curdir = os.path.dirname(__file__)

        directory = os.path.join(curdir, 'data')
        directory_no_init = os.path.join(curdir, 'data_no_init')
        
        self.input = starsmashertools.lib.input.Input(directory)
        self.input_no_init = starsmashertools.lib.input.Input(directory_no_init)

    def test_src(self):
        self.assertEqual(
            self.input.src,
            os.path.realpath(os.path.join(self.input.directory, 'src')),
        )

        # We cannot locate the src directory if it is missing the init.f file
        with self.assertRaises(FileNotFoundError):
            self.input_no_init.src

    def test_get_init_file(self):
        self.input.get_init_file()
        with self.assertRaises(FileNotFoundError):
            self.input_no_init.get_init_file()

    def test_get_input_filename(self):
        self.assertEqual(
            self.input.get_input_filename(),
            'sph.input',
        )

    def test_initialize(self):
        self.assertFalse(self.input._initialized)
        self.input.initialize()
        self.assertTrue(self.input._initialized)

        with self.assertRaises(Exception, msg="Cannot initialize an Input object that is already initialized"):
            self.input.initialize()

    def test_values_check(self):
        expected = {
            'tf' : 1000,
            'dtout' : 10,
            'n' : 1000,
            'nnopt' : 27,
            'alpha' : 1,
            'beta' : 2,
            'nrelax' : 1,
            'sep0' : 20000,
            'bimpact' : 1.e30,
            'vinf2' : 1.e30,
            'equalmass' : 0,
            'treloff' : 1000,
            'nitpot' : 1,
            'tscanon' : 0,
            'sepfinal' : 1.e30,
            'nintvar' : 2,
            'ngravprocs' : -1,
            'qthreads' : 0,
            'gflag' : 1,
            'runit' : 6.9599e10,
            'munit' : 1.9891e33,
            'cn1' : 0.4,
            'cn2' : 0.06,
            'cn3' : 0.06,
            'cn4' : 1.e30,
            'cn5' : 0.02,
            'cn6' : 0.02,
            'cn7' : 4,
            'computeexclusivemode' : 0,
            'ppn' : 4,
            'nkernel' : 2,
            'neos' : 2,
            'starmass' : 0.16,

            # In init.f:
            'icosahedron_subdivisions' : 0,
            'cooling_type' : 0,
            'thetawindow' : 0.,
            'phiwindow' : 0.,
            'throwaway' : False,
            'starradius' : 1e0,
        }

        for key, val in expected.items():
            self.assertEqual(self.input[key], val)

if __name__ == "__main__":
    unittest.main(failfast=True)
