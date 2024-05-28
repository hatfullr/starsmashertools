"""
To properly test the flux calculations, we need to:
   1) Test on both optically thin and optically thick particles
   2) Test on particles with temperatures within and out of the dust regime
   3) Test various image resolutions
Each test needs to ensure that the output is exactly matching with the expected
outcomes from the original flux code.
"""

import unittest
import os
import starsmashertools
import starsmashertools.lib.flux
import numpy as np
import basetest
import collections
import copy
import warnings

curdir = os.getcwd()

eps = 1000 * np.finfo(float).eps

class TestFlux(basetest.BaseTest):
    # We expect these keys to be different
    ignore_keys = [
        'version',
        ('extra', 'particle_flux'),
        ('extra', 'particle_teff'),
        ('extra', 'particle_kappa'),
        ('extra', 'particle_tau'),
        ('extra', 'particle_rloc'),
        'units',
        ('image', 'Teff_cell'),
        ('image', 'kappa_cell'),
        ('image', 'rho_cell'),
        ('image', 'T_cell'),
    ]

    priority = [
        ('image', 'surf_d'),
        ('image', 'surf_id'),
        ('image', 'ray_n'),
    ]
    
    def setUp(self):
        directory = os.path.join(curdir, 'flux')
        self.simulation = starsmashertools.get_simulation(directory)

    def assertAlmostEqual(self, a, b, **kwargs):
        if not isinstance(a, np.ndarray): a = np.asarray([a], dtype = float)
        if not isinstance(b, np.ndarray): b = np.asarray([b], dtype = float)
        
        if a.shape != b.shape:
            m = "Array shapes %s != %s" % (a.shape, b.shape)
            if 'msg' in kwargs.keys(): m += ' : ' + str(kwargs['msg'])
            raise AssertionError(m)
        
        diff = np.abs(a - b)

        are_equal = np.logical_or(
            np.logical_or(
                np.logical_and(np.isneginf(a), np.isneginf(b)),
                np.logical_and(np.isposinf(a), np.isposinf(b)),
            ),
            np.logical_and(np.isnan(a), np.isnan(b)),
        )
        if are_equal.all(): return
        
        idx = np.logical_or(
            np.logical_or(
                a == 0,
                b == 0,
            ),
            diff < eps,
        )
        are_equal[idx] = True
        if are_equal.all(): return

        warnings.filterwarnings(action = 'ignore')
        idx = diff / (np.abs(a) + np.abs(b)) < eps
        warnings.resetwarnings()
        
        are_equal[idx] = True
        if are_equal.all(): return
        
        if len(a) == 1:
            m = "%s != %s" % (a[0], b[0])
            if 'msg' in kwargs.keys(): m += ' : ' + str(kwargs['msg'])
        else:
            m = "some elements in %s are not equal:\n" % kwargs.get('msg', 'a NumPy array')
            m += "indices = %s\n" % str(np.where(~are_equal))
            m += "%s\n%s" % (a[~are_equal], b[~are_equal])
        
        raise AssertionError('expected != actual ' + m)

    def get_baseline(self, name):
        filename = os.path.join(self.simulation.directory, name)
        arr = np.load(filename + '.npz')
        input_file = arr['input_file'][0]
        inputs = arr['inputs']
        outputs = arr['outputs']
        surf_br = arr['surf_br']
        surf_d = arr['surf_d']
        surf_id = arr['surf_id']
        surf_t = arr['surf_t']
        ray_n = arr['ray_n']
        particle_flux = arr['particle_flux']
        particle_teff = arr['particle_teff']
        particle_kappa = arr['particle_kappa']
        particle_tau = arr['particle_tau']
        particle_rloc = arr['particle_rloc']
        spectrum_output = arr['spectrum_output']

        input_file = os.path.join(self.simulation.directory, input_file)
        output = starsmashertools.lib.output.Output(input_file, self.simulation)

        n_res     = int(inputs[0])
        min_value = inputs[1]
        max_value = inputs[2]
        do_domain = int(inputs[7])
        theta     = inputs[8]
        phi       = inputs[9]
        tau_s     = inputs[10]
        tau_skip  = inputs[11]
        do_fluffy = int(inputs[12])
        do_dust   = int(inputs[13])

        time, ltot, l_v, l_r,l_i, teff_aver,teff_spectrum,teff_min_spectrum,teff_max_spectrum, rad_eff, xmin,xmax,ymin,ymax, dx,dy,flux_tot,ltot_spectrum = outputs
        
        extent = [xmin, xmax, ymin, ymax]

        kwargs = {
            'resolution' : (n_res - 1, n_res - 1),
            'extent' : extent,
            'theta' : theta,
            'phi' : phi,
            'fluffy' : bool(do_fluffy),
            'rays' : True, # Constant in flux_main.py code
            'tau_s' : tau_s,
            'tau_skip' : tau_skip,
            'dust_opacity' : None if not bool(do_dust) else 1.0, # Constant (kappa_dust)
            'dust_Trange' : [100.0, 1000.0], # Constant (T_dust_min, T_dust)
            'spectrum_size' : 1000, # Constant
            'lmax' : 3000, # Constant (lmax)
            'dl_il' : 3., # Constant
            'dl' : 10, # Constant (dl)
            'nbox' : 10, # Constant (nbox)
            'nboxh' : 5, # Constant (nboxh)
            'l_range' : (10, 3000), # Constant (l_min, l_max)
            'factor1' : 1.191045e-05, # Constant (factor1)
            'factor2' : 1.438728e+00, # Constant (factor2)
            'filters' : {
                'V' : [500, 600], # Constant
                'R' : [590, 730], # Constant
                'I' : [720, 880], # Constant
            },
        }

        return starsmashertools.lib.flux.FluxResult({
            'version' : starsmashertools.__version__,
            'output' : output.path,
            'kwargs' : kwargs,
            'simulation' : self.simulation.directory,
            'units' : {'length' : self.simulation.units.length},
            'time' : time,
            'image' : {
                'flux' : surf_br,
                'surf_d' : surf_d,
                'surf_id' : surf_id,
                'Teff_cell' : surf_t,
                'ray_n' : ray_n,
                'extent' : [xmin, xmax, ymin, ymax],
                'dx' : dx,
                'dy' : dy,
                'teff_aver' : teff_aver,
                'ltot' : ltot,
                'flux_tot' : flux_tot,
            },
            'spectrum' : {
                'ltot' : ltot_spectrum,
                'luminosities' : {
                    'V' : l_v,
                    'R' : l_r,
                    'I' : l_i,
                },
                'output' : spectrum_output,
                'teff' : teff_spectrum,
                'teff_min' : teff_min_spectrum,
                'teff_max' : teff_max_spectrum,
            },
            'extra' : {
                'particle_flux'  : particle_flux,
                'particle_teff'  : particle_teff,
                'particle_kappa' : particle_kappa,
                'particle_tau'   : particle_tau,
                'particle_rloc'  : particle_rloc,
            },
        })

    def get_finder(self, baseline, **kwargs):
        output = starsmashertools.lib.output.Output(
            baseline['output'], self.simulation,
        )
        kw = copy.deepcopy(baseline['kwargs'])
        kw.update(kwargs)
        return starsmashertools.lib.flux.FluxFinder(
            output,
            **kw
        )
    
    def compare(self, baseline_name):
        baseline = self.get_baseline(baseline_name)
        finder = self.get_finder(baseline)
        finder.get()
        result = finder.result

        # Validate the keys
        to_test = []
        for key, val in baseline.flowers():
            if key in TestFlux.ignore_keys: continue
            if key not in result.branches():
                raise KeyError("Missing key '%s' for comparison" % str(key))
            to_test += [key]

        def compare_simple(a, b, msg = None):
            if isinstance(a, (int, bool)):
                self.assertEqual(a, b, msg = msg)
            elif isinstance(a, (float, np.ndarray)):
                self.assertAlmostEqual(a, b, msg = msg)
            elif isinstance(a, (list, tuple)):
                if msg is None: msg = ''
                for i, (v1, v2) in enumerate(zip(a, b)):
                    compare_simple(v1, v2, msg = msg + '[%d]' % i)
            else:
                self.assertEqual(a, b, msg = msg)

        def compare2(key_path, tested_paths = []):
            if key_path in TestFlux.ignore_keys: return
            
            if key_path in tested_paths: return
            tested_paths += [key_path]
            
            base = baseline
            comp = result
            try:
                base = base[key_path]
                comp = comp[key_path]
            except KeyError as e:
                raise Exception("key_path = '%s'" % str(key_path)) from e
                
            if isinstance(base, dict):
                for key in base.keys():
                    compare(tuple(list(key_path) + key), tested_paths = tested_paths)
                return

            compare_simple(
                base, comp,
                msg = str(key_path) + ' (base, comparison)'
            )

        # Test the priority items first
        tested_paths = []
        for path in TestFlux.priority:
            compare2(path, tested_paths = tested_paths)
        
        # Run the actual comparison
        for key in to_test:
            compare2(key, tested_paths = tested_paths)


    def testGrid(self):
        def check(name, **kwargs):
            baseline = self.get_baseline(name)
            finder = self.get_finder(baseline, **kwargs)

            finder._init_grid()

            xmin, xmax, ymin, ymax = finder.extent
            xminb, xmaxb, yminb, ymaxb = baseline['kwargs']['extent']

            self.assertAlmostEqual(xmin, xminb, msg = 'xmin')
            self.assertAlmostEqual(xmax, xmaxb, msg = 'xmax')
            self.assertAlmostEqual(ymin, yminb, msg = 'ymin')
            self.assertAlmostEqual(ymax, ymaxb, msg = 'ymax')

            self.assertAlmostEqual(finder.dx, baseline['image']['dx'], msg = 'dx')
            self.assertAlmostEqual(finder.dy, baseline['image']['dy'], msg = 'dy')
        check('simple', extent = None)
        check('extents')

    def testParticles(self):
        baseline = self.get_baseline('simple')
        finder = self.get_finder(baseline, extent = None)
        ntot = finder.output['ntot']

        finder._init_particles()

        kappa = finder.output['popacity'] * float(self.simulation.units['popacity'])
        if finder.dust_opacity is not None:
            kappa = finder._apply_dust(kappa)

        comparisons = collections.OrderedDict()
        comparisons['particle_rloc']  = [finder._rloc[1:ntot-1], None]
        comparisons['particle_kappa'] = [       kappa[1:ntot-1], None]
        comparisons['particle_tau']   = [ finder._tau[1:ntot-1], None]
        comparisons['particle_flux']  = [finder._flux[1:ntot-1], None]
        comparisons['particle_teff']  = [finder._teff[1:ntot-1], None]

        for key in comparisons.keys():
            comparisons[key][1] = baseline['extra'][key][1:ntot-1]

        for key, (val1, val2) in comparisons.items():
            self.assertAlmostEqual(val1, val2, msg = key)
        
    def testSimple(self): self.compare('simple')
    def testAngles(self): self.compare('angles')
    def testExtents(self): self.compare('extents')
    def testDustless(self): self.compare('dustless')
    def testFluffy(self): self.compare('fluffy')
    def testtau_s(self): self.compare('tau_s')
    def testtau_skip_low(self): self.compare('tau_skip_low')
    def testtau_skip_high(self): self.compare('tau_skip_high')


class TestLoader(unittest.TestLoader, object):
    def getTestCaseNames(self, *args, **kwargs):
        return [
            'testGrid',
            'testParticles',
            'testSimple',
            'testAngles',
            'testExtents',
            'testDustless',
            'testFluffy',
            'testtau_s',
            'testtau_skip_low',
            'testtau_skip_high',
        ]

if __name__ == '__main__':
    import inspect
    import re
    
    comment_checks = [
        # Remove # comments first
        re.compile("(?<!['\"])#.*", flags = re.M),
        # Then remove block comments (which can be commented out by #)
        re.compile('(?<!\')(?<!\\\\)""".*?"""', flags = re.M | re.S),
        re.compile("(?<!\")(?<!\\\\)'''.*?'''", flags = re.M | re.S),
    ]
    
    src = inspect.getsource(starsmashertools.lib.flux)

    # Remove all comments
    for check in comment_checks:
        for match in check.findall(src):
            src = src.replace(match, '')
    
    if '@profile' in src:
        loader = TestLoader()
        suite = unittest.TestSuite()
        for name in loader.getTestCaseNames():
            suite.addTest(TestFlux(name))
        runner = unittest.TextTestRunner()
        runner.run(suite)
    else:
        # This is the normal method
        unittest.main(failfast=True, testLoader=TestLoader())
