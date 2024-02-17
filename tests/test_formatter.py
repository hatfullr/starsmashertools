import unittest
import basetest
import starsmashertools.helpers.formatter

expected = """Output('out000.sph')
      ntot =            1038       nnopt =              27      nrelax =               1               
      nout =               0         nit =               1         ngr =               3               
  ncooling =               0        erad =   0.0000000E+00 cm*cm/s*s                                   
         t =   1.2681773E-03 day           dt =   8.4564990E-04 day        dtout =   4.4267382E+00 hr  
        tf =   1.8444743E+01 day       trelax =   2.6560429E+31 min   tjumpahead =   2.6560429E+31 min 

     alpha =   1.0000000E+00
      beta =   2.0000000E+00
 displacex =   0.0000000E+00 cm
 displacey =   0.0000000E+00 cm
 displacez =   0.0000000E+00 cm
       hco =   6.9599000E+08 m
    hfloor =   0.0000000E+00 cm
       nav =               3
 ndisplace =               0
    omega2 =   0.0000000E+00 1/s*s
      sep0 =   1.3919800E+13 m"""

class TestFormatter(basetest.BaseTest):
    def setUp(self):
        import os
        curdir = os.path.dirname(__file__)
        self.simdir = os.path.join(curdir, 'data')
        
    def test_format_output(self):
        import starsmashertools
        formatter = starsmashertools.helpers.formatter.Formatter('cli')
        simulation = starsmashertools.get_simulation(self.simdir)
        result = formatter.format_output(simulation.get_output(0))
        if result != expected:
            print("Result=\n"+result)
            print('---- Expected=\n'+expected)
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main(failfast=True)


