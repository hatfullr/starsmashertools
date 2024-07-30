import starsmashertools.helpers.fortran
import unittest
import os
import basetest

directory = os.path.join(os.path.dirname(__file__), 'fortran')

class TestFortran(basetest.BaseTest):
    def test_strip(self):
        f = '\n'.join("""
      abc          ! 0
      a &          ! 1
      b ! comment
      a &          ! 2
             b
      a            ! 3
c comment
         & b
      a            ! 4
        b          ! 5
      a&           ! 6
      b
      a&           ! 7
        b
c###################
      a            ! 8
     $   b
      a            ! 9
     0b            ! 10
      a            ! 11
     0     b       ! 12
      a            ! 13
     1 b
********************
      a            ! 14
      0   b
      a            ! 15
          0   b 
!!!!!!!!!!!!!!!!!!!!
      "!"          ! 16
      '!'          ! 17
""".split('\n')[1:])
        expected = [
            'abc',       # 0
            'a b',       # 1
            'a b',       # 2
            'a b',       # 3
            'a',         # 4
            'b',         # 5
            'ab',        # 6
            'a  b',      # 7
            #######
            'a b',       # 8
            'a',         # 9
            'b',         # 10
            'a',         # 11
            'b',         # 12
            'a b',       # 13
            #******
            'a   b',     # 14
            'a   b',     # 15
            #!!!!!!
            '"!"',       # 16
            "'!'",       # 17
        ]
        
        i = 0
        for line in starsmashertools.helpers.fortran.strip(f).split('\n'):
            self.assertEqual(line, expected[i], msg = i)
            i += 1
        self.assertEqual(i, len(expected))


    def test_get_objects(self):
        expected_names = ['checkpt', 'dump', 'output', 'enout', 'duout']
        
        ffile = starsmashertools.helpers.fortran.FortranFile(
            os.path.join(directory, 'output.f')
        )
        objects = list(ffile.get_objects())
        self.assertEqual(len(objects), len(expected_names))
        self.assertEqual(objects[0].name, 'checkpt')
        self.assertEqual(objects[1].name, 'dump')
        self.assertEqual(objects[2].name, 'output')
        self.assertEqual(objects[3].name, 'enout')
        self.assertEqual(objects[4].name, 'duout')


class TestObject(basetest.BaseTest):
    def test_init(self):
        path = os.path.join(directory, 'output.f')
        with open(path, 'r') as f:
            content = []
            for line in f:
                content += [line]
                if line == '      end\n': break
        content = '\n'.join(content)
        content = starsmashertools.helpers.fortran.strip(content)
        obj = starsmashertools.helpers.fortran.Object(content, path)

        self.assertEqual(obj.name, 'checkpt')
        self.assertEqual(obj.kind, 'subroutine')

        expected_body = """if (mod(nit,nitch).eq.0) then
mylength=n_upper-n_lower+1
call mpi_allgatherv(mpi_in_place, mylength, mpi_double_precision, rho, recvcounts,displs, mpi_double_precision, mpi_comm_world, ierr)
call mpi_allgatherv(mpi_in_place, mylength, mpi_double_precision, divv, recvcounts,displs, mpi_double_precision, mpi_comm_world, ierr)
if(myrank.ne.0) return
call cpu_time(time1)
write (69,*) 'checkpt: writing local checkpt file at nit=', nit
open(12,file='restartrad.sph',form='unformatted',err=100)
call dump(12)
close (12)
call cpu_time(time2)
seconds = time2-time1
write (6,*) 'restartrad:',seconds,'s'
endif
return
stop 'checkpt: error opening unit ???'"""
        
        self.assertEqual(obj.body, expected_body)

        expected_header = """include 'mpif.h'
integer nitch,itype
parameter (nitch=1000)
real*8 divv(nmax)
common/commdivv/divv
integer mylength, ierr"""
        self.assertEqual(obj.header, expected_header)

    def test_variables(self):
        path = os.path.join(directory, 'output.f')
        with open(path, 'r') as f:
            content = []
            for line in f:
                content += [line]
                if line == '      end\n': break
        content = '\n'.join(content)
        content = starsmashertools.helpers.fortran.strip(content)
        obj = starsmashertools.helpers.fortran.Object(content, path)

        self.assertEqual(obj._variables, None)

        expected = [
            starsmashertools.helpers.fortran.Variable('nitch', 'integer', 4),
            starsmashertools.helpers.fortran.Variable('itype', 'integer', 4),
            starsmashertools.helpers.fortran.Variable('divv', 'real', 4),
            starsmashertools.helpers.fortran.Variable('mylength', 'integer', 4),
            starsmashertools.helpers.fortran.Variable('ierr', 'integer', 4),
        ]
        variables = obj.variables
        self.assertEqual(len(variables), len(expected))
        for variable, exp in zip(variables, expected):
            self.assertEqual(variable, exp, msg = "'%s' != '%s'" % (variable, exp))
        
        
        
if __name__ == "__main__":
    unittest.main(failfast=True)

