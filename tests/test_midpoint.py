import unittest
import starsmashertools.helpers.midpoint

class TestMidpoint(unittest.TestCase):
    def test(self):
        objects = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        m = starsmashertools.helpers.midpoint.Midpoint(objects)
        m.set_criteria(
            lambda val: val < 8,
            lambda val: val == 8,
            lambda val: val > 8,
        )
        self.assertEqual(m.get(favor='low'), 8)
        self.assertEqual(m.get(favor='mid'), 8)
        self.assertEqual(m.get(favor='high'), 8)
        
        m.set_criteria(
            lambda val: val < 8.5,
            lambda val: val == 8.5,
            lambda val: val > 8.5,
        )
        self.assertEqual(m.get(favor='low'), 8)
        self.assertEqual(m.get(favor='mid'), 8)
        self.assertEqual(m.get(favor='high'), 9)

        objects = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        m.set_criteria(
            lambda val: val < 2,
            lambda val: val == 2,
            lambda val: val > 2,
        )
        self.assertEqual(m.get(favor='low'), 2)
        self.assertEqual(m.get(favor='mid'), 2)
        self.assertEqual(m.get(favor='high'), 2)


        objects = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
        self.assertEqual(m.get(favor='low'), 2)
        self.assertEqual(m.get(favor='mid'), 2)
        self.assertEqual(m.get(favor='high'), 2)

        m.set_criteria(
            lambda val: val < 5,
            lambda val: val == 5,
            lambda val: val > 5,
        )
        self.assertEqual(m.get(favor='low'), 5)
        self.assertEqual(m.get(favor='mid'), 5)
        self.assertEqual(m.get(favor='high'), 5)
        

        class Number(object):
            def __init__(self, value):
                self.value = value

        m.objects = [Number(0), Number(1), Number(2), Number(3), Number(4)]
        m.set_criteria(
            lambda o: o.value < 2,
            lambda o: o.value == 2,
            lambda o: o.value > 2,
        )
        result = m.get()
        self.assertIs(result, m.objects[2])
        self.assertEqual(result.value, 2)

        
        
if __name__ == "__main__":
    unittest.main(failfast=True)
