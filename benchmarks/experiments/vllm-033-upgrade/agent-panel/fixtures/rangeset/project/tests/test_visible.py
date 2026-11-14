import unittest
from rangeset import RangeSet
class Visible(unittest.TestCase):
 def test_union(self):
  r=RangeSet([(5,10),(1,5)]);self.assertEqual(r.ranges(),[(1,10)])
 def test_subtract_splits(self):
  r=RangeSet([(0,10)]);r.remove(3,7);self.assertEqual(r.ranges(),[(0,3),(7,10)])
 def test_half_open(self):
  r=RangeSet([(1,5)]);self.assertTrue(r.contains(1));self.assertFalse(r.contains(5))
