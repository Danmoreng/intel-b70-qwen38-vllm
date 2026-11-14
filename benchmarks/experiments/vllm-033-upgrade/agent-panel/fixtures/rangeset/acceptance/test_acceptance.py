import unittest
from rangeset import RangeSet
class Acceptance(unittest.TestCase):
 def test_constructor_union(self):self.assertEqual(RangeSet([(8,10),(-5,-1),(1,4),(4,8),(2,3),(10,10)]).ranges(),[(-5,-1),(1,10)])
 def test_remove_many(self):
  r=RangeSet([(0,5),(10,20),(30,40)]);r.remove(3,35);self.assertEqual(r.ranges(),[(0,3),(35,40)])
 def test_touch_and_empty(self):
  r=RangeSet([(0,5)]);r.remove(5,10);r.remove(2,2);r.add(7,7);self.assertEqual(r.ranges(),[(0,5)]);self.assertEqual(r.clip(3,3),[])
 def test_half_open(self):
  r=RangeSet([(-3,0),(2,5)]);self.assertEqual([r.contains(x) for x in [-4,-3,-1,0,1,2,4,5]],[False,True,True,False,False,True,True,False])
 def test_snapshots_and_input(self):
  initial=[[0,5]];r=RangeSet(initial);initial[0][1]=100;self.assertEqual(r.ranges(),[(0,5)]);snapshot=r.ranges();snapshot.append((99,100));self.assertEqual(r.ranges(),[(0,5)])
 def test_clip_nonmutating(self):
  r=RangeSet([(0,5),(10,20)]);clipped=r.clip(3,12);self.assertEqual(clipped,[(3,5),(10,12)]);clipped.clear();self.assertEqual(r.ranges(),[(0,5),(10,20)])
 def test_invalid_atomic(self):
  r=RangeSet([(0,10)])
  for method in [r.add,r.remove,r.clip]:
   for args,error in [((4,3),ValueError),((True,4),TypeError),((0,4.0),TypeError)]:
    with self.assertRaises(error):method(*args)
    self.assertEqual(r.ranges(),[(0,10)])
  with self.assertRaises(TypeError):r.contains(False)
 def test_large_and_bridge(self):
  limit=10**18;r=RangeSet([(-limit,limit)]);r.remove(-3,4);self.assertEqual(r.ranges(),[(-limit,-3),(4,limit)]);r.add(-3,4);self.assertEqual(r.ranges(),[(-limit,limit)])
