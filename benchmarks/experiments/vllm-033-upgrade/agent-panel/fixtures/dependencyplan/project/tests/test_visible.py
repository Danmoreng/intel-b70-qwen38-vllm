import unittest
from buildplan import plan_targets,DependencyCycle
class Visible(unittest.TestCase):
 def test_simple(self):self.assertEqual(plan_targets({'app':['lib'],'lib':[]},['app']),['lib','app'])
 def test_global_ready_order(self):self.assertEqual(plan_targets({'a':['z'],'b':[],'z':[]},['a','b']),['b','z','a'])
 def test_duplicates(self):self.assertEqual(plan_targets({'app':['lib','lib'],'lib':[]},['app','app']),['lib','app'])
 def test_cycle(self):
  with self.assertRaises(DependencyCycle):plan_targets({'a':['b'],'b':['a']},['a'])
